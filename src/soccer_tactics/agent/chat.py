"""Local match Q&A orchestration with context-bound tools and citations."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from soccer_tactics.agent.ollama import OllamaChatClient
from soccer_tactics.agent.tools import AnalysisContext, call_tool, tool_specs


ANSWER_FORMAT = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["answered", "insufficient_evidence", "unsupported"]},
        "text": {"type": "string"},
        "citation_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["status", "text", "citation_ids"],
    "additionalProperties": False,
}

MAX_TOOL_ROUNDS = 4
MAX_TOOL_CALLS = 8

# A conservative language guard supplements the model's evidence instructions.
# Apply it to both questions and returned text: citations identify observations,
# but cannot validate a causal relationship between those observations.
_CAUSAL_LANGUAGE = re.compile(
    r"\b(?:cause[sd]?|causing|causal(?:ity|ly)?|causation)\b"
    r"|\b(?:lead(?:s|ing)?|led)\s+to\b"
    r"|\bresult(?:s|ed|ing)?\s+(?:in|from)\b"
    r"|\bbecause\b|\bdue\s+to\b|\bresponsible\s+for\b"
    r"|\battribut(?:e[sd]?|ing|able)\b", re.I,
)


@dataclass(frozen=True)
class ChatAnswer:
    status: str
    text: str
    citation_ids: list[str]


def _system_prompt(context: AnalysisContext) -> str:
    if context.is_idsse:
        capability = (
            "This IDSSE/DFL context supports factual source-event origin records only. "
            "It does not establish event-to-tracking synchronization, attack direction, "
            "pass destinations, successful progression, xT, or causation. Refuse questions "
            "that require those unavailable facts."
        )
    else:
        capability = (
            "Use only the selected team's existing Metrica match report and its evidence. "
            "Report measurements as patterns that may have contributed, never as proven causes. "
            "A single match does not establish causation."
        )
    limitations = "; ".join(context.limitations) or "No additional limitations were supplied."
    return (
        "You are Touchline, a cautious soccer match analysis assistant. "
        f"The active context is source={context.source}, match={context.match_id}, "
        f"team={context.selected_team}. {capability} "
        "Use the supplied analytics tools to retrieve match facts. Do not invent numbers, "
        "players, events, or tactical causes. Return only the required JSON object with status, "
        "text, and citation_ids. For any answer containing match-derived facts, include one or "
        "more citation IDs returned by the tools. If the evidence is inadequate, use "
        "insufficient_evidence; if the source cannot support the question, use unsupported. "
        f"Known limitations: {limitations}"
    )


def _ollama_tools(context: AnalysisContext) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["input_schema"],
            },
        }
        for spec in tool_specs(context)
    ]


def _tool_call_data(call: Mapping[str, Any]) -> tuple[str, dict]:
    if not isinstance(call, Mapping):
        raise ValueError("tool call must be an object")
    function = call.get("function", {})
    if not isinstance(function, Mapping):
        raise ValueError("tool call function must be an object")
    name = function.get("name")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(name, str) or not isinstance(arguments, Mapping):
        raise ValueError("tool call must contain a function name and object arguments")
    return name, dict(arguments)


def _withhold(message: str = "I can't verify that answer from the selected match evidence.") -> ChatAnswer:
    return ChatAnswer("withheld", message, [])


def _causal_abstention() -> ChatAnswer:
    return ChatAnswer("unsupported", "The selected match evidence cannot establish causation or guarantee a different result.", [])


def _report_facts(context: AnalysisContext) -> tuple[dict[str, Any], set[str]]:
    """Make the selected report's small, trusted fact set visible to local models.

    Some local models return a final response instead of requesting tools. Supplying
    these already-computed report facts keeps common numeric questions grounded;
    source lookups and other on-demand queries still use the context-bound tools.
    """
    if not context.is_metrica:
        return {}, set()
    summary = call_tool(context, "match_summary", {})
    summary.pop("citation_ids", None)
    findings = call_tool(context, "team_findings", {})
    players = call_tool(context, "player_metrics", {"n": 10})
    # Keep the initial context concise while retaining a moment for each player.
    player_ids: set[str] = set()
    for row in players.get("rows", []):
        row["citation_ids"] = row.get("citation_ids", [])[:3]
        player_ids.update(row["citation_ids"])
    players["citation_ids"] = sorted(player_ids)
    finding_ids = {
        citation_id for finding in findings.get("findings", [])
        for citation_id in finding.get("evidence_ids", [])
    }
    v4 = {name: call_tool(context, name, {}) for name in
          ("sequence_summary", "defensive_summary", "shot_summary")}
    for name, result in v4.items():
        # Sequence tables can contain hundreds of rows; expose counts initially.
        if name == "sequence_summary":
            result.pop("possessions", None)
            result.pop("transitions", None)
            result["citation_ids"] = result["citation_ids"][:3]
        # Shot/defense rows retain their source IDs; every displayed ID must
        # remain citeable, even when it lies beyond the first three events.
        player_ids.update(result["citation_ids"])
    return {
        **v4,
        "match_summary": summary,
        "team_findings": findings,
        "player_metrics": players,
    }, finding_ids | player_ids


def _parse_answer(content: Any, allowed_ids: set[str]) -> ChatAnswer:
    if not isinstance(content, str):
        return _withhold()
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return _withhold()
    if not isinstance(payload, dict):
        return _withhold()
    if set(payload) != {"status", "text", "citation_ids"}:
        return _withhold()
    status, text, citation_ids = payload["status"], payload["text"], payload["citation_ids"]
    if not isinstance(status, str) or status not in {"answered", "insufficient_evidence", "unsupported"}:
        return _withhold()
    if not isinstance(text, str) or not text.strip():
        return _withhold()
    if (not isinstance(citation_ids, list)
            or not all(isinstance(value, str) for value in citation_ids)
            or len(set(citation_ids)) != len(citation_ids)):
        return _withhold()
    if any(value not in allowed_ids for value in citation_ids):
        return _withhold()
    if status == "answered" and not citation_ids:
        return _withhold()
    if _CAUSAL_LANGUAGE.search(text):
        return _causal_abstention()
    return ChatAnswer(status, text.strip(), citation_ids)


def answer_question(
    context: AnalysisContext,
    question: str,
    history: Iterable[Mapping[str, str]] = (),
    client: Any | None = None,
) -> ChatAnswer:
    """Answer one question using only the current match/team context.

    ``history`` contains prior user/assistant text for the active selection.
    Ollama failures are allowed to propagate so the dashboard can show its
    actionable local-server or model message.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question cannot be empty")
    if _CAUSAL_LANGUAGE.search(question):
        return _causal_abstention()
    client = client or OllamaChatClient()
    messages = [{"role": "system", "content": _system_prompt(context)}]
    report_facts, verified_ids = _report_facts(context)
    if report_facts:
        messages.append({
            "role": "system",
            "content": (
                "Verified selected-report facts (JSON): "
                + json.dumps(report_facts, ensure_ascii=False, allow_nan=False)
                + " Use these values exactly for match measurements. Cite only evidence IDs "
                  "included here or retrieved through an analytics tool."
            ),
        })
    for message in history:
        if (not isinstance(message, Mapping)
                or message.get("role") not in {"user", "assistant"}
                or not isinstance(message.get("content"), str)):
            raise ValueError("chat history must contain user and assistant text messages")
        messages.append({"role": message["role"], "content": message["content"]})
    messages.append({"role": "user", "content": question.strip()})

    tools = _ollama_tools(context)
    tool_call_count = 0
    for _ in range(MAX_TOOL_ROUNDS):
        # Do not combine Ollama's tool-call phase with constrained JSON output.
        # Qwen3 otherwise tends to skip tools and invent report values/IDs.
        response = client.chat(messages, tools=tools)
        model_message = response.get("message") if isinstance(response, Mapping) else None
        if not isinstance(model_message, Mapping):
            return _withhold()
        calls = model_message.get("tool_calls") or []
        if not isinstance(calls, list):
            return _withhold()
        if not calls:
            preliminary = _parse_answer(model_message.get("content"), verified_ids)
            if preliminary.status != "withheld":
                return preliminary
            break
        tool_call_count += len(calls)
        if tool_call_count > MAX_TOOL_CALLS:
            return _withhold("I couldn't complete a bounded evidence lookup for that question.")
        messages.append({
            "role": "assistant",
            "content": model_message.get("content", ""),
            "tool_calls": calls,
        })
        for call in calls:
            try:
                name, arguments = _tool_call_data(call)
                result = call_tool(context, name, arguments)
                content = json.dumps(result, ensure_ascii=False, allow_nan=False)
                returned_ids = result.get("citation_ids", [])
                if isinstance(returned_ids, list):
                    verified_ids.update(value for value in returned_ids if isinstance(value, str))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                content = json.dumps({"error": str(exc)[:240]}, ensure_ascii=False)
                function = call.get("function") if isinstance(call, Mapping) else None
                name = function.get("name", "unknown") if isinstance(function, Mapping) else "unknown"
            messages.append({"role": "tool", "tool_name": name, "content": content})
    else:
        return _withhold("I couldn't complete the evidence lookup within the allowed number of steps.")

    messages.append({
        "role": "system",
        "content": "Return the final answer now using the required JSON schema. "
                   "Cite only source IDs present in the verified report facts or tool results.",
    })
    final_response = client.chat(messages, format=ANSWER_FORMAT)
    final_message = final_response.get("message") if isinstance(final_response, Mapping) else None
    if not isinstance(final_message, Mapping):
        return _withhold()
    return _parse_answer(final_message.get("content"), verified_ids)


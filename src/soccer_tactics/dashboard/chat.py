"""Selection-bound chat helpers for the optional local dashboard assistant."""
from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Callable

import pandas as pd

from soccer_tactics.agent.chat import ChatAnswer, answer_question
from soccer_tactics.agent.tools import AnalysisContext


IDSSE_LIMITATIONS = (
    "IDSSE context contains source event origins only.",
    "Event-to-tracking synchronization and attack direction are unavailable.",
    "Pass destinations, progression, expected threat (xT), and causation are unavailable.",
)


def build_metrica_context(report, team_names=None) -> AnalysisContext:
    """Limit chat to selected-match passes and cited report finding evidence."""
    passes = report.passes
    if {"event_id", "match_id", "team"}.issubset(passes.columns):
        selected = passes[
            (passes.match_id.astype(str) == str(report.match_id))
            & (passes.team == report.team)
        ]
        pass_ids = selected.event_id.astype(str).tolist()
    else:
        pass_ids = []
    finding_ids = [str(evidence.event_id) for finding in report.findings
                   for evidence in finding.evidence
                   if str(evidence.match_id) == str(report.match_id)]
    citation_ids = tuple(dict.fromkeys([*pass_ids, *finding_ids]))
    return AnalysisContext(
        source="Metrica", match_id=str(report.match_id), selected_team=report.team,
        report=report, team_names=team_names or {}, limitations=tuple(report.limitations),
        allowed_citation_ids=citation_ids,
    )


def build_idsse_context(origins: pd.DataFrame, match_id: str, selected_team: str,
                        team_names=None) -> AnalysisContext:
    """Canonicalize only loaded IDSSE event origins for bounded lookup and citations."""
    names = dict(team_names or {})
    canonical = origins.copy(deep=True)
    if "team" not in canonical:
        raise ValueError("IDSSE origin rows require a team column")
    if not {"match_id", "origin_id"}.issubset(canonical.columns):
        raise ValueError("IDSSE origin rows require source match and origin IDs")
    labels = {str(label): side for side, label in names.items()}
    labels.update({"home": "home", "away": "away"})
    canonical["team"] = canonical.team.map(lambda value: labels.get(str(value), value))
    if not canonical.empty and not canonical.team.isin({"home", "away"}).all():
        raise ValueError("IDSSE origin rows must identify the canonical home or away team")
    citation_ids = tuple(
        canonical.loc[canonical.team == selected_team, "origin_id"].astype(str).tolist()
    ) if "origin_id" in canonical else ()
    return AnalysisContext(
        source="IDSSE", match_id=str(match_id), selected_team=selected_team,
        event_origins=canonical, team_names=names, limitations=IDSSE_LIMITATIONS,
        allowed_citation_ids=citation_ids,
    )


def ensure_chat_state(state: MutableMapping[str, Any], selection: tuple[str, str, str]) -> list[dict[str, str]]:
    """Clear transcript and draft question whenever source/match/team changes."""
    active = tuple(str(item) for item in selection)
    if state.get("_touchline_chat_selection") != active:
        state["_touchline_chat_selection"] = active
        state["_touchline_chat_history"] = []
        state["touchline_chat_question"] = ""
    history = state.setdefault("_touchline_chat_history", [])
    return history


def citation_description(context: AnalysisContext, citation_id: str) -> str | None:
    """Resolve an allowed answer citation to its displayed source moment or origin."""
    if citation_id not in context.allowed_citation_ids:
        return None
    if context.is_metrica:
        rows = context.report.passes
        if not {"event_id", "match_id", "team"}.issubset(rows.columns):
            return None
        rows = rows[
            (rows.event_id.astype(str) == citation_id)
            & (rows.match_id.astype(str) == str(context.match_id))
        ]
        if not rows.empty:
            row = rows.iloc[0]
            return (f"P{row.period} · source {float(row.time_s) / 60:.2f} min · "
                    f"{row.player_id} · {row.event} · frame {row.frame}")
        for finding in context.report.findings:
            for evidence in finding.evidence:
                if (str(evidence.event_id) == citation_id
                        and str(evidence.match_id) == str(context.match_id)):
                    return (f"P{evidence.period} · source {float(evidence.time_s) / 60:.2f} min · "
                            f"{evidence.player_id} · {evidence.event} · frame {evidence.frame}")
        return None
    rows = context.event_origins
    if rows is None or "origin_id" not in rows:
        return None
    matches = rows[rows.origin_id.astype(str) == citation_id]
    if matches.empty:
        return None
    row = matches.iloc[0]
    timestamp = row.get("source_timestamp", "time unavailable")
    gameclock = row.get("gameclock_s")
    time_text = f"{float(gameclock):.1f}s source game clock" if pd.notna(gameclock) else str(timestamp)
    origin = "origin unavailable"
    if pd.notna(row.get("source_x_m")) and pd.notna(row.get("source_y_m")):
        origin = f"source {float(row.source_x_m):.1f}, {float(row.source_y_m):.1f} m"
    return f"{citation_id} · P{row.period} · {time_text} · {row.source_code} · {origin}"


def render_chat_panel(st, context: AnalysisContext, selection: tuple[str, str, str],
                      *, answerer: Callable = answer_question) -> None:
    """Render chat without allowing local inference failures to interrupt analysis."""
    history = ensure_chat_state(st.session_state, selection)
    with st.container(border=True):
        st.subheader("Ask the local match assistant")
        if context.is_idsse:
            st.caption("IDSSE answers are limited to factual source event origins.")
            st.caption("Unavailable: tracking synchronization, attack direction, pass destinations, progression, xT, and causation.")
        question = st.text_input("Question about this match", key="touchline_chat_question")
        if st.button("Ask Touchline", key="touchline_chat_submit", disabled=not bool(question.strip())):
            prior_history = list(history)
            history.append({"role": "user", "content": question.strip()})
            try:
                answer: ChatAnswer = answerer(context, question, prior_history)
            except Exception as exc:
                st.error(f"Local assistant unavailable: {exc}")
                st.caption("Start Ollama with `ollama serve`, then install the configured model with `ollama pull qwen3:8b` (or set TOUCHLINE_OLLAMA_MODEL). Analysis remains available while chat is offline.")
            else:
                valid_ids = [value for value in answer.citation_ids
                             if value in context.allowed_citation_ids
                             and citation_description(context, value) is not None]
                status = answer.status
                text = answer.text
                if status == "answered" and not valid_ids:
                    status = "withheld"
                    text = "I can't verify that answer from the selected match evidence."
                history.append({"role": "assistant", "content": text,
                                "status": status, "citation_ids": valid_ids})
        for message in history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
                continue
            st.markdown(f"**Touchline · {message.get('status', 'answered')}**")
            st.write(message["content"])
            citation_ids = [value for value in message.get("citation_ids", [])
                            if value in context.allowed_citation_ids]
            if citation_ids:
                st.caption("Sources")
                for citation_id in citation_ids:
                    detail = citation_description(context, citation_id)
                    if detail is not None:
                        st.caption(detail)


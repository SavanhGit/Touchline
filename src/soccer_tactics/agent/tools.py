"""Context-bound, source-limited tools for local match question answering."""
from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd

from soccer_tactics.analytics.report import MatchReport


@dataclass(frozen=True)
class AnalysisContext:
    """All data a single assistant session may inspect.

    DataFrames are copied at construction, while the frozen wrapper prevents
    replacing the selected match/team or source during a session.
    """

    source: str
    match_id: str
    selected_team: str
    report: MatchReport | None = None
    event_origins: pd.DataFrame | None = None
    team_names: Mapping[str, str] | None = None
    limitations: tuple[str, ...] = ()
    allowed_citation_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.source or not self.match_id:
            raise ValueError("source and match ID are required")
        if self.selected_team not in {"home", "away"}:
            raise ValueError("team must be home or away")
        names = dict(self.team_names or {})
        object.__setattr__(self, "team_names", MappingProxyType(names))
        object.__setattr__(self, "limitations", tuple(str(v) for v in self.limitations))
        citations = tuple(dict.fromkeys(str(v) for v in self.allowed_citation_ids))
        object.__setattr__(self, "allowed_citation_ids", citations)
        if self.is_metrica:
            if self.report is None or self.event_origins is not None:
                raise ValueError("Metrica context requires a report and no origin frame")
            if str(self.report.match_id) != str(self.match_id):
                raise ValueError("report match does not match context match")
            if self.report.team != self.selected_team:
                raise ValueError("report team does not match selected team")
            evidence = _report_evidence(self.report)
            unknown = set(citations) - set(evidence)
            if unknown:
                raise ValueError("allowed citation IDs must refer to this report's match evidence")
        elif self.is_idsse:
            if self.report is not None or self.event_origins is None:
                raise ValueError("IDSSE context requires event origins and no Metrica report")
            origins = self.event_origins.copy(deep=True).reset_index(drop=True)
            if origins.empty and citations:
                raise ValueError("allowed citation IDs must refer to this match's origin rows")
            if not origins.empty:
                if "match_id" not in origins or set(origins.match_id.astype(str)) != {str(self.match_id)}:
                    raise ValueError("origin rows do not match context match")
                if "team" not in origins or not origins.team.isin(["home", "away"]).all():
                    raise ValueError("origin rows require home or away team")
                if "origin_id" not in origins:
                    raise ValueError("origin rows require committed source origin IDs")
                origin_ids = origins.origin_id.astype(str)
                expected = rf"IDSSE:{re.escape(str(self.match_id))}:\d{{6}}"
                if origin_ids.duplicated().any() or not origin_ids.str.fullmatch(expected).all():
                    raise ValueError("origin rows require unique IDs for this source match")
                selected_origin_ids = set(origins.loc[
                    origins.team == self.selected_team, "origin_id"
                ].astype(str))
                unknown = set(citations) - selected_origin_ids
                if unknown:
                    raise ValueError("allowed citation IDs must refer to selected-team origin rows")
            object.__setattr__(self, "event_origins", origins)
        else:
            raise ValueError(f"unsupported match source: {self.source}")

    @property
    def is_idsse(self) -> bool:
        return self.source.casefold().startswith("idsse")

    @property
    def is_metrica(self) -> bool:
        return self.source.casefold().startswith("metrica")


def _report_evidence(report: MatchReport) -> dict[str, Any]:
    """Index finding evidence and source-addressable report passes in scope."""
    rows = {}
    for finding in report.findings:
        for evidence in finding.evidence:
            # A selected-team finding can be supported by an opponent event,
            # such as a shot or final-third entry. Scope it to this report's
            # match; the report itself determines why the event is relevant.
            if str(evidence.match_id) == str(report.match_id):
                rows[str(evidence.event_id)] = evidence
    if {"event_id", "match_id", "team"}.issubset(report.passes.columns):
        fields = ("event_id", "match_id", "period", "frame", "time_s", "team",
                  "player_id", "event", "x0", "y0", "x1", "y1")
        for row in report.passes.to_dict("records"):
            citation_id = str(row["event_id"])
            if (str(row["match_id"]) == str(report.match_id) and row["team"] == report.team
                    and citation_id not in rows):
                rows[citation_id] = {key: row[key] for key in fields if key in row}
    for evidence in report.event_evidence:
        if str(evidence.match_id) == str(report.match_id) and evidence.team == report.team:
            rows[str(evidence.event_id)] = evidence
    return rows


def _json_value(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        value = vars(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _unique_citations(context: AnalysisContext) -> list[str]:
    available = set(_report_evidence(context.report)) if context.report else set()
    return [cid for cid in context.allowed_citation_ids if cid in available]


def _spec(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object", "properties": properties, "required": required or [],
            "additionalProperties": False,
        },
    }


_METRICA_SPECS = [
    _spec("match_summary", "Summarize the selected match using its existing analysis report.", {}),
    _spec("team_findings", "Return the selected team's report findings and cited evidence IDs.", {}),
    _spec("player_metrics", "Read player metrics already present in the selected team's report.", {
        "metric": {"type": "string", "description": "Optional metric column to sort by."},
        "n": {"type": "integer", "description": "Maximum rows to return (1-20)."},
    }),
    _spec("evidence_lookup", "Look up one cited event from the selected team's report.", {
        "citation_id": {"type": "string"},
    }, ["citation_id"]),
    _spec("sequence_summary", "Read observed possessions and transitions within 10 seconds; no causation.", {}),
    _spec("defensive_summary", "Read recognized defensive actions and observed-exposure rates.", {}),
    _spec("shot_summary", "Read shot locations and outcomes by attacking channel; no shot quality estimate.", {}),
]
_IDSSE_SPECS = [
    _spec("origin_lookup", "Look up factual source event origins for the selected team.", {
        "event": {"type": "string"},
        "player_id": {"type": "string"},
        "period": {"type": "integer"},
        "n": {"type": "integer", "description": "Maximum rows to return (1-50)."},
    }),
]


def tool_specs(context: AnalysisContext) -> list[dict]:
    """Return only the tool definitions available to this context."""
    if context.is_idsse:
        return deepcopy(_IDSSE_SPECS)
    specs = deepcopy(_METRICA_SPECS)
    numeric_metrics = sorted(
        column for column in context.report.players.select_dtypes(include="number").columns
        if column not in {"match_id"}
    )
    specs[2]["input_schema"]["properties"]["metric"]["enum"] = numeric_metrics
    return specs


def _validate_args(name: str, arguments: Mapping[str, Any], specs: list[dict]) -> dict:
    if not isinstance(arguments, Mapping):
        raise ValueError("tool arguments must be an object")
    spec = next((item for item in specs if item["name"] == name), None)
    if spec is None:
        raise KeyError(f"unknown analytics tool: {name}")
    props = spec["input_schema"]["properties"]
    extra = set(arguments) - set(props)
    missing = set(spec["input_schema"]["required"]) - set(arguments)
    if extra or missing:
        raise ValueError(f"invalid arguments; unexpected={sorted(extra)}, missing={sorted(missing)}")
    result = dict(arguments)
    for key, value in result.items():
        if key == "n" or key == "period":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
        elif not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        elif "enum" in props[key] and value not in props[key]["enum"]:
            raise ValueError(f"{key} must be one of the supported values")
    return result


def call_tool(context: AnalysisContext, name: str, arguments: Mapping[str, Any]) -> dict:
    """Dispatch one tool against the supplied context, with source citations."""
    args = _validate_args(name, arguments, tool_specs(context))
    if context.is_idsse:
        return _origin_lookup(context, args)
    if name in {"sequence_summary", "defensive_summary", "shot_summary"}:
        return _v4_summary(context, name)
    if name == "match_summary":
        report = context.report
        return _json_value({
            "source": context.source, "match_id": context.match_id,
            "team": context.selected_team,
            "team_name": context.team_names.get(context.selected_team),
            "score": report.score, "outcome": report.outcome,
            "comparisons": report.comparisons[
                report.comparisons.team == context.selected_team
            ].to_dict("records") if "team" in report.comparisons else [],
            "limitations": list(context.limitations or report.limitations),
            "citation_ids": _unique_citations(context),
        })
    if name == "team_findings":
        items = []
        allowed = set(context.allowed_citation_ids)
        for finding in context.report.findings:
            evidence_ids = [str(row.event_id) for row in finding.evidence
                            if str(row.match_id) == str(context.match_id)
                            and str(row.event_id) in allowed]
            items.append({"category": finding.category, "title": finding.title,
                          "observation": finding.observation, "suggestion": finding.suggestion,
                          "tradeoff": finding.tradeoff, "metrics": finding.metrics,
                          "evidence_ids": evidence_ids})
        return _json_value({"findings": items, "citation_ids": _unique_citations(context)})
    if name == "player_metrics":
        n = args.get("n", 10)
        if not 1 <= n <= 20:
            raise ValueError("n must be between 1 and 20")
        rows = context.report.players
        rows = rows[(rows.team == context.selected_team)
                    & (rows.match_id.astype(str) == str(context.match_id))] \
            if {"team", "match_id"}.issubset(rows.columns) else rows.iloc[0:0]
        metric = args.get("metric")
        if metric is not None:
            if not isinstance(metric, str) or metric not in rows.columns:
                raise ValueError("metric must name a report player metric")
            rows = rows.sort_values(metric, ascending=False, na_position="last")
        allowed = set(context.allowed_citation_ids)
        pass_rows = context.report.passes
        player_citations = {}
        if {"event_id", "match_id", "team", "player_id"}.issubset(pass_rows.columns):
            selected_passes = pass_rows[
                (pass_rows.match_id.astype(str) == str(context.match_id))
                & (pass_rows.team == context.selected_team)
            ]
            for player_id, player_rows in selected_passes.groupby("player_id", dropna=False):
                player_citations[str(player_id)] = [
                    str(value) for value in player_rows.event_id if str(value) in allowed
                ]
        output_rows = []
        citations = []
        for row in rows.head(n).to_dict("records"):
            row_citations = player_citations.get(str(row.get("player_id")), [])
            row["citation_ids"] = row_citations
            citations.extend(row_citations)
            output_rows.append(row)
        return _json_value({"rows": output_rows,
                            "citation_ids": list(dict.fromkeys(citations))})
    if name == "evidence_lookup":
        citation_id = args["citation_id"]
        if not isinstance(citation_id, str) or citation_id not in context.allowed_citation_ids:
            raise ValueError("citation ID is not allowed in this context")
        evidence = _report_evidence(context.report).get(citation_id)
        if evidence is None:
            raise ValueError("citation ID is not included in this match report")
        return _json_value({"evidence": vars(evidence) if not isinstance(evidence, dict) else evidence,
                            "citation_ids": [citation_id]})
    raise KeyError(f"unknown analytics tool: {name}")


_ORIGIN_FIELDS = ("period", "team", "event", "success", "player_id", "source_code",
                  "source_timestamp", "gameclock_s", "source_x_m", "source_y_m", "x_m", "y_m")


def _origin_lookup(context: AnalysisContext, args: dict) -> dict:
    n = args.get("n", 20)
    if not 1 <= n <= 50:
        raise ValueError("n must be between 1 and 50")
    frame = context.event_origins
    rows = frame[frame.team == context.selected_team]
    for key in ("event", "player_id", "period"):
        if key in args:
            if key not in rows.columns:
                raise ValueError(f"origin data has no {key} field")
            rows = rows[rows[key] == args[key]]
    allowed = set(context.allowed_citation_ids)
    output = []
    for index, row in rows.head(n).iterrows():
        citation_id = str(row["origin_id"])
        if citation_id not in allowed:
            continue
        item = {key: row[key] for key in _ORIGIN_FIELDS if key in row.index}
        item["citation_id"] = citation_id
        output.append(_json_value(item))
    return {"origins": output, "citation_ids": [row["citation_id"] for row in output]}


V4_LIMITATIONS = {
    "sequence_summary": "Observed event sequences only, without inferred carries. Transitions stop at possession end, period end, or 10 seconds; they do not establish that a turnover caused a shot.",
    "defensive_summary": "Only explicitly recognized source defensive actions are counted. Rates use observed tracking minutes, not full-match exposure; missing or ambiguous actions are excluded.",
    "shot_summary": "Raw modeling-shot and goal counts in the team's attacking direction; penalties are excluded. Sparse samples do not establish shot quality or statistical significance; no learned shot-quality estimate is fitted.",
}


def _v4_summary(context: AnalysisContext, name: str) -> dict:
    fields = {"sequence_summary": ("possessions", "transitions"),
              "defensive_summary": ("defensive_actions",),
              "shot_summary": ("shot_summaries",)}[name]
    allowed = set(_unique_citations(context))
    output = {"match_id": context.match_id, "team": context.selected_team,
              "limitations": [V4_LIMITATIONS[name]], "sample_counts": {}}
    citations = []
    for field in fields:
        frame = getattr(context.report, field)
        rows = frame[frame.team == context.selected_team] if "team" in frame else frame.iloc[:0]
        if "match_id" in rows:
            rows = rows[rows.match_id.astype(str) == str(context.match_id)]
        records = []
        for row in rows.to_dict("records"):
            row.pop("evidence", None)
            ids = [str(cid) for cid in row.get("event_ids", []) if str(cid) in allowed]
            row["event_ids"] = ids
            for key in ("start_event_id", "end_event_id", "source_turnover_event_id"):
                if key in row and str(row[key]) not in allowed:
                    row[key] = None
                elif key in row:
                    citations.append(str(row[key]))
            citations.extend(ids)
            records.append(row)
        output[field] = records
        count_column = {"defensive_actions": "total_actions", "shot_summaries": "shots"}.get(field)
        output["sample_counts"][field] = int(rows[count_column].sum()) if count_column and count_column in rows else len(rows)
    output["citation_ids"] = list(dict.fromkeys(citations))
    return _json_value(output)

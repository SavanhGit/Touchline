import pandas as pd
import json

from soccer_tactics.agent.chat import answer_question
from soccer_tactics.agent.tools import AnalysisContext
from soccer_tactics.analytics.report import Evidence, Finding, MatchReport


class SequenceClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, tools=None, *, format=None):
        self.calls.append({"messages": messages, "tools": tools, "format": format})
        return self.replies.pop(0)


def _context():
    evidence = Evidence("m1:1:12:0", "m1", 1, 12, 12., "home", "p1", "pass", 0., 0., 1., 1.)
    finding = Finding("progression", "Limited entries", "1 of 5 passes entered", "Review support",
                      "Could increase turnovers", {"numerator": 4, "denominator": 5}, [evidence])
    report = MatchReport("m1", "home", {"home": 0, "away": 1}, "loss", pd.DataFrame(),
                         pd.DataFrame([{"team": "home", "passes": 5}]), pd.DataFrame(),
                         pd.DataFrame(), pd.DataFrame(), [finding], {"source": "Metrica"}, [])
    return AnalysisContext("Metrica", "m1", "home", report=report,
                           allowed_citation_ids=(evidence.event_id,))


def test_answer_question_runs_context_tool_then_validates_structured_citations():
    client = SequenceClient([
        {"message": {"role": "assistant", "tool_calls": [{"function": {
            "name": "team_findings", "arguments": {},
        }}]}},
        {"message": {"role": "assistant", "content": '{"status":"answered","text":"Four of five passes did not enter.","citation_ids":["m1:1:12:0"]}'}},
    ])

    answer = answer_question(_context(), "Where did progression stall?", [], client)

    assert answer.status == "answered"
    assert answer.citation_ids == ["m1:1:12:0"]
    assert [tool["function"]["name"] for tool in client.calls[0]["tools"]] == [
        "match_summary", "team_findings", "player_metrics", "evidence_lookup",
    ]
    assert client.calls[0]["format"]["required"] == ["status", "text", "citation_ids"]
    assert client.calls[1]["messages"][-1]["role"] == "tool"
    assert "m1:1:12:0" in client.calls[1]["messages"][-1]["content"]


def test_answer_question_withholds_unverifiable_citations():
    client = SequenceClient([{"message": {"content":
        '{"status":"answered","text":"They lost because of player X.","citation_ids":["foreign:1"]}'
    }}])

    answer = answer_question(_context(), "Why did we lose?", [], client)

    assert answer.status == "withheld"
    assert answer.citation_ids == []
    assert "verify" in answer.text.lower()


def test_answer_question_allows_careful_unsupported_response_without_citations():
    client = SequenceClient([{"message": {"content":
        '{"status":"unsupported","text":"This data cannot establish that cause.","citation_ids":[]}'
    }}])

    answer = answer_question(_context(), "Which defender caused the goal?", [], client)

    assert answer.status == "unsupported"
    assert answer.citation_ids == []


def test_answer_question_withholds_malformed_status_types():
    for malformed_status in ([], {}, None):
        client = SequenceClient([{"message": {"content": json.dumps({
            "status": malformed_status, "text": "Unsupported format.", "citation_ids": [],
        })}}])
        answer = answer_question(_context(), "What happened?", [], client)
        assert answer.status == "withheld"


def test_idsse_chat_only_receives_origin_lookup_tool():
    context = AnalysisContext("IDSSE/DFL", "i1", "home", event_origins=pd.DataFrame([{
        "match_id": "i1", "period": 1, "team": "home", "event": "pass", "success": True,
        "player_id": "p1", "source_code": "Play_Pass", "origin_id": "IDSSE:i1:000001",
        "source_timestamp": "2020-01-01T00:00:00Z", "gameclock_s": 12.,
        "source_x_m": 10., "source_y_m": 20., "x_m": -40., "y_m": -14.,
    }]), allowed_citation_ids=("IDSSE:i1:000001",))
    client = SequenceClient([{"message": {"content":
        '{"status":"answered","text":"One pass origin is recorded.","citation_ids":["IDSSE:i1:000001"]}'
    }}])

    answer = answer_question(context, "What is recorded?", [], client)

    assert answer.status == "answered"
    assert [tool["function"]["name"] for tool in client.calls[0]["tools"]] == ["origin_lookup"]


def test_idsse_context_rejects_opponent_citation_allowlist():
    origins = pd.DataFrame([
        {"match_id": "i1", "period": 1, "team": "home", "event": "pass",
         "origin_id": "IDSSE:i1:000001"},
        {"match_id": "i1", "period": 1, "team": "away", "event": "shot",
         "origin_id": "IDSSE:i1:000002"},
    ])
    import pytest
    with pytest.raises(ValueError, match="selected-team origin"):
        AnalysisContext("IDSSE", "i1", "home", event_origins=origins,
                        allowed_citation_ids=("IDSSE:i1:000002",))


def test_malformed_tool_call_is_rejected_without_crashing_or_crossing_context():
    client = SequenceClient([
        {"message": {"tool_calls": [{"function": "not an object"}]}},
        {"message": {"content":
            '{"status":"unsupported","text":"No supported evidence.","citation_ids":[]}'
        }},
    ])

    answer = answer_question(_context(), "Can you track the live match?", [], client)

    assert answer.status == "unsupported"
    assert '"error"' in client.calls[1]["messages"][-1]["content"]


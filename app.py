"""Launch with: python -m streamlit run app.py."""
from pathlib import Path
from math import isfinite
from numbers import Real

import pandas as pd
import plotly.express as px
import streamlit as st

from soccer_tactics.analytics.report import analyze_catalogued_metrica
from soccer_tactics.dashboard.charts import evidence_pitch, pass_map, shot_map, snapshot, tracking_window
from soccer_tactics.agent.chat import answer_question
from soccer_tactics.agent.tools import call_tool
from soccer_tactics.dashboard.chat import build_idsse_context, build_metrica_context, render_chat_panel
from soccer_tactics.loaders import load_metrica_bundle, load_idsse_bundle
from soccer_tactics.loaders.catalog import list_match_catalogue

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw" / "metrica"
IDSSE_DATA = ROOT / "data" / "raw" / "idsse"
IDSSE_MANIFEST = ROOT / "docs" / "validation" / "idsse-v1.json"

st.set_page_config(page_title="Touchline | Soccer Match Analysis", page_icon="⚽", layout="wide")


@st.cache_resource(max_entries=3)
def load_bundle(game, revision):
    # Shared source bundles are read-only. Analytics build their own derived tables.
    return load_metrica_bundle(DATA, game)


@st.cache_data(max_entries=6, show_spinner="Building the evidence report…")
def load_report(game, team, revision):
    return analyze_catalogued_metrica(
        game, team, load_catalogue(revision), lambda g: load_bundle(g, revision))


@st.cache_data(max_entries=2)
def load_catalogue(revision):
    return list_match_catalogue(DATA, IDSSE_DATA, IDSSE_MANIFEST)


@st.cache_resource(max_entries=2)
def load_idsse_selected(match_id, source_path, revision):
    return load_idsse_bundle(source_path, match_id.removeprefix("idsse_"))


@st.cache_data(max_entries=4)
def load_idsse_origin_report(match_id, source_path, revision):
    bundle = load_idsse_selected(match_id, source_path, revision)
    columns = ["origin_id", "match_id", "period", "team", "event", "success",
               "player_id", "source_code", "source_timestamp", "gameclock_s",
               "source_x_m", "source_y_m", "x_m", "y_m"]
    origins = bundle.event_origins.reindex(columns=columns).copy()
    origins["team"] = origins.team.map(bundle.team_names)
    return origins, bundle.score, bundle.team_names, bundle.metadata, bundle.quality


def display_value(value, *, rate=False):
    """Format missing measurements without implying zero or converting NaN to int."""
    if pd.isna(value) or (isinstance(value, Real) and not isfinite(value)):
        return "Unavailable"
    if isinstance(value, Real):
        if rate:
            return f"{value:.1%}"
        return str(int(value)) if value % 1 == 0 else f"{value:.3f}"
    return str(value)


def main():
    st.markdown("""<style>
    .block-container {padding-top:3.5rem; max-width:1450px}
    h1 {letter-spacing:-.045em} h3 {letter-spacing:-.02em}
    [data-testid="stMetric"] {background:#142b25; border:1px solid #28473d; border-radius:12px; padding:16px}
    [data-testid="stMetricValue"] {font-size:1.5rem}
    [data-testid="stSidebar"] {border-right:1px solid #28473d}
    .eyebrow {color:#69d6ab;font-size:.78rem;letter-spacing:.18em;font-weight:700}
    </style>""", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("## ⚽ TOUCHLINE")
        st.caption("SOCCER MATCH ANALYSIS")
        st.divider()
        st.markdown("### Match room")
        source_files = [DATA / "data" / f"Sample_Game_{g}" / f"Sample_Game_{g}_Raw{suffix}.csv"
                        for g in (1, 2)
                        for suffix in ("EventsData", "TrackingData_Home_Team", "TrackingData_Away_Team")]
        source_files += [DATA / "data" / "Sample_Game_3" / f"Sample_Game_3_{suffix}"
                         for suffix in ("metadata.xml", "tracking.txt", "events.json")]
        catalogue_revision = tuple((str(p.resolve()), p.stat().st_mtime_ns, p.stat().st_size)
                                   for p in source_files + sorted(IDSSE_DATA.glob("*.xml")) + [IDSSE_MANIFEST]
                                   if p.is_file())
        entries = load_catalogue(catalogue_revision)
        ready_entries = [entry for entry in entries if entry.ready]
        if not ready_entries:
            st.warning("No validated source matches are available. Restore Metrica sample data or install and validate IDSSE data.")
            st.code(str(DATA))
            for entry in entries:
                if entry.reason:
                    st.caption(entry.reason)
            st.code("git clone --depth 1 https://github.com/metrica-sports/sample-data data/raw/metrica")
            st.stop()
        option_ids = [entry.match_id for entry in ready_entries]
        previous = st.session_state.get("_previous_match_id")
        if previous is not None and previous in option_ids and st.session_state.get("selected_match") != previous:
            st.session_state["selected_team"] = "away"
        selected_id = st.selectbox("Match", option_ids,
                                   format_func=lambda match_id: next(e.display_name for e in ready_entries if e.match_id == match_id),
                                   key="selected_match")
        st.session_state["_previous_match_id"] = selected_id
        selected = next(entry for entry in ready_entries if entry.match_id == selected_id)
        team = st.selectbox("Analyze team", ["away", "home"],
                            format_func=lambda side: selected.team_names.get(side, side.capitalize()),
                            key="selected_team")
        st.divider()
        st.caption("POST-MATCH ANALYSIS")
        st.write("Explore the evidence. Question the explanation.")
        st.caption("Data: Metrica Sports. Teams and players are anonymized. This is a portfolio research tool.")
    engine_files = [Path(__file__)] + sorted((ROOT / "src" / "soccer_tactics").rglob("*.py"))
    revision = tuple((str(f.resolve()), f.stat().st_mtime_ns, f.stat().st_size)
                     for f in source_files + engine_files if f.is_file())
    if selected.provider == "IDSSE":
        try:
            origins, score, team_names, metadata, quality = load_idsse_origin_report(
                selected.match_id, selected.source_path, catalogue_revision)
        except (ValueError, FileNotFoundError, ImportError) as exc:
            st.error(f"Match analysis unavailable: {exc}")
            st.stop()
        st.markdown('<div class="eyebrow">SOURCE EVENT ORIGINS</div>', unsafe_allow_html=True)
        st.title("What the source records show")
        score_text = f"Home {score['home']} – {score['away']} Away" if score is not None else "Score unknown"
        st.markdown(f"**{selected.display_name} · {score_text}** / {team_names[team]} perspective")
        st.caption("IDSSE/DFL source events are shown in the unrotated source frame. These records describe event origins only.")
        st.warning("Pass endpoints, attack direction, and event-to-tracking synchronization are unavailable. No tracking playback, xT, progression, or pass routes are shown.")
        st.subheader("Source event origins")
        display_columns = ["period", "team", "event", "source_code", "source_timestamp",
                           "gameclock_s", "source_x_m", "source_y_m"]
        st.dataframe(origins.loc[origins.team == team_names[team], display_columns],
                     hide_index=True, width="stretch")
        st.caption("Times and event codes come from the source event records; coordinates remain in the source frame.")
        st.subheader("Source and quality")
        st.write("Source: IDSSE / Deutsche Fußball Liga (DFL). Bassek et al., *Scientific Data* 12, 195 (2025).")
        st.json({"metadata": metadata, "quality": quality})
        chat_context = build_idsse_context(
            origins, selected.match_id.removeprefix("idsse_"), team, team_names
        )
        render_chat_panel(
            st, chat_context, (selected.provider, selected.match_id, team),
            answerer=answer_question,
        )
        return
    game = int(selected.match_id.removeprefix("metrica_"))
    revision = catalogue_revision
    try:
        bundle = load_bundle(game, revision)
        report = load_report(game, team, revision)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        st.error(f"Match analysis unavailable: {exc}")
        st.stop()
    st.markdown('<div class="eyebrow">THE MATCH, EXPLAINED THROUGH EVIDENCE</div>', unsafe_allow_html=True)
    st.title("What shaped the result?")
    score_text = f"Home {report.score['home']} – {report.score['away']} Away" if report.score is not None else "Score unknown"
    st.markdown(f"**Sample game {game} · {score_text}**  /  {team.capitalize()} perspective · {report.outcome.upper()}")
    st.caption("Patterns that may have contributed, with adjustments to investigate. These findings do not establish causation.")
    overview, risk, players_tab, evidence_tab, v4_tab = st.tabs(["Overview", "Risk & progression", "Players", "Evidence", "Sequences & shots"])
    own = report.comparisons.set_index("team").loc[team]
    with overview:
        cols = st.columns(4)
        cols[0].metric("Buildup losses", f"{int(own.buildup_losses)} / {int(own.buildup_passes)}")
        cols[1].metric("Completed passes", f"{int(own.completed)} / {int(own.passes)}")
        cols[2].metric("Final-third entries", int(own.final_third_entries))
        cols[3].metric("Shots*", int(own.modeling_shots))
        st.caption("*Modeling shots exclude penalties; the score includes them. Entries count completed passes, not carries.")
        st.subheader("Patterns worth reviewing")
        if not report.findings:
            st.info("No pattern meets the evidence safeguards for this match and team.")
        for i, finding in enumerate(report.findings, 1):
            with st.container(border=True):
                st.markdown(f"### {i:02d} / {finding.title}")
                st.write(finding.observation)
                st.markdown(f"**Adjustment to investigate:** {finding.suggestion}")
                st.caption(f"Tradeoff: {finding.tradeoff}")
                st.caption(f"{len(finding.evidence)} supporting moments available in Evidence.")
        left, right = st.columns([1, 1.3])
        with left:
            st.subheader("Goal timeline")
            if report.score is None:
                st.info("Complete scoring records unavailable.")
            elif report.goals.empty:
                st.write("No goals recorded.")
            else:
                for goal in report.goals.itertuples():
                    st.write(f"P{goal.period} · source {goal.time_s / 60:.2f} min — **{goal.team.capitalize()}** · {goal.player_id}" + (" · penalty" if goal.penalty else ""))
            st.caption("Source clock in decimal minutes; periods are shown separately. All scored penalties are included.")
        with right:
            st.subheader("Both teams, same measurements")
            comparison = report.comparisons.set_index("team").T
            comparison.index = comparison.index.str.replace("_", " ").str.capitalize()
            comparison = comparison.apply(lambda row: row.map(lambda v: display_value(v, rate=row.name.endswith("rate"))), axis=1)
            st.dataframe(comparison, width="stretch")
    with risk:
        st.subheader("Where possession became vulnerable")
        p = report.passes[report.passes.team == team]
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(pass_map(p[p.buildup_loss], team, "Buildup turnover locations", losses=True), width="stretch")
            st.caption("Location where the failed pass ended; marker size reflects modeled turnover cost. This is not an intended-target map.")
        with c2:
            st.plotly_chart(pass_map(p[p.entry], team, "Completed passes entering the final third"), width="stretch")
            st.caption("Each line is a completed pass crossing into the final third. Carries and dribbles are not included.")
        st.subheader("Channel comparisons")
        st.caption("Each team's channels use its own attacking direction: left is positive y after rotation. Turnover channels use pass origins.")
        fig = px.bar(report.channels, x="channel", y="entries", color="team", barmode="group", color_discrete_map={"home": "#51dfad", "away": "#ffad70"})
        st.plotly_chart(fig, width="stretch")
        channels = report.channels.copy()
        for name in ("entry_rate", "buildup_loss_rate"):
            channels[name] = channels[name].map(lambda v: display_value(v, rate=True))
        st.dataframe(channels, hide_index=True, width="stretch")
        st.caption("Unavailable rates have no relevant passes in their denominator; they do not mean 0%.")
    with players_tab:
        st.subheader("Who contributed through passing?")
        st.warning("Single-match rankings are noisy. These metrics credit passes only; goalkeeper roles are inferred from position.")
        metric = st.selectbox("Rank by", ["impact_p90", "buildup_turnovers_p90", "detriment_p90"], key="player_metric")
        columns = ["player_id", "minutes", "passes", "impact_p90", "buildup_turnovers_p90", "detriment_p90", "avg_pressure_m"]
        table = report.players.sort_values(metric, ascending=False)
        st.caption("Outfield players · at least 15 tracked minutes. Nearest-opponent distance is context, not a pressure-adjusted score.")
        config = {name: st.column_config.NumberColumn(name.replace("_", " ").capitalize(), format="%.3f") for name in columns if name not in {"player_id", "passes"}}
        outfield = table.loc[~table.is_gk, columns]
        if outfield.empty:
            st.info("No eligible outfield players have at least 15 tracked minutes for this team.")
        else:
            st.dataframe(outfield, hide_index=True, width="stretch", column_config=config)
        with st.expander("Goalkeepers · shown separately"):
            goalkeepers = table.loc[table.is_gk, columns]
            if goalkeepers.empty:
                st.info("No eligible goalkeeper has at least 15 tracked minutes for this team.")
            else:
                st.dataframe(goalkeepers, hide_index=True, width="stretch", column_config=config)
    with evidence_tab:
        st.subheader("Inspect the supporting moments")
        if not report.findings:
            st.info("No qualifying evidence to show.")
        else:
            context = f"{game}_{team}"
            index = st.selectbox("Finding", range(len(report.findings)), format_func=lambda i: report.findings[i].title, key=f"finding_{context}")
            finding = report.findings[index]
            moment_index = st.selectbox("Supporting moment", range(len(finding.evidence)), format_func=lambda i: finding.evidence[i].label, key=f"moment_{context}_{index}")
            moment = finding.evidence[moment_index]
            st.write(finding.observation)
            st.caption(f"Evidence ID: {moment.event_id} · {moment.team} · {moment.event}")
            offset = st.slider("Seconds relative to the event", -3.0, 3.0, 0.0, .2, key=f"offset_{context}_{index}_{moment_index}")
            control = st.checkbox("Show pitch control at this moment", key="show_control")
            window = tracking_window(bundle, moment)
            frame = snapshot(window, moment.time_s + offset, bundle.fps)
            if frame.empty:
                st.warning("Tracking is unavailable at this time. The source event arrow remains visible.")
            elif not {"home", "away"} <= set(frame.team):
                st.warning("Tracking for one team is missing; pitch control is unavailable.")
            st.plotly_chart(evidence_pitch(frame, moment, control=control, trail=window), width="stretch")
            st.caption("Tracking reconstruction, not match video. The gold arrow stays fixed to the source event; player positions follow the slider. Trails show the preceding two seconds. Home always attacks right.")
            if control:
                st.caption("Green indicates home control; orange indicates away control. Simplified time-to-intercept omits ball flight and acceleration; it is not pass-success probability.")
    chat_context = build_metrica_context(report, bundle.team_names)
    with v4_tab:
        sequence = call_tool(chat_context, "sequence_summary", {})
        st.subheader("Possessions and transitions")
        st.caption(f"{sequence['sample_counts']['possessions']} observed possessions · {sequence['sample_counts']['transitions']} qualifying transitions")
        st.caption(sequence["limitations"][0])
        for key, empty_message in (("possessions", "No observed possessions are available."),
                                   ("transitions", "No qualifying transitions are available.")):
            if sequence[key]:
                st.dataframe(pd.DataFrame(sequence[key]), hide_index=True, width="stretch")
            else:
                st.info(empty_message)
        defense = call_tool(chat_context, "defensive_summary", {})
        st.subheader("Defensive actions")
        st.caption(f"{defense['sample_counts']['defensive_actions']} recognized defensive actions")
        st.caption(defense["limitations"][0])
        if not defense["sample_counts"]["defensive_actions"]:
            st.info("No recognized defensive actions are recorded; this does not mean no defending occurred.")
        if defense["defensive_actions"]:
            table = pd.DataFrame(defense["defensive_actions"])
            table["actions_per_90"] = table.actions_per_90.map(display_value)
            st.dataframe(table, hide_index=True, width="stretch")
        shots = call_tool(chat_context, "shot_summary", {})
        st.subheader("Shot locations and outcomes")
        st.caption(f"{shots['sample_counts']['shot_summaries']} shots · raw outcomes by attacking channel")
        st.caption(shots["limitations"][0])
        if not shots["sample_counts"]["shot_summaries"]:
            st.info("No shots are recorded for this team.")
        else:
            st.plotly_chart(shot_map(report.event_evidence, team), width="stretch")
        if shots["shot_summaries"]:
            st.dataframe(pd.DataFrame(shots["shot_summaries"]), hide_index=True, width="stretch")
    render_chat_panel(
        st, chat_context, (selected.provider, selected.match_id, team),
        answerer=answer_question,
    )
    with st.expander("Methods, provenance & data quality"):
        for limitation in report.limitations:
            st.write(f"• {limitation}")
        st.write("xT training matches:", report.provenance["training_match_ids"])
        st.json(bundle.quality)
        st.markdown("Source: [Metrica Sports sample data](https://github.com/metrica-sports/sample-data). Raw data is downloaded separately and is not distributed with this repository.")


if __name__ == "__main__":
    main()

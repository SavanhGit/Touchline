"""Launch with: python -m streamlit run app.py."""
from pathlib import Path
from math import isfinite
from numbers import Real

import pandas as pd
import plotly.express as px
import streamlit as st

from soccer_tactics.analytics.report import analyze_match
from soccer_tactics.dashboard.charts import evidence_pitch, pass_map, snapshot, tracking_window
from soccer_tactics.loaders import load_metrica_bundle

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw" / "metrica"

st.set_page_config(page_title="Touchline | Soccer Match Analysis", page_icon="⚽", layout="wide")


@st.cache_resource(max_entries=2)
def load_bundle(game, revision):
    # Shared source bundles are read-only. Analytics build their own derived tables.
    return load_metrica_bundle(DATA, game)


@st.cache_data(max_entries=4, show_spinner="Building the evidence report…")
def load_report(game, team, revision):
    selected = load_bundle(game, revision)
    training = load_bundle(3 - game, revision)
    return analyze_match(selected, team, [training])


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
        missing = [f for f in source_files if not f.is_file() or f.stat().st_size == 0]
        if missing:
            st.warning("Source data is missing or empty. Both matches need events, home tracking and away tracking before analysis can start. Download or restore the Metrica sample data at the path below.")
            st.code(str(DATA))
            st.write("Missing or empty files:", ", ".join(f.name for f in missing))
            st.code("git clone --depth 1 https://github.com/metrica-sports/sample-data data/raw/metrica")
            st.stop()
        game = st.selectbox("Match", [1, 2], format_func=lambda g: f"Metrica · Sample game {g}", key="selected_match")
        team = st.selectbox("Analyze team", ["away", "home"], format_func=lambda t: f"{t.capitalize()} (anonymous)", key="selected_team")
        st.divider()
        st.caption("POST-MATCH ANALYSIS")
        st.write("Explore the evidence. Question the explanation.")
        st.caption("Data: Metrica Sports. Teams and players are anonymized. This is a portfolio research tool.")
    engine_files = [Path(__file__)] + sorted((ROOT / "src" / "soccer_tactics").rglob("*.py"))
    revision = tuple((str(f.resolve()), f.stat().st_mtime_ns, f.stat().st_size) for f in source_files + engine_files)
    try:
        bundle = load_bundle(game, revision)
        report = load_report(game, team, revision)
    except (ValueError, FileNotFoundError) as exc:
        st.error(f"Match analysis unavailable: {exc}")
        st.stop()
    st.markdown('<div class="eyebrow">THE MATCH, EXPLAINED THROUGH EVIDENCE</div>', unsafe_allow_html=True)
    st.title("What shaped the result?")
    score_text = f"Home {report.score['home']} – {report.score['away']} Away" if report.score is not None else "Score unknown"
    st.markdown(f"**Sample game {game} · {score_text}**  /  {team.capitalize()} perspective · {report.outcome.upper()}")
    st.caption("Patterns that may have contributed, with adjustments to investigate. These findings do not establish causation.")
    overview, risk, players_tab, evidence_tab = st.tabs(["Overview", "Risk & progression", "Players", "Evidence"])
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
    with st.expander("Methods, provenance & data quality"):
        for limitation in report.limitations:
            st.write(f"• {limitation}")
        st.write("xT training matches:", report.provenance["training_match_ids"])
        st.json(bundle.quality)
        st.markdown("Source: [Metrica Sports sample data](https://github.com/metrica-sports/sample-data). Raw data is downloaded separately and is not distributed with this repository.")


if __name__ == "__main__":
    main()

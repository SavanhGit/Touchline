"""Pitch figures and period-aware tracking windows."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from soccer_tactics.analytics.features import add_kinematics
from soccer_tactics.analytics.pitch_control import pitch_control
from soccer_tactics.analytics.xt import centers

COLORS = {"home": "#51dfad", "away": "#ffad70", "ball": "#ffffff"}


def pitch(title: str = "", attacking_team: str | None = None) -> go.Figure:
    fig = go.Figure()
    line = dict(color="#506e68", width=1)
    fig.add_shape(type="rect", x0=-52.5, x1=52.5, y0=-34, y1=34, line=line)
    fig.add_shape(type="line", x0=0, x1=0, y0=-34, y1=34, line=line)
    fig.add_shape(type="circle", x0=-9.15, x1=9.15, y0=-9.15, y1=9.15, line=line)
    for sign in (-1, 1):
        for depth, width in ((16.5, 40.32), (5.5, 18.32)):
            ends = sorted([sign * 52.5, sign * (52.5 - depth)])
            fig.add_shape(type="rect", x0=ends[0], x1=ends[1], y0=-width / 2, y1=width / 2, line=line)
        fig.add_shape(type="rect", x0=sign * 52.5, x1=sign * 54.5, y0=-3.66, y1=3.66, line=line)
    fig.update_layout(
        title=title, paper_bgcolor="#10231f", plot_bgcolor="#10231f", font=dict(color="#deebe6"),
        height=470, margin=dict(l=15, r=15, t=45, b=20), showlegend=True,
        xaxis=dict(range=[-57, 57], visible=False, constrain="domain"),
        yaxis=dict(range=[-38, 38], visible=False, scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", y=-.05),
    )
    if attacking_team:
        fig.add_annotation(x=0, y=36, text=f"{attacking_team.capitalize()} attacking frame →", showarrow=False)
    return fig


def pass_map(rows: pd.DataFrame, team: str, title: str, losses: bool = False) -> go.Figure:
    fig = pitch(title, team)
    sign = 1 if team == "home" else -1
    if losses:
        fig.add_trace(go.Scatter(
            x=rows.x1 * sign, y=rows.y1 * sign, mode="markers",
            marker=dict(size=8 + np.sqrt(rows.turnover_cost.to_numpy()) * 12, color="#ffad70", opacity=.8),
            text=[f"{r.player_id} · P{r.period} · {r.time_s / 60:.2f} min<br>Cost {r.turnover_cost:.3f} xT" for r in rows.itertuples()],
            hovertemplate="%{text}<extra></extra>", name="Buildup loss location",
        ))
    else:
        xs, ys = [], []
        for r in rows.itertuples():
            xs.extend([r.x0 * sign, r.x1 * sign, None])
            ys.extend([r.y0 * sign, r.y1 * sign, None])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#51dfad", width=1.2), opacity=.45, name="Completed entry routes", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=rows.x1 * sign, y=rows.y1 * sign, mode="markers", marker=dict(size=5, color="#51dfad"), name="Entry endpoints"))
    return fig


def tracking_window(bundle, evidence, radius_s: float = 3.4) -> pd.DataFrame:
    t = bundle.tracking
    mask = (t.period == evidence.period) & ((t.time_s - evidence.time_s).abs() <= radius_s)
    return add_kinematics(t.loc[mask], fps=bundle.fps) if mask.any() else t.iloc[:0].copy()


def snapshot(window: pd.DataFrame, target_s: float, fps: float) -> pd.DataFrame:
    if window.empty:
        return window
    clocks = window[["frame", "time_s"]].drop_duplicates()
    nearest = clocks.loc[(clocks.time_s - target_s).abs().idxmin()]
    if abs(nearest.time_s - target_s) > max(1.5 / fps, .06):
        return window.iloc[:0]
    return window[window.frame == nearest.frame]


def evidence_pitch(frame: pd.DataFrame, evidence, *, control: bool = False, trail: pd.DataFrame | None = None) -> go.Figure:
    fig = pitch("Tracking reconstruction · home → / away ←")
    if control and not frame.empty and {"home", "away"} <= set(frame.team):
        grid = pitch_control(frame)
        X, Y = centers(*grid.shape)
        fig.add_trace(go.Heatmap(x=X[:, 0], y=Y[0], z=grid.T, zmin=0, zmax=1,
                                colorscale=[[0, "#b36e41"], [.5, "#16382f"], [1, "#349a79"]],
                                opacity=.55, showscale=False, hoverinfo="skip", name="Home pitch control"))
    if trail is not None and not frame.empty:
        trail = trail[(trail.time_s <= frame.time_s.iloc[0]) & (trail.time_s >= frame.time_s.iloc[0] - 2)]
        for (side, _), group in trail.groupby(["team", "player_id"]):
            group = group.sort_values("frame").iloc[::5]
            fig.add_trace(go.Scatter(x=group.x, y=group.y, mode="lines", line=dict(color=COLORS[side], width=1), opacity=.3, showlegend=False, hoverinfo="skip"))
    for side in ("home", "away", "ball"):
        players = frame[frame.team == side]
        fig.add_trace(go.Scatter(
            x=players.x, y=players.y, mode="markers+text" if side != "ball" else "markers",
            marker=dict(size=12 if side != "ball" else 7, color=COLORS[side], line=dict(color="#10231f", width=1)),
            text=players.player_id.str.replace("Player", "", regex=False), textposition="top center", textfont=dict(size=10),
            customdata=players.player_id, hovertemplate="%{customdata}<br>%{x:.1f}, %{y:.1f} m<extra></extra>", name=side.capitalize(),
        ))
    if all(np.isfinite(v) for v in (evidence.x0, evidence.y0, evidence.x1, evidence.y1)):
        fig.add_annotation(x=evidence.x1, y=evidence.y1, ax=evidence.x0, ay=evidence.y0,
                           xref="x", yref="y", axref="x", ayref="y", arrowhead=3, arrowwidth=3, arrowcolor="#f3dc87")
    return fig

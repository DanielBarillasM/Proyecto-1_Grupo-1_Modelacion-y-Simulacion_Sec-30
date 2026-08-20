from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from simulation import SimulationResult


BG = "#070a08"
PANEL = "#0d1410"
GRID = "#26342b"
TEXT = "#dbe7de"
MUTED = "#8fa79a"
GREEN = "#8cff8c"
LIME = "#b7ff5a"
RED = "#ff4d5f"
ORANGE = "#ff9f43"
STEEL = "#70867a"


def _box_mesh(cx: float, cy: float, w: float, d: float, h: float, color: str, name: str) -> go.Mesh3d:
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - d / 2, cy + d / 2
    z0, z1 = 0.0, h
    x = [x0, x1, x1, x0, x0, x1, x1, x0]
    y = [y0, y0, y1, y1, y0, y0, y1, y1]
    z = [z0, z0, z0, z0, z1, z1, z1, z1]
    i = [0, 0, 0, 1, 2, 4, 4, 5, 6, 3, 3, 7]
    j = [1, 2, 4, 2, 3, 5, 6, 6, 7, 0, 4, 4]
    k = [2, 3, 5, 5, 7, 6, 7, 2, 3, 4, 7, 0]
    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=color, opacity=0.72, flatshading=True,
        name=name, hoverinfo="skip", showscale=False,
    )


def arena_3d(result: SimulationResult, t_view: float) -> go.Figure:
    cfg = result.config
    fig = go.Figure()

    # Ground plane.
    grid = np.linspace(-cfg.arena_radius * 1.1, cfg.arena_radius * 1.1, 22)
    xx, yy = np.meshgrid(grid, grid)
    zz = np.zeros_like(xx)
    fig.add_trace(
        go.Surface(
            x=xx, y=yy, z=zz,
            colorscale=[[0, "#0a0f0c"], [1, "#111a14"]],
            showscale=False, opacity=1.0, hoverinfo="skip", name="Terreno"
        )
    )

    # Perimeter and danger ring.
    theta = np.linspace(0, 2 * np.pi, 240)
    for radius, color, width in [
        (cfg.arena_radius, STEEL, 3),
        (cfg.weapon_range, GREEN, 2),
        (cfg.contact_radius, RED, 3),
    ]:
        fig.add_trace(
            go.Scatter3d(
                x=radius * np.cos(theta), y=radius * np.sin(theta), z=np.full_like(theta, 0.08),
                mode="lines", line=dict(color=color, width=width), hoverinfo="skip",
                showlegend=False,
            )
        )

    # Ruins / cover blocks to make the scene feel more like an apocalypse arena.
    ruins = [
        (-12.0, 10.0, 5.0, 4.0, 3.2),
        (11.5, 10.5, 4.5, 5.2, 4.6),
        (-12.8, -10.5, 4.0, 5.5, 5.4),
        (12.3, -9.8, 5.5, 3.5, 2.8),
    ]
    for idx, (x, y, w, d, h) in enumerate(ruins, start=1):
        fig.add_trace(_box_mesh(x, y, w, d, h, "#26312b", f"Ruina {idx}"))

    # Survivor.
    fig.add_trace(
        go.Scatter3d(
            x=[0], y=[0], z=[1.1], mode="markers+text",
            marker=dict(size=11, color=LIME, symbol="diamond", line=dict(width=2, color="#ecffd5")),
            text=["SURVIVOR"], textposition="top center", textfont=dict(color=LIME, size=11),
            name="Jugador", hovertemplate="Jugador<extra></extra>",
        )
    )

    alive = result.enemies[
        (result.enemies["spawn_time"] <= t_view)
        & (
            result.enemies["death_time"].isna()
            | (result.enemies["death_time"] > t_view)
        )
    ].copy()

    if not alive.empty:
        xs, ys, zs, dists, hover = [], [], [], [], []
        for row in alive.itertuples(index=False):
            elapsed = max(0.0, t_view - float(row.spawn_time))
            radius = max(cfg.contact_radius, float(row.spawn_radius) - float(row.speed) * elapsed)
            x = radius * math.cos(float(row.angle))
            y = radius * math.sin(float(row.angle))
            xs.append(x); ys.append(y); zs.append(0.55); dists.append(radius)
            hover.append(
                f"Enemigo #{int(row.enemy_id)}<br>Distancia: {radius:.1f} m"
                f"<br>Velocidad: {float(row.speed):.2f} m/s"
            )

        fig.add_trace(
            go.Scatter3d(
                x=xs, y=ys, z=zs, mode="markers",
                marker=dict(
                    size=np.clip(10 - np.array(dists) * 0.18, 5, 10),
                    color=dists, colorscale=[[0, RED], [0.45, ORANGE], [1, "#8a9890"]],
                    cmin=0, cmax=cfg.arena_radius, showscale=False,
                    line=dict(color="#29090d", width=1), opacity=0.95,
                ),
                text=hover, hovertemplate="%{text}<extra></extra>", name="Infectados",
            )
        )

        # Threat vectors for the closest enemies.
        closest = np.argsort(np.array(dists))[: min(8, len(dists))]
        for idx in closest:
            fig.add_trace(
                go.Scatter3d(
                    x=[xs[idx], 0], y=[ys[idx], 0], z=[0.2, 0.2], mode="lines",
                    line=dict(color="#5a2027", width=2, dash="dot"),
                    hoverinfo="skip", showlegend=False,
                )
            )

    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter, system-ui, sans-serif"),
        title=dict(text=f"Arena táctica 3D · t = {t_view:.1f} s", x=0.02, font=dict(size=18, color=TEXT)),
        legend=dict(bgcolor="rgba(8,12,9,.7)", bordercolor=GRID, borderwidth=1),
        scene=dict(
            bgcolor=BG,
            aspectmode="cube",
            xaxis=dict(range=[-cfg.arena_radius * 1.12, cfg.arena_radius * 1.12], visible=False),
            yaxis=dict(range=[-cfg.arena_radius * 1.12, cfg.arena_radius * 1.12], visible=False),
            zaxis=dict(range=[0, 11], visible=False),
            camera=dict(eye=dict(x=1.45, y=1.55, z=1.15)),
        ),
    )
    return fig


def timeline_figure(result: SimulationResult) -> go.Figure:
    df = result.timeline
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["player_hp"], mode="lines",
            name="Vida", line=dict(color=GREEN, width=3),
            hovertemplate="t=%{x:.1f}s<br>HP=%{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["active_enemies"], mode="lines",
            name="Enemigos activos", yaxis="y2", line=dict(color=RED, width=2),
            hovertemplate="t=%{x:.1f}s<br>Activos=%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TEXT),
        title="Estado del sistema durante la partida",
        xaxis=dict(title="Tiempo (s)", gridcolor=GRID, zeroline=False),
        yaxis=dict(title="HP del jugador", gridcolor=GRID, range=[0, max(result.config.player_hp * 1.05, 105)]),
        yaxis2=dict(title="Enemigos activos", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return fig


def arrival_process_figure(result: SimulationResult) -> go.Figure:
    if result.arrivals.empty:
        x = [0.0, result.config.duration]
        y = [0, 0]
    else:
        times = result.arrivals["arrival_time"].to_numpy()
        x = np.r_[0.0, np.repeat(times, 2), result.config.duration]
        counts = np.arange(1, len(times) + 1)
        y = np.r_[0, np.column_stack((counts - 1, counts)).ravel(), len(times)]

    fig = go.Figure(
        go.Scatter(
            x=x, y=y, mode="lines", line=dict(color=LIME, width=2),
            fill="tozeroy", fillcolor="rgba(183,255,90,0.07)",
            hovertemplate="t=%{x:.2f}s<br>N(t)=%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TEXT),
        title="Proceso de conteo N(t)",
        xaxis=dict(title="Tiempo (s)", gridcolor=GRID),
        yaxis=dict(title="Enemigos acumulados", gridcolor=GRID, rangemode="tozero"),
    )
    return fig


def interarrival_figure(result: SimulationResult) -> go.Figure:
    deltas = result.arrivals["delta"].to_numpy() if not result.arrivals.empty else np.array([])
    fig = go.Figure()
    if len(deltas):
        fig.add_trace(
            go.Histogram(
                x=deltas, histnorm="probability density", nbinsx=min(30, max(8, int(np.sqrt(len(deltas)) * 2))),
                name="Simulado", opacity=0.72, marker=dict(color="#4f725d"),
            )
        )
        xmax = max(np.quantile(deltas, 0.98), 3 / result.config.lambda_rate)
        xs = np.linspace(0, xmax, 250)
        pdf = result.config.lambda_rate * np.exp(-result.config.lambda_rate * xs)
        fig.add_trace(
            go.Scatter(x=xs, y=pdf, mode="lines", name="PDF exponencial teórica", line=dict(color=ORANGE, width=3))
        )
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TEXT),
        title="Tiempos entre llegadas: simulación vs teoría",
        xaxis=dict(title="Δᵢ (s)", gridcolor=GRID),
        yaxis=dict(title="Densidad", gridcolor=GRID),
        barmode="overlay",
    )
    return fig


def monte_carlo_figure(batch: pd.DataFrame) -> go.Figure:
    survived = int(batch["survived"].sum())
    failed = int(len(batch) - survived)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Sobrevive", "Cae"], values=[survived, failed], hole=0.67,
                marker=dict(colors=[GREEN, RED]), textinfo="label+percent",
                hovertemplate="%{label}: %{value}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=330, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=PANEL, font=dict(color=TEXT),
        title="Resultado Monte Carlo",
        showlegend=False,
        annotations=[dict(text=f"{len(batch)}<br>partidas", x=0.5, y=0.5, showarrow=False, font=dict(size=20, color=TEXT))],
    )
    return fig


def survival_curve_figure(curve: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=curve["lambda"], y=curve["survival_probability"] * 100,
            mode="lines+markers", line=dict(color=LIME, width=3), marker=dict(size=9),
            hovertemplate="λ=%{x:.2f}<br>Supervivencia=%{y:.1f}%<extra></extra>",
        )
    )
    fig.add_hline(y=50, line_dash="dash", line_color="#8a9890", annotation_text="50%")
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TEXT),
        title="Curva de dificultad: λ vs probabilidad de sobrevivir",
        xaxis=dict(title="λ (enemigos/s)", gridcolor=GRID),
        yaxis=dict(title="Probabilidad de supervivencia (%)", gridcolor=GRID, range=[0, 100]),
    )
    return fig

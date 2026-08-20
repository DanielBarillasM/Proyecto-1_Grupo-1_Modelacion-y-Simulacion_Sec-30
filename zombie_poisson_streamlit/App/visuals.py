"""Visualizaciones estadísticas y escena táctica tridimensional.

Las funciones de este módulo son puras respecto al estado de Streamlit: reciben
DataFrames o un ``SimulationResult`` y devuelven figuras de Plotly. Esto permite
probar el motor por separado y reutilizar las figuras en otra interfaz.

La escena 3D no carga modelos externos. Construye cajas y octaedros, los combina
en pocas trazas ``Mesh3d`` y reduce así el costo de renderizar muchos infectados.
Los edificios y vehículos son contexto visual; no forman parte de la física del
modelo, que conserva movimiento radial sin colisiones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from simulation import SimulationResult


# Paleta compartida. Centralizar los colores garantiza correspondencia entre
# tarjetas de Streamlit, modelos estadísticos y elementos de la arena.
BG = "#070908"
PANEL = "#0d1210"
GRID = "#28342d"
TEXT = "#e8eee9"
MUTED = "#91a098"
GREEN = "#9fe870"
LIME = "#c7ff73"
RED = "#ff5964"
ORANGE = "#f7a64a"
BLUE = "#66b6d6"
POISSON_COLOR = "#b8f36b"
POLAR_COLOR = "#68b8d8"


@dataclass
class MeshBuilder:
    """Acumula vértices y triángulos antes de crear una traza Plotly.

    Plotly representa una malla con tres arreglos de coordenadas y tres arreglos
    de índices ``i``, ``j`` y ``k``. Cada llamada a ``box`` u ``octahedron``
    agrega geometría con un desplazamiento de índices correcto. Consolidar muchas
    primitivas en una sola traza es considerablemente más eficiente que crear una
    traza por extremidad o edificio.
    """

    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    z: list[float] = field(default_factory=list)
    i: list[int] = field(default_factory=list)
    j: list[int] = field(default_factory=list)
    k: list[int] = field(default_factory=list)
    facecolor: list[str] = field(default_factory=list)

    def box(
        self,
        cx: float,
        cy: float,
        cz: float,
        width: float,
        depth: float,
        height: float,
        color: str,
        angle: float = 0.0,
    ) -> None:
        """Agrega un prisma rectangular rotado alrededor del eje vertical.

        Args:
            cx, cy, cz: centro de la caja en coordenadas de mundo.
            width, depth, height: dimensiones antes de rotar.
            color: color CSS aplicado a sus doce caras triangulares.
            angle: rotación en radianes alrededor del eje Z.
        """

        local = np.array([
            [-width / 2, -depth / 2, -height / 2],
            [width / 2, -depth / 2, -height / 2],
            [width / 2, depth / 2, -height / 2],
            [-width / 2, depth / 2, -height / 2],
            [-width / 2, -depth / 2, height / 2],
            [width / 2, -depth / 2, height / 2],
            [width / 2, depth / 2, height / 2],
            [-width / 2, depth / 2, height / 2],
        ])
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        local[:, :2] = local[:, :2] @ rotation.T
        local += np.array([cx, cy, cz])
        offset = len(self.x)
        self.x.extend(local[:, 0].tolist())
        self.y.extend(local[:, 1].tolist())
        self.z.extend(local[:, 2].tolist())
        # Cada cara rectangular se divide en dos triángulos. El orden conserva
        # una orientación consistente para que la iluminación no se invierta.
        faces = [
            (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
            (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
            (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0),
        ]
        for a, b, c in faces:
            self.i.append(offset + a)
            self.j.append(offset + b)
            self.k.append(offset + c)
            self.facecolor.append(color)

    def octahedron(self, cx: float, cy: float, cz: float, radius: float, color: str) -> None:
        """Agrega un octaedro usado como cabeza de bajo poligonaje.

        Se elige un octaedro porque necesita solo seis vértices y ocho caras,
        pero sigue siendo legible desde cualquier ángulo de cámara.
        """

        vertices = [
            (cx + radius, cy, cz), (cx - radius, cy, cz),
            (cx, cy + radius, cz), (cx, cy - radius, cz),
            (cx, cy, cz + radius * 1.25), (cx, cy, cz - radius * 1.25),
        ]
        offset = len(self.x)
        self.x.extend(v[0] for v in vertices)
        self.y.extend(v[1] for v in vertices)
        self.z.extend(v[2] for v in vertices)
        faces = [
            (4, 0, 2), (4, 2, 1), (4, 1, 3), (4, 3, 0),
            (5, 2, 0), (5, 1, 2), (5, 3, 1), (5, 0, 3),
        ]
        for a, b, c in faces:
            self.i.append(offset + a)
            self.j.append(offset + b)
            self.k.append(offset + c)
            self.facecolor.append(color)

    def trace(self, name: str, opacity: float = 1.0) -> go.Mesh3d:
        """Materializa todo lo acumulado como una única traza ``Mesh3d``.

        Args:
            name: etiqueta mostrada en la leyenda de Plotly.
            opacity: transparencia global de la malla.
        """

        return go.Mesh3d(
            x=self.x, y=self.y, z=self.z,
            i=self.i, j=self.j, k=self.k,
            facecolor=self.facecolor,
            flatshading=True,
            lighting=dict(ambient=0.45, diffuse=0.72, roughness=0.9, specular=0.15),
            lightposition=dict(x=30, y=-20, z=45),
            opacity=opacity,
            name=name,
            hoverinfo="skip",
            showscale=False,
        )


def _offset(cx: float, cy: float, angle: float, forward: float, lateral: float) -> tuple[float, float]:
    """Convierte un desplazamiento local del personaje a coordenadas de mundo."""

    return (
        cx + forward * math.cos(angle) - lateral * math.sin(angle),
        cy + forward * math.sin(angle) + lateral * math.cos(angle),
    )


def _add_humanoid(
    mesh: MeshBuilder,
    x: float,
    y: float,
    angle: float,
    scale: float,
    torso_color: str,
    skin_color: str,
    pose: float = 0.0,
) -> None:
    """Construye un humanoide estilizado con primitivas de bajo costo.

    ``angle`` orienta el torso, ``scale`` controla el tamaño completo y ``pose``
    inclina los brazos en direcciones opuestas. El protagonista y los infectados
    comparten geometría; colores, escala y postura comunican su rol.
    """

    for lateral in (-0.18, 0.18):
        px, py = _offset(x, y, angle, 0.0, lateral * scale)
        mesh.box(px, py, 0.42 * scale, 0.24 * scale, 0.24 * scale, 0.84 * scale, "#252b28", angle)
    mesh.box(x, y, 1.23 * scale, 0.74 * scale, 0.38 * scale, 0.86 * scale, torso_color, angle)
    neck_x, neck_y = _offset(x, y, angle, 0.0, 0.0)
    mesh.octahedron(neck_x, neck_y, 1.93 * scale, 0.32 * scale, skin_color)
    for lateral, lean in ((-0.52, -pose), (0.52, pose)):
        arm_x, arm_y = _offset(x, y, angle, 0.03 * scale, lateral * scale)
        mesh.box(
            arm_x,
            arm_y,
            1.22 * scale,
            0.20 * scale,
            0.20 * scale,
            0.86 * scale,
            skin_color,
            angle + lean,
        )


def _scene_mesh(arena_radius: float) -> MeshBuilder:
    """Crea la geometría ambiental estática de la zona de contención.

    Args:
        arena_radius: escala de referencia para distribuir barricadas.

    Returns:
        Constructor con edificios, vehículos, barreras y plataforma central.

    Notes:
        Esta geometría no altera trayectorias; hace explícito el límite entre
        representación visual y reglas matemáticas del simulador.
    """

    mesh = MeshBuilder()
    buildings = [
        (-16.0, 13.5, 7.5, 5.0, 7.2),
        (-8.2, 16.5, 6.0, 4.2, 4.8),
        (15.5, 13.0, 6.5, 5.5, 8.6),
        (16.2, -12.5, 7.0, 5.0, 5.7),
        (-15.8, -13.2, 7.2, 5.2, 6.6),
    ]
    for index, (x, y, width, depth, height) in enumerate(buildings):
        mesh.box(x, y, height / 2, width, depth, height, "#242b27" if index % 2 else "#2c332e")
        mesh.box(x + width * 0.18, y - depth * 0.51, height * 0.62, width * 0.22, 0.08, height * 0.2, "#8f5f38")
    for x, y, angle in [(-10.5, 5.6, 0.3), (9.8, -6.8, -0.45), (5.0, 12.8, 1.1)]:
        mesh.box(x, y, 0.55, 3.4, 1.65, 0.85, "#4c514e", angle)
        mesh.box(x, y, 1.05, 1.75, 1.45, 0.48, "#313a37", angle)
    for angle in np.linspace(0, 2 * np.pi, 14, endpoint=False):
        radius = arena_radius * 0.82
        x, y = radius * math.cos(angle), radius * math.sin(angle)
        mesh.box(x, y, 0.48, 2.0, 0.42, 0.96, "#5a4d3e", angle + np.pi / 2)
    mesh.box(0, 0, 0.16, 3.2, 3.2, 0.32, "#26322a")
    return mesh


def arena_3d(result: SimulationResult, t_view: float) -> go.Figure:
    """Reconstruye la arena en un instante de la partida.

    Args:
        result: resultado completo que contiene configuración y entidades.
        t_view: segundo de la misión que se desea observar.

    Returns:
        Figura Plotly rotatoria con terreno, escenario, anillos y personajes.

    La posición se recalcula desde ``spawn_time``, velocidad y ángulo; no se
    almacenan coordenadas por frame. ``death_time`` impide mostrar una entidad
    después de su eliminación.
    """

    cfg = result.config
    fig = go.Figure()
    # Una superficie de 28 x 28 mantiene textura visible con pocos polígonos.
    grid = np.linspace(-cfg.arena_radius * 1.15, cfg.arena_radius * 1.15, 28)
    xx, yy = np.meshgrid(grid, grid)
    texture = 0.03 * np.sin(xx * 0.75) * np.cos(yy * 0.65)
    fig.add_trace(go.Surface(
        x=xx, y=yy, z=texture,
        surfacecolor=np.sqrt(xx**2 + yy**2),
        colorscale=[[0, "#151d18"], [0.5, "#0d1210"], [1, "#090c0a"]],
        showscale=False,
        hoverinfo="skip",
        name="Terreno",
    ))
    fig.add_trace(_scene_mesh(cfg.arena_radius).trace("Escenario"))

    # Los tres anillos codifican perímetro, alcance del arma y contacto.
    theta = np.linspace(0, 2 * np.pi, 260)
    for radius, color, width, dash in [
        (cfg.arena_radius, "#59665e", 4, "solid"),
        (cfg.weapon_range, GREEN, 4, "dash"),
        (cfg.contact_radius, RED, 5, "solid"),
    ]:
        fig.add_trace(go.Scatter3d(
            x=radius * np.cos(theta),
            y=radius * np.sin(theta),
            z=np.full_like(theta, 0.11),
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hoverinfo="skip",
            showlegend=False,
        ))

    player_mesh = MeshBuilder()
    _add_humanoid(player_mesh, 0, 0, math.pi / 2, 1.05, "#a6e36d", "#d1b38b")
    fig.add_trace(player_mesh.trace("Sobreviviente"))

    # Seleccionar entidades vivas en t_view permite explorar pasado y presente
    # con la misma tabla final de enemigos.
    alive = result.enemies[
        (result.enemies["spawn_time"] <= t_view)
        & (result.enemies["death_time"].isna() | (result.enemies["death_time"] > t_view))
    ].copy()
    zombie_mesh = MeshBuilder()
    xs: list[float] = []
    ys: list[float] = []
    hover: list[str] = []
    for position, row in enumerate(alive.itertuples(index=False)):
        elapsed = max(0.0, t_view - float(row.spawn_time))
        radius = max(cfg.contact_radius, float(row.spawn_radius) - float(row.speed) * elapsed)
        x = radius * math.cos(float(row.angle))
        y = radius * math.sin(float(row.angle))
        facing = math.atan2(-y, -x)
        danger = radius <= cfg.contact_radius + 0.05
        _add_humanoid(
            zombie_mesh,
            x,
            y,
            facing,
            0.78,
            "#6d3034" if danger else "#43553f",
            "#82936c",
            pose=0.16 if position % 2 else -0.16,
        )
        xs.append(x)
        ys.append(y)
        hover.append(
            f"Infectado {int(row.enemy_id)}<br>Distancia: {radius:.1f} m"
            f"<br>Velocidad: {float(row.speed):.2f} m/s"
        )
    if zombie_mesh.x:
        fig.add_trace(zombie_mesh.trace("Infectados"))
        # Los marcadores transparentes aportan un blanco de hover amplio sin
        # ocultar la geometría detallada de la malla.
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=np.full(len(xs), 1.45),
            mode="markers",
            marker=dict(size=8, color="rgba(0,0,0,0.01)"),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        height=690,
        margin=dict(l=0, r=0, t=52, b=0),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter, Segoe UI, sans-serif"),
        title=dict(text=f"Zona de contención | t = {t_view:.1f} s", x=0.02, font=dict(size=19)),
        legend=dict(bgcolor="rgba(7,9,8,.82)", bordercolor=GRID, borderwidth=1),
        scene=dict(
            bgcolor=BG,
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.48),
            xaxis=dict(range=[-cfg.arena_radius * 1.18, cfg.arena_radius * 1.18], visible=False),
            yaxis=dict(range=[-cfg.arena_radius * 1.18, cfg.arena_radius * 1.18], visible=False),
            zaxis=dict(range=[0, 11], visible=False),
            camera=dict(eye=dict(x=1.45, y=1.55, z=1.05), center=dict(x=0, y=0, z=-0.12)),
        ),
    )
    return fig


def timeline_figure(result: SimulationResult) -> go.Figure:
    """Grafica HP y concurrencia con dos escalas verticales sincronizadas."""

    df = result.timeline
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["player_hp"], mode="lines", name="Vida",
        line=dict(color=GREEN, width=3), fill="tozeroy", fillcolor="rgba(159,232,112,.08)",
        hovertemplate="t=%{x:.1f} s<br>HP=%{y:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["active_enemies"], mode="lines", name="Infectados activos",
        yaxis="y2", line=dict(color=RED, width=2),
        hovertemplate="t=%{x:.1f} s<br>Activos=%{y}<extra></extra>",
    ))
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TEXT),
        title="Evolución del sistema",
        xaxis=dict(title="Tiempo (s)", gridcolor=GRID, zeroline=False),
        yaxis=dict(title="HP", gridcolor=GRID, range=[0, result.config.player_hp * 1.05]),
        yaxis2=dict(title="Activos", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    return fig


def arrival_process_figure(result: SimulationResult) -> go.Figure:
    """Construye la trayectoria escalonada del proceso de conteo observado.

    Cada instante se duplica en X para dibujar un salto vertical exacto de
    ``N(t)-1`` a ``N(t)``. La curva termina en el cierre real de la partida.
    """

    arrivals = result.arrivals
    end_time = result.survival_time
    if arrivals.empty:
        x, y = [0.0, end_time], [0, 0]
    else:
        times = arrivals["arrival_time"].to_numpy()
        x = np.r_[0.0, np.repeat(times, 2), end_time]
        counts = np.arange(1, len(times) + 1)
        y = np.r_[0, np.column_stack((counts - 1, counts)).ravel(), len(times)]
    color = POISSON_COLOR if result.config.arrival_model == "poisson" else POLAR_COLOR
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="lines", line=dict(color=color, width=2.5),
        fill="tozeroy", fillcolor="rgba(184,243,107,.07)",
        hovertemplate="t=%{x:.2f} s<br>N(t)=%{y}<extra></extra>",
    ))
    fig.update_layout(
        height=350, margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TEXT),
        title="Proceso de conteo observado N(t)",
        xaxis=dict(title="Tiempo (s)", gridcolor=GRID),
        yaxis=dict(title="Llegadas acumuladas", gridcolor=GRID, rangemode="tozero"),
    )
    return fig


def interarrival_figure(result: SimulationResult) -> go.Figure:
    """Superpone interarribos observados y densidad teórica del modelo.

    La figura es diagnóstica: una sola partida produce una muestra censurada por
    el horizonte y no constituye por sí misma una prueba de bondad de ajuste.
    """

    deltas = result.arrivals["delta"].to_numpy() if not result.arrivals.empty else np.array([])
    fig = go.Figure()
    if len(deltas):
        color = POISSON_COLOR if result.config.arrival_model == "poisson" else POLAR_COLOR
        fig.add_trace(go.Histogram(
            x=deltas, histnorm="probability density",
            nbinsx=min(30, max(8, int(np.sqrt(len(deltas)) * 2))),
            name="Observado", opacity=0.68, marker=dict(color=color),
        ))
        xmax = max(float(np.quantile(deltas, 0.98)), 3.0 / result.config.lambda_rate)
        xs = np.linspace(0.001, xmax, 300)
        if result.config.arrival_model == "poisson":
            pdf = result.config.lambda_rate * np.exp(-result.config.lambda_rate * xs)
            label = "Exponencial teórica"
        else:
            cv = result.config.polar_cv
            sigma2 = math.log(1 + cv**2)
            sigma = math.sqrt(sigma2)
            mu = math.log(1 / result.config.lambda_rate) - sigma2 / 2
            pdf = np.exp(-((np.log(xs) - mu) ** 2) / (2 * sigma2)) / (xs * sigma * np.sqrt(2 * np.pi))
            label = "Lognormal teórica"
        fig.add_trace(go.Scatter(
            x=xs, y=pdf, mode="lines", name=label, line=dict(color=ORANGE, width=3)
        ))
    fig.update_layout(
        height=350, margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TEXT),
        title="Interarribos observados y densidad de referencia",
        xaxis=dict(title="Delta (s)", gridcolor=GRID),
        yaxis=dict(title="Densidad", gridcolor=GRID),
        barmode="overlay",
        legend=dict(orientation="h", y=1.12, x=0),
    )
    return fig


def monte_carlo_figure(batch: pd.DataFrame) -> go.Figure:
    """Resume un lote en un gráfico circular de supervivencia y fracaso."""

    survived = int(batch["survived"].sum())
    failed = int(len(batch) - survived)
    fig = go.Figure(go.Pie(
        labels=["Sobrevive", "No sobrevive"], values=[survived, failed], hole=0.68,
        marker=dict(colors=[GREEN, RED]), textinfo="label+percent",
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    fig.update_layout(
        height=330, margin=dict(l=10, r=10, t=42, b=10),
        paper_bgcolor=PANEL, font=dict(color=TEXT), title="Resultados Monte Carlo",
        showlegend=False,
        annotations=[dict(
            text=f"{len(batch)}<br>partidas", x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color=TEXT),
        )],
    )
    return fig


def survival_curve_figure(curve: pd.DataFrame) -> go.Figure:
    """Grafica la curva de dificultad con errores asimétricos de Wilson."""

    color = POISSON_COLOR if curve.iloc[0]["model"] == "poisson" else POLAR_COLOR
    probabilities = curve["survival_probability"] * 100
    fig = go.Figure(go.Scatter(
        x=curve["lambda"] * 60,
        y=probabilities,
        error_y=dict(
            type="data",
            symmetric=False,
            array=(curve["ci_high"] * 100 - probabilities),
            arrayminus=(probabilities - curve["ci_low"] * 100),
            color=color,
        ),
        mode="lines+markers",
        line=dict(color=color, width=3), marker=dict(size=8),
        hovertemplate="Tasa=%{x:.1f}/min<br>Supervivencia=%{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="#7d8882", annotation_text="50 %")
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TEXT),
        title="Tasa de llegada y probabilidad de supervivencia",
        xaxis=dict(title="Llegadas esperadas por minuto", gridcolor=GRID),
        yaxis=dict(title="Supervivencia (%)", gridcolor=GRID, range=[0, 100]),
    )
    return fig


def model_comparison_figure(summary: pd.DataFrame) -> go.Figure:
    """Compara ambos modelos con barras e intervalos de confianza al 95 %."""

    labels = summary["model"].map({"poisson": "Poisson", "polar": "Polar-lognormal"})
    probabilities = summary["survival_probability"] * 100
    fig = go.Figure(go.Bar(
        x=labels,
        y=probabilities,
        marker_color=[POISSON_COLOR, POLAR_COLOR],
        error_y=dict(
            type="data",
            symmetric=False,
            array=summary["ci_high"] * 100 - probabilities,
            arrayminus=probabilities - summary["ci_low"] * 100,
        ),
        text=[f"{value:.1f} %" for value in probabilities],
        textposition="outside",
        hovertemplate="%{x}<br>Supervivencia=%{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TEXT),
        title="Comparación con IC de Wilson al 95 %",
        yaxis=dict(title="Supervivencia (%)", range=[0, 108], gridcolor=GRID),
        xaxis=dict(title="Modelo de llegadas"),
    )
    return fig

"""Interfaz web del proyecto Outbreak: Stochastic Survival Lab.

Este archivo actúa como capa de presentación y orquestación. No implementa la
matemática del proceso ni crea directamente geometría Plotly: delega esas tareas
a ``simulation.py`` y ``visuals.py``. Sus responsabilidades son:

* recoger y validar indirectamente la configuración del formulario;
* conservar el último escenario ejecutado en ``st.session_state``;
* cachear simulaciones costosas con entradas serializables;
* organizar la narrativa académica en pestañas;
* presentar tablas, indicadores y figuras interactivas.

Ejecución recomendada desde ``zombie_poisson_streamlit``::

    streamlit run App/app.py

Las rutas de recursos se calculan a partir de ``__file__``. Por ello, abrir la
aplicación desde otro directorio no rompe las imágenes del encabezado.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Resolve both imports and resources from the file location. Streamlit normally
# places the script directory on ``sys.path``, but explicit resolution also
# supports IDEs, test harnesses and launch commands issued from any directory.
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
ASSETS_DIR = PROJECT_DIR / "assets"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from simulation import (
    GeneratorValidationResult,
    SimulationConfig,
    compare_arrival_models,
    compare_normal_generators,
    config_as_dict,
    estimate_survival_probability,
    normal_generator_decision_matrix,
    paired_comparison_statistics,
    simulate,
    summarize_batch,
    survival_curve,
    validate_arrival_generators,
)
from visuals import (
    arena_3d,
    arrival_process_figure,
    generator_validation_figure,
    interarrival_figure,
    model_comparison_figure,
    monte_carlo_figure,
    normal_method_comparison_figure,
    paired_effects_figure,
    survival_curve_figure,
    timeline_figure,
)


# The hero follows the same cwd-independent resource convention.
HERO_IMAGE = ASSETS_DIR / "outbreak-command-center.png"

st.set_page_config(
    page_title="Outbreak | Stochastic Survival Lab",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit ofrece temas globales, pero la identidad visual del laboratorio
# necesita componentes específicos (cabecera, tarjetas y estados terminales).
# Los selectores se concentran aquí para que la lógica Python permanezca limpia.
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Share+Tech+Mono&display=swap');
:root {
    --bg:#070908; --panel:#0d1210; --panel2:#121915; --line:#29362f;
    --text:#e8eee9; --muted:#92a29a; --lime:#c7ff73; --green:#9fe870;
    --red:#ff5964; --amber:#f7a64a; --blue:#68b8d8;
    --font-display:'Rajdhani', sans-serif;
    --font-mono:'Share Tech Mono', monospace;
}
.stApp {
    background:
      radial-gradient(circle at 18% -8%, rgba(113,155,79,.16), transparent 32%),
      radial-gradient(circle at 95% 8%, rgba(137,39,46,.13), transparent 28%),
      linear-gradient(180deg,#060806 0%,#090d0a 100%);
    color:var(--text);
}
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0c110e,#080b09);
    border-right:1px solid var(--line);
}
[data-testid="stSidebar"] .block-container {padding-top:1.2rem}
.block-container {padding-top:1.3rem;padding-bottom:4rem;max-width:1540px}
.eyebrow {color:var(--lime);font-size:.72rem;letter-spacing:.19em;text-transform:uppercase;font-weight:800;font-family:var(--font-display)}
.hero-copy {padding:18px 0 8px}
.hero-copy h1 {font-family:var(--font-display);font-weight:700;font-size:clamp(2.5rem,5vw,5.4rem);line-height:.88;letter-spacing:-.02em;margin:.45rem 0 1rem}
.hero-copy p {color:var(--muted);font-size:1.02rem;line-height:1.7;max-width:760px}
.hero-copy .rule {width:72px;height:4px;background:var(--lime);border-radius:10px;margin:1.3rem 0}
.stImage img {border-radius:22px;border:1px solid #334139;box-shadow:0 22px 70px rgba(0,0,0,.36)}
.section-title {margin:1.2rem 0 .7rem;padding-bottom:.55rem;border-bottom:1px solid var(--line)}
.section-title small {display:block;color:var(--lime);letter-spacing:.16em;text-transform:uppercase;font-weight:800;font-size:.68rem;font-family:var(--font-display)}
.section-title strong {font-size:1.3rem;letter-spacing:0;font-family:var(--font-display);font-weight:700}
.status-strip {border:1px solid var(--line);border-left:4px solid var(--lime);border-radius:14px;padding:13px 16px;background:#0e1511;margin:.4rem 0 1rem}
.status-strip.failed {border-left-color:var(--red)}
.callout {border:1px solid var(--line);border-radius:14px;padding:15px 17px;background:#0d1410;color:#c8d3cc;margin:.7rem 0 1rem}
.callout.blue {border-left:3px solid var(--blue)}
.callout.amber {border-left:3px solid var(--amber)}
.model-card {height:100%;border:1px solid var(--line);border-radius:18px;padding:18px;background:linear-gradient(180deg,#111813,#0c110e);transition:transform .15s ease, box-shadow .15s ease}
.model-card:hover {transform:translateY(-3px);box-shadow:0 16px 38px rgba(0,0,0,.4)}
.model-card h3 {margin:.2rem 0 .6rem;font-family:var(--font-display);font-weight:700}.model-card p {color:var(--muted);line-height:1.55}
/* Streamlit agrega un icono de enlace ancla a cualquier h1-h6, incluso dentro
   de HTML propio (como el título del hero). Se oculta en toda la app: es
   ruido visual aquí, no navegación real. */
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {display:none !important}
.tag {display:inline-block;border:1px solid #3a4b40;border-radius:999px;padding:.24rem .55rem;margin:.3rem .25rem .2rem 0;color:#bac8c0;font-size:.74rem;font-family:var(--font-display);letter-spacing:.02em}
[data-testid="stMetric"] {background:linear-gradient(180deg,#111813,#0c120e);border:1px solid var(--line);padding:13px 15px;border-radius:15px}
[data-testid="stMetricLabel"] {color:#9caca3}[data-testid="stMetricValue"] {color:#f0f5f1;letter-spacing:-.01em;font-family:var(--font-mono)}
.stButton>button,.stFormSubmitButton>button {border-radius:12px;border:1px solid #587345;background:linear-gradient(180deg,#21301e,#151f16);color:#e9f7df;font-weight:800;min-height:46px;letter-spacing:.025em}
.stButton>button:hover,.stFormSubmitButton>button:hover {border-color:var(--lime);color:white}

/* --- Centro de control: pulido puramente visual, mismo orden, misma paleta --- */
/* "Modelo de llegadas" como selector segmentado en vez de radios sueltos. */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    display:flex;gap:.35rem;background:var(--panel2);
    border:1px solid var(--line);border-radius:12px;padding:.3rem;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    flex:1;margin:0 !important;padding:.4rem .5rem !important;
    border-radius:9px;transition:background .15s ease,box-shadow .15s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background:linear-gradient(180deg,#21301e,#151f16);
    box-shadow:inset 0 0 0 1px #587345;
}
/* "Restablecer valores por defecto" queda como acción secundaria (ghost),
   sin competir visualmente con "SIMULAR MISIÓN". Solo afecta a st.button,
   no a st.form_submit_button, así que no toca el botón principal. */
[data-testid="stSidebar"] [data-testid="stButton"] button {
    background:transparent;border:1px solid var(--line);color:var(--muted);
}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
    border-color:var(--lime);color:var(--lime);background:rgba(199,255,115,.06);
}
/* Acento de color por expander, reutilizando colores ya definidos en :root
   (azul=sobreviviente, rojo=infectados, ámbar=precisión) para escanear más
   rápido cuál es cuál sin leer el título completo. */
[data-testid="stSidebar"] [data-testid="stExpander"] {border-radius:14px;overflow:hidden}
[data-testid="stSidebar"] [data-testid="stExpander"]:nth-of-type(1) {border-left:3px solid var(--blue)}
[data-testid="stSidebar"] [data-testid="stExpander"]:nth-of-type(2) {border-left:3px solid var(--red)}
[data-testid="stSidebar"] [data-testid="stExpander"]:nth-of-type(3) {border-left:3px solid var(--amber)}
/* Leve resplandor en el control activo de cada slider, sensación más táctil.
   (el estilo completo de la pista se define más abajo, para toda la app) */
/* Scrollbar del sidebar a juego con el tema (solo navegadores basados en Chromium/WebKit). */
[data-testid="stSidebar"] ::-webkit-scrollbar {width:8px}
[data-testid="stSidebar"] ::-webkit-scrollbar-track {background:transparent}
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {background:var(--line);border-radius:8px}
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb:hover {background:#3a4b40}
/* Pequeña barra de acento bajo "Configurar misión", igual a la que ya usa
   el título del hero, para que ambos títulos se sientan de la misma familia. */
[data-testid="stSidebar"] h2::after {
    content:"";display:block;width:42px;height:3px;
    background:var(--lime);border-radius:10px;margin-top:.5rem;
}
/* Input de semilla con las mismas esquinas redondeadas que el resto del panel. */
[data-testid="stSidebar"] [data-testid="stNumberInput"] > div {
    border-radius:10px;border-color:var(--line) !important;
}
/* Los iconos "?" de ayuda se iluminan en lima al pasar el cursor, en vez de
   quedarse en el gris apagado por defecto. */
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] {transition:color .15s ease}
[data-testid="stSidebar"] [data-testid="stTooltipIcon"]:hover {color:var(--lime) !important}

/* "Barritas" (sliders) en toda la app: pista con extremos redondeados y algo
   más de cuerpo, y el control (thumb) con un anillo oscuro para que resalte
   sobre la pista en vez de fundirse con el verde del relleno. */
/* "Barritas" (sliders): el control (thumb) con un anillo oscuro para que
   resalte sobre la pista en vez de fundirse con el verde del relleno.
   (Se probó también redondear/engrosar la pista, pero eso reveló las
   etiquetas de mínimo/máximo que Streamlit mantiene ocultas por defecto —
   se descartó esa parte por no ser un efecto confiable.) */
[data-testid="stSlider"] [role="slider"] {
    box-shadow:0 0 0 4px rgba(199,255,115,.18),0 2px 6px rgba(0,0,0,.45);
    border:2px solid var(--bg) !important;
}

.stTabs [data-baseweb="tab-list"] {gap:.42rem;border-bottom:1px solid var(--line);padding-bottom:.55rem}
.stTabs [data-baseweb="tab"] {background:#0e1510;border:1px solid var(--line);border-radius:10px;padding:.62rem .95rem}
.stTabs [aria-selected="true"] {border-color:#738f5d!important;color:var(--lime)!important}
code {color:#d8ffae!important;font-family:var(--font-mono)!important} hr {border-color:var(--line)!important}
@media(max-width:900px){
    .hero-copy h1{font-size:3rem}
    .block-container{padding-top:.8rem}
    [data-testid="stHorizontalBlock"]{flex-wrap:wrap;row-gap:.6rem}
    [data-testid="stHorizontalBlock"] [data-testid="column"]{min-width:46%}
    .stTabs [data-baseweb="tab-list"]{flex-wrap:wrap}
}
@media(max-width:560px){
    [data-testid="stHorizontalBlock"] [data-testid="column"]{min-width:100%}
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def run_main(config_dict: dict[str, object]):
    """Ejecuta una partida y memoriza el resultado por configuración exacta.

    Se recibe un diccionario porque es una entrada sencilla y estable para el
    sistema de hashing de Streamlit. La función reconstruye la dataclass antes de
    entrar al motor.
    """

    return simulate(SimulationConfig(**config_dict), keep_timeline=True)


@st.cache_data(show_spinner=False)
def run_monte_carlo(config_dict: dict[str, object], runs: int) -> pd.DataFrame:
    """Cachea un lote Monte Carlo del modelo actualmente seleccionado."""

    return estimate_survival_probability(SimulationConfig(**config_dict), runs=runs)


@st.cache_data(show_spinner=False)
def run_comparison(
    config_dict: dict[str, object],
    runs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cachea corridas, resúmenes y contrastes pareados de ambos modelos."""

    trials, summary = compare_arrival_models(
        SimulationConfig(**config_dict), runs=runs
    )
    effects, contingency = paired_comparison_statistics(trials)
    return trials, summary, effects, contingency


@st.cache_data(show_spinner=False)
def run_validation(
    config_dict: dict[str, object],
    sample_size: int,
    count_repetitions: int,
) -> GeneratorValidationResult:
    """Cachea la auditoría estadística para no repetir muestras al navegar."""

    return validate_arrival_generators(
        SimulationConfig(**config_dict),
        sample_size=sample_size,
        count_repetitions=count_repetitions,
    )


@st.cache_data(show_spinner=False)
def run_normal_method_comparison(
    sample_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compara dos transformaciones normales bajo un protocolo reproducible.

    La comparación se separa de la validación de interarribos porque aquí
    Marsaglia Polar y Box--Muller persiguen exactamente la misma distribución
    ``N(0,1)``. Esto permite atribuir las diferencias observadas al algoritmo
    de generación y no a modelos de llegada distintos.
    """

    tests, performance = compare_normal_generators(
        sample_size=sample_size,
        benchmark_repetitions=9,
        seed=77_411,
    )
    decision = normal_generator_decision_matrix(tests, performance)
    return tests, performance, decision


@st.cache_data(show_spinner=False)
def run_curve(
    config_dict: dict[str, object], lambdas: tuple[float, ...], runs_per_lambda: int
) -> pd.DataFrame:
    """Cachea una curva de supervivencia para una cuadrícula inmutable de tasas."""

    return survival_curve(
        SimulationConfig(**config_dict), list(lambdas), runs_per_lambda=runs_per_lambda
    )


def section_header(kicker: str, title: str) -> None:
    """Dibuja un encabezado de sección consistente en todas las pestañas."""

    st.markdown(
        f'<div class="section-title"><small>{kicker}</small><strong>{title}</strong></div>',
        unsafe_allow_html=True,
    )


# Configuración compartida para todos los gráficos Plotly: oculta el logo y
# los botones de la barra de herramientas que no aportan a esta demo (lasso,
# selección de caja, comparar al pasar el cursor, rotación de tabla), tanto
# para las figuras 2D como para la escena 3D.
PLOTLY_CONFIG: dict[str, object] = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "select2d",
        "lasso2d",
        "autoScale2d",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
        "toggleSpikelines",
        "hoverClosest3d",
        "tableRotation",
        "resetCameraLastSave3d",
    ],
}


# ---------------------------------------------------------------------------
# Panel de configuración
# ---------------------------------------------------------------------------
# Valores por defecto de cada widget del sidebar. Se usan para el primer
# render y para el botón "Restablecer valores por defecto".
SIDEBAR_DEFAULTS: dict[str, object] = {
    "model_label_selector": "Poisson",
    "last_polar_cv": 0.60,
    "arrivals_per_minute": 55.8,
    "duration": 90,
    "player_hp": 110,
    "player_dps": 42,
    "weapon_range": 10.0,
    "enemy_hp": 42,
    "enemy_speed": 1.55,
    "enemy_dps": 9.5,
    "seed": 22193,
    "dt": 0.05,
}

# El formulario agrupa widgets para evitar que cada movimiento de un deslizador
# dispare una simulación. Solo el botón final confirma y publica la configuración.
with st.sidebar:
    st.markdown('<div class="eyebrow">Centro de control</div>', unsafe_allow_html=True)
    st.markdown("## Configurar misión")
    st.caption(
        "Elige el modelo de llegadas y luego ajusta el resto de parámetros; "
        "se aplican al pulsar el botón final del formulario."
    )

    # El modelo vive fuera del form porque decide si el slider de CV Polar
    # tiene sentido mostrarse. Al estar fuera, cambia de inmediato al tocarlo,
    # pero la configuración simulada sigue sin actualizarse hasta el envío.
    model_label = st.radio(
        "Modelo de llegadas",
        ["Poisson", "Polar-lognormal"],
        help="Poisson usa interarribos exponenciales. Polar-lognormal usa normales de Marsaglia para construir tiempos positivos.",
        key="model_label_selector",
    )

    with st.form("mission_configuration", border=False):
        if model_label == "Polar-lognormal":
            # El widget se destruye cuando el modelo es Poisson, así que Streamlit
            # olvida su valor. Lo recordamos aparte para no perder el ajuste del
            # usuario al ir y venir entre modelos.
            polar_cv = st.slider(
                "Variabilidad Polar (CV)",
                0.15,
                1.20,
                st.session_state.get("last_polar_cv", SIDEBAR_DEFAULTS["last_polar_cv"]),
                0.05,
                help="Un CV menor genera llegadas más regulares.",
            )
            st.session_state["last_polar_cv"] = polar_cv
        else:
            polar_cv = st.session_state.get("last_polar_cv", SIDEBAR_DEFAULTS["last_polar_cv"])
            st.caption("La variabilidad Polar (CV) no aplica: el modelo activo es Poisson.")

        arrivals_per_minute = st.slider(
            "Llegadas esperadas por minuto",
            min_value=6.0,
            max_value=90.0,
            value=SIDEBAR_DEFAULTS["arrivals_per_minute"],
            step=1.0,
            help="Se convierte internamente a lambda por segundo.",
            key="arrivals_per_minute",
        )
        duration = st.slider(
            "Duración de la misión (s)",
            30,
            180,
            SIDEBAR_DEFAULTS["duration"],
            10,
            help="Horizonte temporal que el sobreviviente debe resistir.",
            key="duration",
        )

        with st.expander("Capacidad del sobreviviente", expanded=False):
            player_hp = st.slider(
                "Vida inicial (HP)", 60, 200, SIDEBAR_DEFAULTS["player_hp"], 5,
                help="Puntos de vida con los que arranca el sobreviviente.",
                key="player_hp",
            )
            player_dps = st.slider(
                "Daño por segundo", 15, 90, SIDEBAR_DEFAULTS["player_dps"], 1,
                help="Daño que inflige el sobreviviente por segundo a su objetivo.",
                key="player_dps",
            )
            weapon_range = st.slider(
                "Alcance del arma (m)", 5.0, 15.0, SIDEBAR_DEFAULTS["weapon_range"], 0.5,
                help="Distancia máxima a la que el arma puede atacar a un infectado.",
                key="weapon_range",
            )

        with st.expander("Características de los infectados", expanded=False):
            enemy_hp = st.slider(
                "Vida base (HP)", 20, 100, SIDEBAR_DEFAULTS["enemy_hp"], 2,
                help="Vida promedio de cada infectado antes de aplicar variación individual.",
                key="enemy_hp",
            )
            enemy_speed = st.slider(
                "Velocidad base (m/s)", 0.8, 3.0, SIDEBAR_DEFAULTS["enemy_speed"], 0.05,
                help="Velocidad de avance promedio de los infectados hacia el sobreviviente.",
                key="enemy_speed",
            )
            enemy_dps = st.slider(
                "Daño de contacto por segundo", 3.0, 20.0, SIDEBAR_DEFAULTS["enemy_dps"], 0.5,
                help="Daño que inflige cada infectado en contacto con el sobreviviente.",
                key="enemy_dps",
            )

        with st.expander("Reproducibilidad y precisión", expanded=False):
            seed = st.number_input(
                "Semilla", 1, 999999, SIDEBAR_DEFAULTS["seed"], 1,
                help="Fija el generador aleatorio: misma semilla, misma partida.",
                key="seed",
            )
            dt = st.select_slider(
                "Paso temporal dt (s)", options=[0.10, 0.05, 0.025], value=SIDEBAR_DEFAULTS["dt"],
                help="Paso de integración del combate. Menor dt = más precisión, más costo.",
                key="dt",
            )

        submitted = st.form_submit_button(
            "SIMULAR MISIÓN", type="primary", width="stretch"
        )

    if submitted:
        # La interfaz trabaja en llegadas/minuto porque es más legible; el motor
        # mantiene segundos como unidad base y recibe lambda dividido entre 60.
        selected_model = "poisson" if model_label == "Poisson" else "polar"
        new_config = SimulationConfig(
            duration=float(duration),
            lambda_rate=float(arrivals_per_minute / 60.0),
            arrival_model=selected_model,
            polar_cv=float(polar_cv),
            player_hp=float(player_hp),
            player_dps=float(player_dps),
            weapon_range=float(weapon_range),
            enemy_hp=float(enemy_hp),
            enemy_speed=float(enemy_speed),
            enemy_dps=float(enemy_dps),
            dt=float(dt),
            seed=int(seed),
        )
        # Guardar una instantánea separa valores editados de valores simulados.
        # También invalida experimentos que ya no describen el escenario activo.
        st.session_state["active_config"] = config_as_dict(new_config)
        st.session_state["simulation_count"] = st.session_state.get("simulation_count", 0) + 1
        st.session_state.pop("comparison_result", None)
        st.session_state.pop("mc_result", None)
        st.session_state.pop("validation_result", None)

    # Fuera del form porque debe reaccionar de inmediato, sin esperar a
    # "SIMULAR MISIÓN". Solo reescribe los widgets del panel; no toca la
    # partida ya simulada hasta que se vuelva a pulsar el botón principal.
    # Usa on_click porque Streamlit no permite reescribir session_state de un
    # widget (p. ej. el radio del modelo) después de que ya se instanció en
    # el mismo run; el callback corre antes de que eso vuelva a suceder.
    def _restore_sidebar_defaults() -> None:
        for default_key, default_value in SIDEBAR_DEFAULTS.items():
            st.session_state[default_key] = default_value

    st.button(
        "Restablecer valores por defecto",
        width="stretch",
        on_click=_restore_sidebar_defaults,
    )

    st.markdown("---")
    st.markdown(
        '<div class="callout blue"><b>Consejo</b><br>'
        "Mantén la misma semilla para reproducir una partida o cámbiala para "
        "observar otra realización.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cabecera editorial y estado inicial
# ---------------------------------------------------------------------------
hero_left, hero_right = st.columns([1.15, 1], gap="large", vertical_alignment="center")
with hero_left:
    st.markdown(
        """
        <div class="hero-copy">
          <div class="eyebrow">Laboratorio de Modelación y Simulación</div>
          <h1>OUTBREAK:<br>STOCHASTIC<br>SURVIVAL</h1>
          <div class="rule"></div>
          <p>
            Un experimento reproducible sobre llegadas aleatorias, presión de combate y
            probabilidad de supervivencia. Compara un proceso de Poisson con una alternativa
            de renovación construida mediante el método polar de Marsaglia.
          </p>
          <span class="tag">Proceso de Poisson</span>
          <span class="tag">Marsaglia Polar</span>
          <span class="tag">Monte Carlo</span>
          <span class="tag">Escena 3D procedural</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hero_right:
    if HERO_IMAGE.exists():
        st.image(HERO_IMAGE, width="stretch")


# No ejecutar automáticamente al abrir la página: el usuario debe comprender
# que los controles no tienen efecto hasta pulsar "SIMULAR MISIÓN".
if "active_config" not in st.session_state:
    st.markdown(
        """
        <div class="status-strip">
          <strong>Simulador preparado.</strong><br>
          1&#41; Elige el modelo y ajusta los parámetros en el centro de control &nbsp;&middot;&nbsp;
          2&#41; Pulsa <b>SIMULAR MISIÓN</b> &nbsp;&middot;&nbsp;
          3&#41; Explora las pestañas para ver la escena, la matemática y las pruebas.
        </div>
        """,
        unsafe_allow_html=True,
    )
    section_header("Antes de comenzar", "Qué observar durante el experimento")
    a, b, c = st.columns(3)
    a.markdown("**Llegadas**\n\nCada enemigo aparece en un instante aleatorio acumulado.")
    b.markdown("**Combate**\n\nEl sistema actualiza movimiento, ataque y daño cada `dt` segundos.")
    c.markdown("**Inferencia**\n\nMuchas partidas permiten estimar la probabilidad de sobrevivir.")
    st.caption(
        "¿Buscas el contexto académico completo (objetivos, variables, supuestos)? "
        "Está en la pestaña **Guía del proyecto**, disponible después de la primera partida."
    )
    st.stop()


# A partir de este punto siempre existe una configuración confirmada. La caché
# evita repetir trabajo cuando una interacción solo cambia la vista temporal.
config = SimulationConfig(**st.session_state["active_config"])
result = run_main(config_as_dict(config))
model_name = "Poisson" if config.arrival_model == "poisson" else "Polar-lognormal"
status_class = "" if result.survived else " failed"
status_title = "MISIÓN COMPLETADA" if result.survived else "MISIÓN FALLIDA"
status_detail = (
    "El sobreviviente alcanzó el horizonte temporal."
    if result.survived
    else "El HP llegó a cero antes del horizonte temporal."
)
st.markdown(
    f'<div class="status-strip{status_class}"><strong>{status_title}</strong> | '
    f'{model_name} | semilla {config.seed}<br>{status_detail}</div>',
    unsafe_allow_html=True,
)

metrics = st.columns(6)
metrics[0].metric("Tiempo resistido", f"{result.survival_time:.1f} s", f"de {config.duration:.0f} s")
metrics[1].metric("HP final", f"{result.final_hp:.1f}", f"{result.final_hp-config.player_hp:+.1f}")
count_reference = (
    f"E[N]={result.expected_arrivals:.1f}"
    if config.arrival_model == "poisson"
    else f"lambda*T ref.={result.expected_arrivals:.1f}"
)
metrics[2].metric("Llegadas ocurridas", result.generated, count_reference)
metrics[3].metric("Eliminados", result.eliminated, f"{100*result.eliminated/max(result.generated,1):.0f} %")
metrics[4].metric("Activos al cierre", result.remaining)
metrics[5].metric("Pico simultáneo", result.max_concurrent)

st.caption(
    "Empieza por la escena interactiva y avanza hacia la matemática, la validación "
    "estadística y la comparación entre modelos."
)

# Las pestañas siguen el orden de una investigación: observar, fundamentar,
# validar los generadores, comparar los modelos e inferir mediante repetición.
simulation_tab, math_tab, validation_tab, comparison_tab, monte_carlo_tab, guide_tab = st.tabs([
    "Simulación 3D",
    "Desarrollo matemático",
    "Validación de generadores",
    "Comparación de modelos",
    "Laboratorio Monte Carlo",
    "Guía del proyecto",
])


with simulation_tab:
    # El deslizador no vuelve a simular; selecciona la muestra temporal más
    # cercana y reconstruye posiciones desde los tiempos de nacimiento y muerte.
    section_header("Escena interactiva", "Explorar la partida en el tiempo")
    left, right = st.columns([2.05, 1], gap="large")
    with left:
        t_view = st.slider(
            "Instante de observación (s)",
            0.0,
            float(max(result.survival_time, 0.1)),
            float(max(result.survival_time, 0.1)),
            0.5,
            key="timeline_explorer",
        )
        st.plotly_chart(arena_3d(result, t_view), use_container_width=True, config=PLOTLY_CONFIG)
        st.caption(
            "Arrastra para rotar, usa la rueda para acercar y pasa el cursor sobre los infectados. "
            "El anillo verde marca el alcance; el rojo, la zona de contacto."
        )
    with right:
        closest_row = result.timeline.iloc[(result.timeline["time"] - t_view).abs().argmin()]
        st.markdown("### Estado seleccionado")
        state_cols = st.columns(2)
        state_cols[0].metric("HP", f"{closest_row['player_hp']:.1f}")
        state_cols[1].metric("Activos", int(closest_row["active_enemies"]))
        state_cols[0].metric("Llegadas", int(closest_row["spawned"]))
        state_cols[1].metric("Bajas", int(closest_row["eliminated"]))
        st.markdown(
            f"""
            <div class="callout amber">
              <b>Lectura de lambda</b><br>
              {config.lambda_rate*60:.1f} llegadas/min implica un interarribo medio de
              {1/config.lambda_rate:.2f} s. El modelo activo es <b>{model_name}</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("**Reglas operativas**")
        st.markdown(
            """
            1. El jugador permanece en el centro.
            2. Ataca al objetivo vivo más cercano dentro del alcance.
            3. Los infectados avanzan radialmente y dañan al contactar.
            4. Los atacantes en contacto suman su daño.
            5. La misión termina al agotar HP o alcanzar el tiempo objetivo.
            """
        )
    st.plotly_chart(timeline_figure(result), use_container_width=True, config=PLOTLY_CONFIG)


with math_tab:
    # Esta pestaña muestra el procedimiento aplicado a la semilla activa, no un
    # ejemplo desconectado. Por eso las tablas provienen de ``result.arrivals``.
    section_header("Modelo estocástico", "De uniformes a instantes de aparición")
    st.markdown(
        """
        <div class="callout blue">
          El conteo <b>N(t)</b> es discreto. Los tiempos entre llegadas <b>Delta</b>
          son variables aleatorias continuas. La simulación conecta ambos mediante
          la suma acumulada de interarribos.
        </div>
        """,
        unsafe_allow_html=True,
    )
    poisson_col, polar_col = st.columns(2)
    with poisson_col:
        st.markdown(
            """
            <div class="model-card">
              <div class="eyebrow">Modelo 1</div><h3>Proceso de Poisson</h3>
              <p>Llegadas independientes, tasa constante y propiedad de falta de memoria.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.latex(r"N(t)\sim\operatorname{Poisson}(\lambda t)")
        st.latex(r"P\{N(t)=k\}=e^{-\lambda t}\frac{(\lambda t)^k}{k!}")
        st.latex(r"\Delta_i=-\frac{\ln(U_i)}{\lambda},\quad U_i\sim U(0,1)")
        st.latex(r"\mathbb E[N(t)]=\operatorname{Var}[N(t)]=\lambda t")
    with polar_col:
        st.markdown(
            """
            <div class="model-card">
              <div class="eyebrow">Modelo 2</div><h3>Renovación Polar-lognormal</h3>
              <p>Marsaglia Polar genera normales; una transformación lognormal produce interarribos positivos.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.latex(r"V_1,V_2\sim U(-1,1),\quad S=V_1^2+V_2^2<1")
        st.latex(r"Z=V_1\sqrt{\frac{-2\ln S}{S}}\sim N(0,1)")
        st.latex(r"\Delta_i=e^{\mu+\sigma Z_i}")
        st.latex(r"\sigma^2=\ln(1+CV^2),\quad \mu=\ln(1/\lambda)-\sigma^2/2")

    st.markdown("### Construcción común")
    st.latex(r"t_0=0,\qquad t_i=t_{i-1}+\Delta_i")
    theory = st.columns(4)
    theory[0].metric("Tasa", f"{config.lambda_rate*60:.1f}/min")
    theory[1].metric("E[Delta]", f"{1/config.lambda_rate:.3f} s")
    count_label = "E[N(T)]" if config.arrival_model == "poisson" else "lambda*T (referencia)"
    theory[2].metric(count_label, f"{config.lambda_rate*config.duration:.1f}")
    theory[3].metric("Paso dt", f"{config.dt:.3f} s")

    st.markdown("### Primeras variables generadas")
    if result.arrivals.empty:
        st.info("No ocurrieron llegadas antes de finalizar la misión.")
    elif config.arrival_model == "poisson":
        table = result.arrivals[["enemy_id", "u", "delta", "arrival_time"]].head(20).copy()
        table.columns = ["Enemigo", "U", "Delta (s)", "t (s)"]
        st.dataframe(table.style.format({"U": "{:.6f}", "Delta (s)": "{:.4f}", "t (s)": "{:.4f}"}), width="stretch", hide_index=True)
    else:
        table = result.arrivals[["enemy_id", "polar_v1", "polar_v2", "z", "delta", "arrival_time"]].head(20).copy()
        table.columns = ["Enemigo", "V1", "V2", "Z", "Delta (s)", "t (s)"]
        st.dataframe(table.style.format({"V1": "{:.5f}", "V2": "{:.5f}", "Z": "{:.5f}", "Delta (s)": "{:.4f}", "t (s)": "{:.4f}"}), width="stretch", hide_index=True)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(arrival_process_figure(result), use_container_width=True, config=PLOTLY_CONFIG)
    with chart_right:
        st.plotly_chart(interarrival_figure(result), use_container_width=True, config=PLOTLY_CONFIG)
    st.caption(
        "El histograma usa interarribos observados antes del cierre y sirve como diagnóstico visual; "
        "la truncación temporal puede sesgar una muestra pequeña y no sustituye una prueba formal."
    )


with validation_tab:
    # Este laboratorio usa muestras fijas ajenas al horizonte de la misión.
    # Así se evita confundir la censura temporal con un defecto del generador.
    section_header("Control de calidad estadístico", "¿Generamos las distribuciones declaradas?")
    st.markdown(
        """
        La auditoría recorre la cadena en el mismo orden en que se construyen
        las variables. **Primero la fuente uniforme**, porque tanto la transformada
        inversa como Marsaglia Polar son funciones de uniformes y un defecto ahí
        contamina todo lo demás: se contrastan forma marginal, discrepancia máxima,
        orden secuencial, correlación conjunta y estructura en el cubo, tanto sobre
        PCG64 como sobre un LCG propio. **Después las transformaciones** y, por
        último, **el conteo del calendario**.

        Ningún contraste repite la evidencia de otro. En particular no se prueba la
        bondad de ajuste de la lognormal: Kolmogorov--Smirnov es invariante ante
        transformaciones monótonas, así que devolvería exactamente el mismo número
        que el KS de `Z`. Lo que ese KS no puede ver —si la parametrización cumple
        `E[Delta] = 1/lambda`— se contrasta aparte con la media.

        Si los 18 contrastes fueran independientes y todas sus hipótesis nulas
        fueran ciertas, leer solo el valor p nominal daría una probabilidad
        ilustrativa cercana al 60 % de al menos un falso rechazo. Como varias
        pruebas comparten muestras, esa cifra no es una probabilidad exacta del
        experimento. La decisión se toma sobre la columna corregida por Holm,
        procedimiento que no exige independencia. No rechazar no demuestra perfección;
        indica que la muestra no contradice el modelo bajo el contraste utilizado.
        """
    )
    sample_col, count_col = st.columns(2)
    validation_sample_size = sample_col.select_slider(
        "Interarribos por generador",
        options=[1_000, 2_000, 4_000, 8_000],
        value=4_000,
        key="validation_sample_size",
    )
    validation_count_repetitions = count_col.select_slider(
        "Calendarios para validar N(T)",
        options=[200, 400, 600, 1_000],
        value=600,
        key="validation_count_repetitions",
    )
    validation_signature = (
        tuple(sorted(config_as_dict(config).items())),
        int(validation_sample_size),
        int(validation_count_repetitions),
    )
    if st.button("EJECUTAR VALIDACIÓN ESTADÍSTICA", type="primary", width="stretch"):
        with st.spinner("Generando muestras y ejecutando contrastes..."):
            validation = run_validation(
                config_as_dict(config),
                int(validation_sample_size),
                int(validation_count_repetitions),
            )
            normal_tests, normal_performance, normal_decision = (
                run_normal_method_comparison(max(20_000, int(validation_sample_size)))
            )
        st.session_state["validation_result"] = {
            "signature": validation_signature,
            "validation": validation,
            "normal_tests": normal_tests,
            "normal_performance": normal_performance,
            "normal_decision": normal_decision,
        }

    validation_saved = st.session_state.get("validation_result")
    if validation_saved is None:
        st.info("Ejecuta la validación para producir valores p, momentos, gráficos Q-Q y benchmark.")
    elif validation_saved["signature"] != validation_signature:
        st.warning("Cambió la configuración del laboratorio. Ejecuta nuevamente.")
    else:
        validation = validation_saved["validation"]
        passed = int(validation.tests["passes_holm"].sum())
        passed_nominal = int(validation.tests["passes"].sum())
        total = len(validation.tests)
        validation_metrics = st.columns(4)
        validation_metrics[0].metric(
            "No rechazados (Holm)",
            f"{passed}/{total}",
            f"{passed_nominal}/{total} sin corregir",
        )
        validation_metrics[1].metric(
            "Media exponencial",
            f"{validation.moments.iloc[0]['mean_observed']:.3f} s",
            f"teórica {validation.moments.iloc[0]['mean_theoretical']:.3f}",
        )
        validation_metrics[2].metric(
            "CV lognormal",
            f"{validation.moments.iloc[1]['cv_observed']:.3f}",
            f"objetivo {validation.moments.iloc[1]['cv_theoretical']:.3f}",
        )
        polar_speed = float(
            validation.performance.loc[
                validation.performance["generator"] == "Polar-lognormal",
                "microseconds_per_value",
            ].iloc[0]
        )
        validation_metrics[3].metric("Costo Polar", f"{polar_speed:.2f} us/valor")
        if passed == total:
            st.success(
                f"Con alpha=0.05 corregido por Holm, ninguno de los {total} contrastes "
                "rechaza el comportamiento esperado de los generadores."
            )
        else:
            st.warning(
                "Al menos un contraste fue rechazado tras la corrección por multiplicidad. "
                "Revisa el valor p ajustado y el generador implicado antes de concluir; "
                "cambiar de semilla hasta que ninguno falle no es un método válido."
            )

        st.plotly_chart(
            generator_validation_figure(
                validation, config.lambda_rate, config.polar_cv
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
        tests_display = validation.tests.copy()
        tests_display.columns = [
            "Generador", "Contraste", "Estadístico", "Valor p", "Alpha",
            "No se rechaza", "Valor p (Holm)", "No se rechaza (Holm)",
        ]
        st.dataframe(
            tests_display.style.format({
                "Estadístico": "{:.5f}",
                "Valor p": "{:.5f}",
                "Alpha": "{:.2f}",
                "Valor p (Holm)": "{:.5f}",
            }),
            width="stretch",
            hide_index=True,
        )

        st.markdown("### Control negativo: uniformidad no es aleatoriedad")
        st.markdown(
            """
            Una batería que solo aprueba generadores buenos no demuestra nada
            mientras no se compruebe que también reprueba a los malos. El control
            es un contador `x = (x + 1) mod m`: recorre su período completo, así
            que su histograma es exactamente plano y su ajuste a la uniforme es
            perfecto. Y aun así es totalmente predecible. Se reporta fuera de la
            tabla principal porque se espera que falle; incluirlo ahí inflaría
            artificialmente el número de contrastes rechazados.
            """
        )
        control_display = validation.uniform_control.copy()
        control_display.columns = [
            "Generador", "Contraste", "Estadístico", "Valor p", "Alpha", "No se rechaza"
        ]
        st.dataframe(
            control_display.style.format({
                "Estadístico": "{:.5f}",
                "Valor p": "{:.2e}",
                "Alpha": "{:.2f}",
            }),
            width="stretch",
            hide_index=True,
        )
        shape_survivors = int(
            validation.uniform_control["passes"].iloc[:2].sum()
        )
        st.markdown(
            f'<div class="callout amber"><b>Lectura del control</b><br>'
            f'El contador aprueba {shape_survivors} de los 2 contrastes de forma y '
            f'fracasa en los 3 de dependencia. Esa asimetría es la justificación de '
            f'por qué la validación no se agota en un histograma: la forma marginal '
            f'y la independencia son propiedades distintas y exigen pruebas distintas.</div>',
            unsafe_allow_html=True,
        )
        details_left, details_right = st.columns(2)
        with details_left:
            st.markdown("### Momentos empíricos y teóricos")
            moments_display = validation.moments.copy()
            moments_display.columns = [
                "Generador", "Media teórica", "Media observada",
                "Varianza teórica", "Varianza observada", "CV teórico", "CV observado"
            ]
            st.dataframe(
                moments_display.style.format(precision=4),
                width="stretch",
                hide_index=True,
            )
        with details_right:
            st.markdown("### Rendimiento descriptivo")
            performance_display = validation.performance.copy()
            performance_display.columns = [
                "Generador", "Tamaño", "Repeticiones", "Mediana (ms)", "us por valor"
            ]
            st.dataframe(
                performance_display.style.format({
                    "Mediana (ms)": "{:.4f}",
                    "us por valor": "{:.4f}",
                }),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "El benchmark depende del equipo y se usa como criterio de costo, no como prueba de ajuste."
            )

        st.markdown("### Comparación directa: Marsaglia Polar frente a Box--Muller")
        st.markdown(
            """
            Esta segunda comparación mantiene fija la distribución objetivo
            `N(0,1)`. Por tanto, sí contrasta dos algoritmos equivalentes:
            calidad estadística, costo vectorizado, fracción de propuestas
            aprovechadas, robustez numérica y facilidad de auditoría. Un valor
            p mayor no significa que un generador sea mejor; solo se comprueba
            si cada muestra contradice o no el modelo normal al nivel elegido.
            """
        )
        if "normal_tests" not in validation_saved:
            st.info(
                "Este resultado fue creado con una versión anterior de la app. "
                "Ejecuta nuevamente la validación para comparar ambos métodos."
            )
        else:
            normal_tests = validation_saved["normal_tests"]
            normal_performance = validation_saved["normal_performance"]
            normal_decision = validation_saved["normal_decision"]
            st.plotly_chart(
                normal_method_comparison_figure(normal_tests, normal_performance),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
            method_left, method_right = st.columns(2)
            with method_left:
                method_tests_display = normal_tests.rename(columns={
                    "method": "Método",
                    "ks_statistic": "KS D",
                    "p_value": "Valor p",
                    "passes": "No se rechaza",
                    "mean": "Media",
                    "variance": "Varianza",
                    "skewness": "Asimetría",
                    "excess_kurtosis": "Curtosis exc.",
                    "rejected_fraction": "Fracción rechazada",
                })
                st.dataframe(
                    method_tests_display.style.format(precision=5),
                    width="stretch",
                    hide_index=True,
                )
            with method_right:
                decision_display = normal_decision.rename(columns={
                    "method": "Método",
                    "weighted_total": "Puntaje ponderado",
                })
                st.dataframe(
                    decision_display.style.format(precision=3),
                    width="stretch",
                    hide_index=True,
                )
            winner = str(normal_decision.iloc[0]["method"])
            st.markdown(
                f'<div class="callout blue"><b>Decisión del escenario de referencia</b><br>'
                f'La matriz reproducible favorece a <b>{winner}</b>. Ambos métodos '
                f'siguen siendo estadísticamente válidos si KS no los rechaza; '
                f'Polar se conserva en el modelo alternativo porque ilustra de forma '
                f'explícita el método de aceptación--rechazo solicitado en el curso.</div>',
                unsafe_allow_html=True,
            )


with comparison_tab:
    # Los resultados se guardan junto a una firma. Si cambia una entrada, la UI
    # conserva el lote para inspección interna pero exige recalcularlo antes de
    # presentarlo como vigente.
    section_header("Experimento pareado", "Poisson frente a Polar-lognormal")
    st.markdown(
        """
        Ambos modelos conservan el mismo interarribo medio. Poisson permite rachas y pausas largas
        por su distribución exponencial (CV = 1). La alternativa Polar-lognormal usa el CV elegido
        para controlar regularidad. Las mismas semillas de corrida reducen ruido al comparar.
        """
    )
    comparison_runs = st.select_slider(
        "Partidas por modelo", options=[50, 100, 200, 400], value=100, key="comparison_runs"
    )
    if st.button("EJECUTAR COMPARACIÓN POISSON VS POLAR", type="primary", width="stretch"):
        with st.spinner("Ejecutando el experimento pareado..."):
            trials, summary, effects, contingency = run_comparison(
                config_as_dict(config), int(comparison_runs)
            )
        st.session_state["comparison_result"] = {
            "signature": (tuple(sorted(config_as_dict(config).items())), int(comparison_runs)),
            "trials": trials,
            "summary": summary,
            "effects": effects,
            "contingency": contingency,
        }
    comparison_saved = st.session_state.get("comparison_result")
    comparison_signature = (tuple(sorted(config_as_dict(config).items())), int(comparison_runs))
    if comparison_saved is None:
        st.info("Ejecuta la comparación para obtener resultados con la configuración actual.")
    elif comparison_saved["signature"] != comparison_signature:
        st.warning("La configuración o el número de partidas cambió. Ejecuta nuevamente.")
    else:
        summary = comparison_saved["summary"]
        trials = comparison_saved["trials"]
        effects = comparison_saved["effects"]
        contingency = comparison_saved["contingency"]
        left, right = st.columns([1.4, 1])
        with left:
            st.plotly_chart(model_comparison_figure(summary), use_container_width=True, config=PLOTLY_CONFIG)
        with right:
            display_summary = summary.copy()
            display_summary["model"] = display_summary["model"].map({"poisson": "Poisson", "polar": "Polar-lognormal"})
            display_summary["IC 95 %"] = display_summary.apply(
                lambda row: f"[{row['ci_low']*100:.1f} %, {row['ci_high']*100:.1f} %]", axis=1
            )
            display_summary = display_summary[["model", "survival_probability", "IC 95 %", "mean_survival_time", "mean_generated", "mean_max_concurrent"]]
            display_summary.columns = ["Modelo", "P(supervivencia)", "IC 95 %", "Tiempo medio", "Llegadas medias", "Pico medio"]
            st.dataframe(display_summary.style.format({"P(supervivencia)": "{:.1%}", "Tiempo medio": "{:.2f}", "Llegadas medias": "{:.2f}", "Pico medio": "{:.2f}"}), width="stretch", hide_index=True)
            poisson_p = float(summary.loc[summary["model"] == "poisson", "survival_probability"].iloc[0])
            polar_p = float(summary.loc[summary["model"] == "polar", "survival_probability"].iloc[0])
            difference = (poisson_p - polar_p) * 100
            st.markdown(
                f'<div class="callout blue"><b>Diferencia observada</b><br>'
                f'Poisson - Polar = {difference:+.1f} puntos porcentuales. '
                f'La conveniencia de Poisson se justifica por sus supuestos y parsimonia, no por garantizar mayor supervivencia.</div>',
                unsafe_allow_html=True,
            )
        with st.expander("Ver corridas del experimento"):
            st.dataframe(trials, width="stretch", hide_index=True)

        section_header("Inferencia sobre diferencias", "Efectos pareados e incertidumbre")
        effect_left, effect_right = st.columns([1.35, 1])
        with effect_left:
            st.plotly_chart(
                paired_effects_figure(effects),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
        with effect_right:
            effects_display = effects.copy()
            effects_display["metric"] = effects_display["metric"].map({
                "survived": "Supervivencia",
                "survival_time": "Tiempo resistido",
                "generated": "Llegadas",
                "max_concurrent": "Pico activo",
            })
            effects_display.columns = [
                "Métrica", "Media Poisson", "Media Polar", "Diferencia",
                "IC inferior", "IC superior", "Valor p", "Prueba"
            ]
            st.dataframe(
                effects_display.style.format({
                    "Media Poisson": "{:.4f}",
                    "Media Polar": "{:.4f}",
                    "Diferencia": "{:+.4f}",
                    "IC inferior": "{:+.4f}",
                    "IC superior": "{:+.4f}",
                    "Valor p": "{:.5f}",
                }),
                width="stretch",
                hide_index=True,
            )
            st.markdown("### Tabla de contingencia")
            st.dataframe(contingency, width="stretch", hide_index=True)

        survival_effect = effects.loc[effects["metric"] == "survived"].iloc[0]
        mcnemar_p = float(survival_effect["p_value"])
        paired_ci_low = float(survival_effect["ci_low"]) * 100.0
        paired_ci_high = float(survival_effect["ci_high"]) * 100.0
        inference = (
            "existe evidencia de una diferencia de supervivencia"
            if mcnemar_p < 0.05
            else "no hay evidencia suficiente para declarar una diferencia de supervivencia"
        )
        st.markdown(
            f'<div class="callout amber"><b>Conclusión pareada</b><br>'
            f'Con McNemar exacta, p={mcnemar_p:.5f}: {inference} al 5 %. '
            f'El IC bootstrap del efecto Poisson - Polar es '
            f'[{paired_ci_low:+.1f}, {paired_ci_high:+.1f}] puntos porcentuales.</div>',
            unsafe_allow_html=True,
        )


with monte_carlo_tab:
    # Laboratorio del modelo activo: estima p para el escenario puntual y luego
    # construye una curva alrededor de lambda con intervalos de Wilson.
    section_header("Inferencia por repetición", f"Probabilidad de supervivencia bajo {model_name}")
    run_col, curve_col = st.columns(2)
    runs = run_col.select_slider("Partidas del escenario", [50, 100, 200, 400], value=100)
    curve_runs = curve_col.select_slider("Partidas por punto de la curva", [20, 40, 60, 80], value=40)
    mc_signature = (tuple(sorted(config_as_dict(config).items())), int(runs), int(curve_runs))
    if st.button("EJECUTAR LABORATORIO MONTE CARLO", type="primary", width="stretch"):
        with st.spinner("Simulando partidas y estimando incertidumbre..."):
            batch = run_monte_carlo(config_as_dict(config), int(runs))
            lambda_grid = tuple(np.round(np.linspace(
                max(0.08, config.lambda_rate * 0.45),
                min(1.8, config.lambda_rate * 1.65),
                9,
            ), 3))
            curve = run_curve(config_as_dict(config), lambda_grid, int(curve_runs))
        st.session_state["mc_result"] = {
            "signature": mc_signature,
            "batch": batch,
            "curve": curve,
        }
    saved = st.session_state.get("mc_result")
    if saved is None:
        st.info("Ejecuta el laboratorio para estimar la probabilidad y su intervalo de confianza.")
    elif saved["signature"] != mc_signature:
        st.warning("Los parámetros del experimento cambiaron. Ejecuta nuevamente.")
    else:
        batch, curve = saved["batch"], saved["curve"]
        summary = summarize_batch(batch, config.arrival_model)
        mc_metrics = st.columns(4)
        mc_metrics[0].metric("P estimada", f"{float(summary['survival_probability'])*100:.1f} %")
        mc_metrics[1].metric("IC 95 %", f"{float(summary['ci_low'])*100:.1f} - {float(summary['ci_high'])*100:.1f} %")
        mc_metrics[2].metric("Tiempo medio", f"{float(summary['mean_survival_time']):.1f} s")
        mc_metrics[3].metric("Llegadas medias", f"{float(summary['mean_generated']):.1f}")
        left, right = st.columns([1, 1.6])
        with left:
            st.plotly_chart(monte_carlo_figure(batch), use_container_width=True, config=PLOTLY_CONFIG)
        with right:
            st.plotly_chart(survival_curve_figure(curve), use_container_width=True, config=PLOTLY_CONFIG)
        below = curve[curve["survival_probability"] < 0.5]
        if not below.empty:
            critical_rate = float(below.iloc[0]["lambda"] * 60)
            st.warning(
                f"En la cuadrícula evaluada, la primera tasa con supervivencia menor a 50 % es "
                f"aproximadamente {critical_rate:.1f} llegadas/min. Es una aproximación Monte Carlo, no un valor exacto."
            )
        with st.expander("Ver datos de las partidas"):
            st.dataframe(batch, width="stretch", hide_index=True)


with guide_tab:
    # Resumen autocontenido para que la demostración también pueda explicar
    # objetivos, variables, supuestos, decisiones tecnológicas y entregables.
    section_header("Lectura académica", "Objetivo, variables, supuestos y alcance")
    st.markdown(
        """
        <div class="callout blue">
          <b>En resumen</b><br>
          Simulamos la aparición de infectados con un proceso de Poisson (o una
          alternativa Polar-lognormal) y medimos si un sobreviviente aguanta el
          ataque durante un horizonte de tiempo objetivo. Comparamos ambos modelos
          con pruebas estadísticas formales y estimamos la probabilidad de
          sobrevivir mediante repetición Monte Carlo.
        </div>
        """,
        unsafe_allow_html=True,
    )
    objective_col, variables_col = st.columns(2)
    with objective_col:
        st.markdown("### Objetivo general")
        st.write(
            "Modelar el subsistema real de aparición y balance de enemigos de un videojuego "
            "de supervivencia, y estimar la probabilidad de completar una misión bajo reglas "
            "de combate controladas."
        )
        st.markdown("### Pregunta de investigación")
        st.write(
            "¿Cómo cambian la congestión y la supervivencia al variar la tasa y la forma "
            "de los tiempos entre llegadas?"
        )
    with variables_col:
        st.markdown("### Variables principales")
        st.markdown(
            """
            - **Entrada:** tasa, duración, modelo, atributos de combate y semilla.
            - **Estado:** HP del jugador, enemigos activos, posiciones y vida individual.
            - **Aleatorias:** interarribos, ángulo, radio inicial, velocidad, HP y DPS enemigos.
            - **Salida:** supervivencia, tiempo resistido, llegadas, bajas y concurrencia máxima.
            """
        )
    st.markdown("### Fuentes de aleatoriedad y justificación")
    st.markdown(
        """
        | Fuente | Distribución | Justificación |
        |---|---|---|
        | Interarribo principal | Exponencial con tasa `lambda` | Tasa constante, independencia y falta de memoria |
        | Interarribo de contraste | Lognormal con media `1/lambda` y CV configurable | Tiempo positivo y regularidad controlable |
        | Normal auxiliar | Marsaglia Polar | Método de generación visto en el curso |
        | Ángulo de aparición | Uniforme en `[0, 2*pi)` | Simetría alrededor de la arena |
        | Radio de aparición | Uniforme entre 94 % y 103 % del radio base | Variación espacial acotada |
        | Velocidad | Uniforme entre 88 % y 112 % del valor base | Heterogeneidad sin extremos irreales |
        | HP y DPS | Uniforme entre 90 % y 112 % del valor base | Diferencias individuales controladas |
        """
    )
    st.caption(
        "Los flujos de llegadas y atributos están separados para que cambiar el método temporal "
        "no modifique accidentalmente las características del enemigo de igual índice."
    )
    st.markdown("### Supuestos")
    st.markdown(
        """
        1. La tasa permanece constante dentro de cada partida.
        2. El protagonista es estacionario y enfoca un objetivo a la vez.
        3. Los infectados siguen trayectorias radiales sin colisiones ni obstáculos.
        4. El daño se integra con paso fijo; reducir `dt` mejora precisión y aumenta costo.
        5. Poisson supone incrementos independientes; Polar-lognormal es un proceso de renovación distinto.
        6. La escena 3D representa el estado del modelo y no reemplaza un motor de videojuegos.
        """
    )
    st.markdown("### Criterio de comparación")
    st.write(
        "La elección no depende únicamente de qué modelo produce más supervivencia. Se evalúan "
        "ajuste distributivo, independencia, costo de generación, interpretabilidad, efecto "
        "pareado sobre las salidas y coherencia de los supuestos con un sistema de apariciones."
    )
    st.markdown("### Decisión tecnológica 3D")
    st.write(
        "Pygame es adecuado para un bucle interactivo 2D y puede actuar como capa de ventana para OpenGL, "
        "pero no se integra naturalmente dentro del ciclo reactivo de Streamlit. Panda3D es la opción de "
        "Python para una aplicación 3D independiente. Esta entrega utiliza mallas Plotly procedurales para "
        "mantener controles, visualización y análisis estadístico en la misma aplicación web."
    )
    st.markdown("### Archivos de entrega")
    st.markdown(
        "- `../README.md`: instalación, arquitectura, metodología y uso.\n"
        "- `report/informe.tex`: informe académico compilable.\n"
        "- `report/generate_report_assets.py`: regeneración trazable de cifras, tablas y resultados.\n"
        "- `report/data/`: datos CSV auditables usados por el informe y la presentación.\n"
        "- `presentation/presentacion.html`: exposición navegable con teclado y controles.\n"
        "- `tests/test_simulation.py`: verificaciones automáticas del motor."
    )


st.markdown("---")
st.caption(
    "OUTBREAK: STOCHASTIC SURVIVAL LAB | Proyecto académico de Modelación y Simulación | "
    "Resultados reproducibles mediante semilla controlada"
)

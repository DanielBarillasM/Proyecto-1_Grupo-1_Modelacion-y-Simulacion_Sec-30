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
    SimulationConfig,
    compare_arrival_models,
    config_as_dict,
    estimate_survival_probability,
    simulate,
    summarize_batch,
    survival_curve,
)
from visuals import (
    arena_3d,
    arrival_process_figure,
    interarrival_figure,
    model_comparison_figure,
    monte_carlo_figure,
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
:root {
    --bg:#070908; --panel:#0d1210; --panel2:#121915; --line:#29362f;
    --text:#e8eee9; --muted:#92a29a; --lime:#c7ff73; --green:#9fe870;
    --red:#ff5964; --amber:#f7a64a; --blue:#68b8d8;
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
.eyebrow {color:var(--lime);font-size:.72rem;letter-spacing:.19em;text-transform:uppercase;font-weight:800}
.hero-copy {padding:18px 0 8px}
.hero-copy h1 {font-size:clamp(2.5rem,5vw,5.4rem);line-height:.88;letter-spacing:-.06em;margin:.45rem 0 1rem}
.hero-copy p {color:var(--muted);font-size:1.02rem;line-height:1.7;max-width:760px}
.hero-copy .rule {width:72px;height:4px;background:var(--lime);border-radius:10px;margin:1.3rem 0}
.stImage img {border-radius:22px;border:1px solid #334139;box-shadow:0 22px 70px rgba(0,0,0,.36)}
.section-title {margin:1.2rem 0 .7rem;padding-bottom:.55rem;border-bottom:1px solid var(--line)}
.section-title small {display:block;color:var(--lime);letter-spacing:.16em;text-transform:uppercase;font-weight:800;font-size:.68rem}
.section-title strong {font-size:1.28rem;letter-spacing:-.02em}
.status-strip {border:1px solid var(--line);border-left:4px solid var(--lime);border-radius:14px;padding:13px 16px;background:#0e1511;margin:.4rem 0 1rem}
.status-strip.failed {border-left-color:var(--red)}
.callout {border:1px solid var(--line);border-radius:14px;padding:15px 17px;background:#0d1410;color:#c8d3cc;margin:.7rem 0 1rem}
.callout.blue {border-left:3px solid var(--blue)}
.callout.amber {border-left:3px solid var(--amber)}
.model-card {height:100%;border:1px solid var(--line);border-radius:18px;padding:18px;background:linear-gradient(180deg,#111813,#0c110e)}
.model-card h3 {margin:.2rem 0 .6rem}.model-card p {color:var(--muted);line-height:1.55}
.tag {display:inline-block;border:1px solid #3a4b40;border-radius:999px;padding:.24rem .55rem;margin:.3rem .25rem .2rem 0;color:#bac8c0;font-size:.72rem}
[data-testid="stMetric"] {background:linear-gradient(180deg,#111813,#0c120e);border:1px solid var(--line);padding:13px 15px;border-radius:15px}
[data-testid="stMetricLabel"] {color:#9caca3}[data-testid="stMetricValue"] {color:#f0f5f1;letter-spacing:-.04em}
.stButton>button,.stFormSubmitButton>button {border-radius:12px;border:1px solid #587345;background:linear-gradient(180deg,#21301e,#151f16);color:#e9f7df;font-weight:800;min-height:46px;letter-spacing:.025em}
.stButton>button:hover,.stFormSubmitButton>button:hover {border-color:var(--lime);color:white}
.stTabs [data-baseweb="tab-list"] {gap:.42rem;border-bottom:1px solid var(--line);padding-bottom:.55rem}
.stTabs [data-baseweb="tab"] {background:#0e1510;border:1px solid var(--line);border-radius:10px;padding:.62rem .95rem}
.stTabs [aria-selected="true"] {border-color:#738f5d!important;color:var(--lime)!important}
code {color:#d8ffae!important} hr {border-color:var(--line)!important}
@media(max-width:800px){.hero-copy h1{font-size:3rem}.block-container{padding-top:.8rem}}
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
def run_comparison(config_dict: dict[str, object], runs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cachea la comparación pareada Poisson frente a Polar-lognormal."""

    return compare_arrival_models(SimulationConfig(**config_dict), runs=runs)


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


# ---------------------------------------------------------------------------
# Panel de configuración
# ---------------------------------------------------------------------------
# El formulario agrupa widgets para evitar que cada movimiento de un deslizador
# dispare una simulación. Solo el botón final confirma y publica la configuración.
with st.sidebar:
    st.markdown('<div class="eyebrow">Centro de control</div>', unsafe_allow_html=True)
    st.markdown("## Configurar misión")
    st.caption("Los cambios solo se aplican al pulsar el botón final del formulario.")

    with st.form("mission_configuration", border=False):
        model_label = st.radio(
            "Modelo de llegadas",
            ["Poisson", "Polar-lognormal"],
            help="Poisson usa interarribos exponenciales. Polar-lognormal usa normales de Marsaglia para construir tiempos positivos.",
        )
        arrivals_per_minute = st.slider(
            "Llegadas esperadas por minuto",
            min_value=6.0,
            max_value=90.0,
            value=39.0,
            step=1.0,
            help="Se convierte internamente a lambda por segundo.",
        )
        duration = st.slider("Duración de la misión (s)", 30, 180, 90, 10)
        polar_cv = st.slider(
            "Variabilidad Polar (CV)",
            0.15,
            1.20,
            0.45,
            0.05,
            help="Solo afecta al modelo Polar-lognormal. Un CV menor genera llegadas más regulares.",
        )

        with st.expander("Capacidad del sobreviviente", expanded=False):
            player_hp = st.slider("Vida inicial (HP)", 60, 200, 110, 5)
            player_dps = st.slider("Daño por segundo", 15, 90, 42, 1)
            weapon_range = st.slider("Alcance del arma (m)", 5.0, 15.0, 10.0, 0.5)

        with st.expander("Características de los infectados", expanded=False):
            enemy_hp = st.slider("Vida base (HP)", 20, 100, 42, 2)
            enemy_speed = st.slider("Velocidad base (m/s)", 0.8, 3.0, 1.55, 0.05)
            enemy_dps = st.slider("Daño de contacto por segundo", 3.0, 20.0, 9.0, 0.5)

        with st.expander("Reproducibilidad y precisión", expanded=False):
            seed = st.number_input("Semilla", 1, 999999, 22193, 1)
            dt = st.select_slider(
                "Paso temporal dt (s)", options=[0.10, 0.05, 0.025], value=0.05
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

    st.markdown("---")
    st.caption(
        "Consejo: mantén la misma semilla para reproducir una partida o cámbiala para observar otra realización."
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
          Revisa los parámetros del centro de control y pulsa <b>SIMULAR MISIÓN</b>
          para generar la primera realización.
        </div>
        """,
        unsafe_allow_html=True,
    )
    section_header("Antes de comenzar", "Qué observar durante el experimento")
    a, b, c = st.columns(3)
    a.markdown("**Llegadas**\n\nCada enemigo aparece en un instante aleatorio acumulado.")
    b.markdown("**Combate**\n\nEl sistema actualiza movimiento, ataque y daño cada `dt` segundos.")
    c.markdown("**Inferencia**\n\nMuchas partidas permiten estimar la probabilidad de sobrevivir.")
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
metrics[2].metric("Llegadas ocurridas", result.generated, f"E[N]={result.expected_arrivals:.1f}")
metrics[3].metric("Eliminados", result.eliminated, f"{100*result.eliminated/max(result.generated,1):.0f} %")
metrics[4].metric("Activos al cierre", result.remaining)
metrics[5].metric("Pico simultáneo", result.max_concurrent)

# Las pestañas siguen el orden recomendado de exposición: observar una corrida,
# explicar su origen matemático, comparar modelos y finalmente inferir por lotes.
simulation_tab, math_tab, comparison_tab, monte_carlo_tab, guide_tab = st.tabs([
    "Simulación 3D",
    "Desarrollo matemático",
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
        st.plotly_chart(arena_3d(result, t_view), use_container_width=True, config={"displaylogo": False})
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
    st.plotly_chart(timeline_figure(result), use_container_width=True, config={"displaylogo": False})


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
    theory[2].metric("E[N(T)]", f"{config.lambda_rate*config.duration:.1f}")
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
        st.plotly_chart(arrival_process_figure(result), use_container_width=True, config={"displaylogo": False})
    with chart_right:
        st.plotly_chart(interarrival_figure(result), use_container_width=True, config={"displaylogo": False})
    st.caption(
        "El histograma usa interarribos observados antes del cierre y sirve como diagnóstico visual; "
        "la truncación temporal puede sesgar una muestra pequeña y no sustituye una prueba formal."
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
            trials, summary = run_comparison(config_as_dict(config), int(comparison_runs))
        st.session_state["comparison_result"] = {
            "signature": (tuple(sorted(config_as_dict(config).items())), int(comparison_runs)),
            "trials": trials,
            "summary": summary,
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
        left, right = st.columns([1.4, 1])
        with left:
            st.plotly_chart(model_comparison_figure(summary), use_container_width=True, config={"displaylogo": False})
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
            st.plotly_chart(monte_carlo_figure(batch), use_container_width=True, config={"displaylogo": False})
        with right:
            st.plotly_chart(survival_curve_figure(curve), use_container_width=True, config={"displaylogo": False})
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
    objective_col, variables_col = st.columns(2)
    with objective_col:
        st.markdown("### Objetivo general")
        st.write(
            "Modelar la aparición aleatoria de infectados y estimar la probabilidad de que "
            "un protagonista sobreviva un horizonte temporal bajo reglas de combate controladas."
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
        "- `presentation/presentacion.html`: exposición navegable con teclado y controles.\n"
        "- `tests/test_simulation.py`: verificaciones automáticas del motor."
    )


st.markdown("---")
st.caption(
    "OUTBREAK: STOCHASTIC SURVIVAL LAB | Proyecto académico de Modelación y Simulación | "
    "Resultados reproducibles mediante semilla controlada"
)

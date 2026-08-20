from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from simulation import (
    SimulationConfig,
    estimate_survival_probability,
    simulate,
    survival_curve,
)
from visuals import (
    arena_3d,
    arrival_process_figure,
    interarrival_figure,
    monte_carlo_figure,
    survival_curve_figure,
    timeline_figure,
)

st.set_page_config(
    page_title="Outbreak: Poisson Survival",
    page_icon="☣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --bg:#070a08; --panel:#0d1410; --panel2:#121b15; --line:#26342b;
    --text:#e3ece6; --muted:#90a69a; --lime:#b7ff5a; --green:#82f58a;
    --red:#ff4d5f; --amber:#ffad4d;
}
.stApp {
    background:
      radial-gradient(circle at 15% 0%, rgba(99,160,83,.14), transparent 34%),
      radial-gradient(circle at 92% 8%, rgba(125,35,39,.12), transparent 30%),
      linear-gradient(180deg, #060806 0%, #090d0a 100%);
    color: var(--text);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b100d, #090c0a);
    border-right: 1px solid var(--line);
}
.block-container { padding-top: 1.5rem; padding-bottom: 4rem; max-width: 1500px; }
.hero {
    position: relative; overflow: hidden; border: 1px solid #2b3a31; border-radius: 24px;
    padding: 28px 32px; margin-bottom: 18px;
    background: linear-gradient(120deg, rgba(13,20,16,.96), rgba(8,12,9,.92));
    box-shadow: 0 18px 60px rgba(0,0,0,.28);
}
.hero:after {
    content:""; position:absolute; inset:-30% -10% auto auto; width:360px; height:360px;
    border-radius:50%; background:radial-gradient(circle, rgba(183,255,90,.14), transparent 68%);
}
.kicker { color:var(--lime); font-size:.78rem; letter-spacing:.18em; text-transform:uppercase; font-weight:800; }
.hero h1 { margin:.2rem 0 .25rem; font-size:clamp(2rem,4vw,3.7rem); line-height:1; letter-spacing:-.045em; }
.hero p { color:var(--muted); font-size:1.03rem; max-width:900px; margin:.7rem 0 0; }
.badge {
    display:inline-block; margin:.7rem .45rem 0 0; padding:.28rem .62rem; border:1px solid #33473a;
    border-radius:999px; background:#101812; color:#bdd0c2; font-size:.76rem;
}
.section-label { color:var(--lime); font-size:.74rem; letter-spacing:.16em; text-transform:uppercase; font-weight:800; margin-bottom:.2rem; }
.panel {
    background:linear-gradient(180deg, rgba(16,24,18,.92), rgba(11,17,13,.92));
    border:1px solid var(--line); border-radius:18px; padding:18px 20px; margin:.35rem 0 1rem;
}
.callout {
    border-left:3px solid var(--lime); background:#0d1510; border-radius:10px;
    padding:13px 16px; color:#c8d7cd; margin:.6rem 0 1rem;
}
.danger { border-left-color:var(--red); }
.small { color:var(--muted); font-size:.88rem; }
hr { border-color:var(--line) !important; }
[data-testid="stMetric"] {
    background:linear-gradient(180deg,#101812,#0c120e); border:1px solid var(--line);
    padding:14px 16px; border-radius:16px;
}
[data-testid="stMetricLabel"] { color:#9eb0a5; }
[data-testid="stMetricValue"] { color:#f0f7f2; letter-spacing:-.04em; }
.stButton > button {
    border-radius:12px; border:1px solid #47603f; background:linear-gradient(180deg,#1a2a1d,#121d15);
    color:#dcf7d6; font-weight:750; min-height:44px;
}
.stButton > button:hover { border-color:var(--lime); color:white; }
.stTabs [data-baseweb="tab-list"] { gap:.45rem; }
.stTabs [data-baseweb="tab"] { background:#0f1711; border:1px solid var(--line); border-radius:11px; padding:.6rem .95rem; }
.stTabs [aria-selected="true"] { border-color:#6b8e5d !important; color:var(--lime) !important; }
code { color:#d6ffad !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def run_main(config_dict: dict) -> object:
    return simulate(SimulationConfig(**config_dict), keep_timeline=True)


@st.cache_data(show_spinner=False)
def run_monte_carlo(config_dict: dict, runs: int) -> pd.DataFrame:
    return estimate_survival_probability(SimulationConfig(**config_dict), runs=runs)


@st.cache_data(show_spinner=False)
def run_curve(config_dict: dict, lambdas: tuple[float, ...], runs_per_lambda: int) -> pd.DataFrame:
    return survival_curve(SimulationConfig(**config_dict), list(lambdas), runs_per_lambda=runs_per_lambda)


st.markdown(
    """
    <div class="hero">
      <div class="kicker">Modelación y Simulación · Variables Aleatorias Continuas</div>
      <h1>OUTBREAK: POISSON SURVIVAL</h1>
      <p>
        Simulador de oleadas de infectados en un videojuego de supervivencia. La llegada de enemigos
        se modela como un <b>Proceso de Poisson homogéneo</b>; sus tiempos entre llegadas se generan
        con la <b>transformada inversa de la distribución Exponencial</b>.
      </p>
      <span class="badge">Python</span><span class="badge">Streamlit</span>
      <span class="badge">Poisson Process</span><span class="badge">3D Tactical View</span>
      <span class="badge">Monte Carlo</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ☣️ Centro de mando")
    st.caption("Configura la presión de la oleada y la capacidad del sobreviviente.")

    presets = {
        "Exploración": 0.25,
        "Supervivencia": 0.70,
        "Pesadilla": 1.15,
        "Extinción": 1.70,
    }
    if "lambda_rate" not in st.session_state:
        st.session_state.lambda_rate = 0.70

    def apply_difficulty_preset() -> None:
        selected = st.session_state.difficulty_preset
        if selected in presets:
            st.session_state.lambda_rate = presets[selected]

    difficulty = st.selectbox(
        "Preset de dificultad",
        ["Personalizado", "Exploración", "Supervivencia", "Pesadilla", "Extinción"],
        index=2,
        key="difficulty_preset",
        on_change=apply_difficulty_preset,
    )

    lam = st.slider(
        "Tasa de llegada λ (enemigos/s)", 0.05, 2.50, step=0.05, key="lambda_rate"
    )
    duration = st.slider("Duración de la misión (s)", 30, 180, 90, 10)

    st.markdown("---")
    st.markdown("### 🧍 Sobreviviente")
    player_hp = st.slider("Vida inicial", 60, 200, 110, 10)
    player_dps = st.slider("Daño por segundo (DPS)", 15, 90, 42, 1)
    weapon_range = st.slider("Alcance del arma (m)", 5.0, 16.0, 10.0, 0.5)

    st.markdown("### 🧟 Infectados")
    enemy_hp = st.slider("Vida base del enemigo", 20, 100, 42, 2)
    enemy_speed = st.slider("Velocidad base (m/s)", 0.8, 3.0, 1.55, 0.05)
    enemy_dps = st.slider("Daño base por segundo", 3.0, 20.0, 9.0, 0.5)

    st.markdown("---")
    seed = st.number_input("Semilla reproducible", min_value=1, max_value=999999, value=22193, step=1)
    st.caption("Una misma semilla + parámetros reproduce la misma partida.")

config = SimulationConfig(
    duration=float(duration),
    lambda_rate=float(lam),
    player_hp=float(player_hp),
    player_dps=float(player_dps),
    weapon_range=float(weapon_range),
    enemy_hp=float(enemy_hp),
    enemy_speed=float(enemy_speed),
    enemy_dps=float(enemy_dps),
    seed=int(seed),
)

result = run_main(config.__dict__)

# Header metrics.
st.markdown('<div class="section-label">Telemetría de la misión</div>', unsafe_allow_html=True)
cols = st.columns(6)
status = "SOBREVIVE" if result.survived else "CAÍDO"
cols[0].metric("Estado", status)
cols[1].metric("Tiempo", f"{result.survival_time:.1f} s", f"de {config.duration:.0f} s")
cols[2].metric("HP final", f"{result.final_hp:.1f}", f"{result.final_hp-config.player_hp:+.1f}")
cols[3].metric("Generados", f"{result.generated}", f"E[N]={result.expected_arrivals:.1f}")
cols[4].metric("Eliminados", f"{result.eliminated}", f"{(100*result.eliminated/max(result.generated,1)):.0f}%")
cols[5].metric("Pico activos", f"{result.max_concurrent}")

main_tab, math_tab, stats_tab, lab_tab = st.tabs(
    ["🎮 Simulación 3D", "🧮 Modelo matemático", "📊 Análisis estadístico", "🧪 Monte Carlo"]
)

with main_tab:
    left, right = st.columns([1.9, 1], gap="large")
    with left:
        t_view = st.slider(
            "Explorar estado de la arena en el tiempo",
            min_value=0.0,
            max_value=float(max(result.survival_time, 0.1)),
            value=float(max(result.survival_time, 0.1)),
            step=0.5,
        )
        st.plotly_chart(arena_3d(result, t_view), width="stretch", config={"displaylogo": False})
        st.caption("Arrastra para rotar · rueda para zoom · la circunferencia verde representa el alcance del arma.")

    with right:
        if result.survived:
            st.success("MISIÓN COMPLETADA · El sobreviviente resistió hasta el final.")
        else:
            st.error("GAME OVER · La presión de la oleada superó la capacidad defensiva.")

        st.markdown(
            f"""
            <div class="panel">
              <div class="section-label">Lectura rápida</div>
              <b>λ = {config.lambda_rate:.2f}</b> enemigos/s significa que, bajo el modelo,
              se esperan en promedio <b>{config.lambda_rate*60:.1f} enemigos por minuto</b>.
              El tiempo medio teórico entre llegadas es
              <b>1/λ = {1/config.lambda_rate:.2f} s</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        now = result.timeline.iloc[(result.timeline["time"] - t_view).abs().argmin()]
        c1, c2 = st.columns(2)
        c1.metric("HP en t", f"{now['player_hp']:.1f}")
        c2.metric("Enemigos activos", f"{int(now['active_enemies'])}")
        c1.metric("Generados hasta t", f"{int(now['spawned'])}")
        c2.metric("Eliminados hasta t", f"{int(now['eliminated'])}")

        st.markdown("#### Reglas del simulador")
        st.markdown(
            """
            - El jugador permanece en el centro y ataca al enemigo vivo más cercano dentro de su alcance.
            - Los infectados nacen en el perímetro y avanzan hacia el jugador.
            - Al entrar en distancia de contacto, cada infectado vivo inflige daño de forma concurrente.
            - La misión termina cuando el HP llega a 0 o cuando se alcanza el tiempo objetivo.
            """
        )

    st.plotly_chart(timeline_figure(result), width="stretch", config={"displaylogo": False})

with math_tab:
    st.markdown("## Fundamento matemático")
    st.markdown(
        """
        <div class="callout">
        La aleatoriedad principal del juego no consiste en decidir manualmente cada cuántos segundos aparece
        un enemigo. Se modela la secuencia de apariciones como un <b>Proceso de Poisson homogéneo</b>
        con tasa constante λ.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 1. Proceso de Poisson")
    st.markdown("Sea $N(t)$ el número de enemigos que han aparecido hasta el instante $t$.")
    st.latex(r"N(t)\sim \operatorname{Poisson}(\lambda t)")
    st.latex(r"P\{N(t)=k\}=e^{-\lambda t}\frac{(\lambda t)^k}{k!},\qquad k=0,1,2,\ldots")
    st.latex(r"\mathbb{E}[N(t)]=\lambda t,\qquad \operatorname{Var}[N(t)]=\lambda t")

    st.markdown("### 2. Variable aleatoria continua: tiempo entre llegadas")
    st.markdown(
        "En un Proceso de Poisson homogéneo, los tiempos entre eventos son independientes y siguen una distribución Exponencial:"
    )
    st.latex(r"\Delta_i\sim \operatorname{Exp}(\lambda)")
    st.latex(r"f_{\Delta}(x)=\lambda e^{-\lambda x},\qquad x\ge 0")
    st.latex(r"F_{\Delta}(x)=1-e^{-\lambda x}")

    st.markdown("### 3. Método de la inversa")
    st.markdown(r"Tomamos $U_i\sim U(0,1)$ y resolvemos la inversa de la función acumulada.")
    st.latex(r"U=1-e^{-\lambda x}")
    st.latex(r"e^{-\lambda x}=1-U")
    st.latex(r"x=-\frac{\ln(1-U)}{\lambda}")
    st.markdown("Como $1-U$ también es uniforme en $(0,1)$, se utiliza la forma equivalente:")
    st.latex(r"\boxed{\Delta_i=-\frac{\ln(U_i)}{\lambda}}")

    st.markdown("### 4. Construcción de los instantes de aparición")
    st.latex(r"t_0=0")
    st.latex(r"t_i=t_{i-1}+\Delta_i")
    st.markdown(
        "Así, cada $t_i$ es el instante de aparición del enemigo $i$. Este es exactamente el mecanismo implementado por el simulador."
    )

    theory1, theory2, theory3 = st.columns(3)
    theory1.metric("λ actual", f"{config.lambda_rate:.2f} enemigos/s")
    theory2.metric("E[Δ] = 1/λ", f"{1/config.lambda_rate:.3f} s")
    theory3.metric("E[N(T)] = λT", f"{config.lambda_rate*config.duration:.1f}")

    st.markdown("### Primeras generaciones aleatorias de esta partida")
    if result.arrivals.empty:
        st.info("No hubo llegadas dentro del horizonte temporal seleccionado.")
    else:
        explain = result.arrivals.head(20).copy()
        explain.columns = ["Enemigo", "Uᵢ", "Δᵢ (s)", "tᵢ (s)"]
        st.dataframe(
            explain.style.format({"Uᵢ": "{:.6f}", "Δᵢ (s)": "{:.4f}", "tᵢ (s)": "{:.4f}"}),
            width="stretch",
            hide_index=True,
        )

    st.markdown("### Supuestos del modelo")
    st.markdown(
        r"""
        1. **Tasa constante:** λ se mantiene fija durante una simulación individual.
        2. **Incrementos independientes:** las llegadas en intervalos disjuntos son independientes.
        3. **Eventos individuales:** la probabilidad de dos o más llegadas en un intervalo infinitesimal es despreciable.
        4. **Interarribos exponenciales:** los $\Delta_i$ son i.i.d. con distribución Exponencial$(\lambda)$.
        """
    )

with stats_tab:
    st.markdown("## Diagnóstico estadístico del proceso")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(arrival_process_figure(result), width="stretch", config={"displaylogo": False})
    with c2:
        st.plotly_chart(interarrival_figure(result), width="stretch", config={"displaylogo": False})

    observed_mean = result.mean_interarrival
    theoretical_mean = 1 / config.lambda_rate
    rel_error = abs(observed_mean - theoretical_mean) / theoretical_mean * 100 if not math.isnan(observed_mean) else float("nan")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Δ medio observado", "—" if math.isnan(observed_mean) else f"{observed_mean:.3f} s")
    m2.metric("Δ medio teórico", f"{theoretical_mean:.3f} s")
    m3.metric("Llegadas observadas", f"{len(result.arrivals)}")
    m4.metric("Error relativo Δ", "—" if math.isnan(rel_error) else f"{rel_error:.1f}%")

    st.markdown(
        """
        <div class="callout">
        En una sola realización es normal observar diferencias respecto al valor esperado. Al repetir muchas
        partidas, los promedios empíricos deben estabilizarse alrededor de los valores teóricos del modelo.
        </div>
        """,
        unsafe_allow_html=True,
    )

with lab_tab:
    st.markdown("## Laboratorio Monte Carlo")
    st.write(
        "Repite automáticamente el mismo escenario con semillas distintas para estimar la probabilidad de supervivencia."
    )

    c1, c2, c3 = st.columns([1, 1, 1.4])
    runs = c1.select_slider("Número de partidas", options=[50, 100, 200, 400], value=100)
    curve_runs = c2.select_slider("Partidas por punto de λ", options=[20, 40, 60, 80], value=40)
    c3.caption("Más repeticiones reducen el ruido Monte Carlo, pero requieren más tiempo de cómputo.")

    experiment_signature = (tuple(sorted(config.__dict__.items())), int(runs), int(curve_runs))
    if st.button("▶ Ejecutar experimento Monte Carlo", type="primary", width="stretch"):
        with st.spinner("Ejecutando partidas y construyendo la curva de dificultad..."):
            batch = run_monte_carlo(config.__dict__, int(runs))
            lambda_grid = tuple(
                np.round(
                    np.linspace(
                        max(0.10, config.lambda_rate * 0.35),
                        min(2.5, config.lambda_rate * 1.9 + 0.2),
                        9,
                    ),
                    2,
                )
            )
            curve = run_curve(config.__dict__, lambda_grid, int(curve_runs))
        st.session_state["mc_result"] = {
            "signature": experiment_signature,
            "batch": batch,
            "curve": curve,
        }

    saved = st.session_state.get("mc_result")
    if saved is None:
        st.info("Configura el experimento y pulsa **Ejecutar experimento Monte Carlo**. Esto evita recalcular cientos de partidas en cada cambio de la interfaz.")
    elif saved["signature"] != experiment_signature:
        st.warning("Los parámetros cambiaron desde el último experimento. Vuelve a ejecutar Monte Carlo para actualizar los resultados.")
    else:
        batch = saved["batch"]
        curve = saved["curve"]
        prob = float(batch["survived"].mean())

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("P̂(supervivencia)", f"{prob*100:.1f}%", border=True)
        mc2.metric("Tiempo medio", f"{batch['survival_time'].mean():.1f} s", border=True)
        mc3.metric("Enemigos medios", f"{batch['generated'].mean():.1f}", border=True)
        mc4.metric("Pico medio", f"{batch['max_concurrent'].mean():.1f}", border=True)

        l, r = st.columns([1, 1.6])
        with l:
            st.plotly_chart(monte_carlo_figure(batch), width="stretch", config={"displaylogo": False})
        with r:
            st.plotly_chart(survival_curve_figure(curve), width="stretch", config={"displaylogo": False})

        below = curve[curve["survival_probability"] < 0.5]
        if not below.empty:
            critical_lambda = float(below.iloc[0]["lambda"])
            st.warning(
                f"En este barrido, la supervivencia cae por debajo de 50% aproximadamente a partir de λ ≈ {critical_lambda:.2f} enemigos/s."
            )
        else:
            st.info("En el rango evaluado, la supervivencia se mantuvo por encima de 50%.")

        with st.expander("Ver datos de las repeticiones"):
            st.dataframe(batch, width="stretch", hide_index=True)

st.markdown("---")
st.caption(
    "OUTBREAK: POISSON SURVIVAL · Proyecto académico de Modelación y Simulación · "
    "La vista 3D es una representación táctica del estado matemático del simulador, no un motor de videojuegos en tiempo real."
)

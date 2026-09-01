"""Regenera evidencia, datos y figuras estáticas del informe.

El informe no contiene porcentajes inventados ni capturas manuales. Este script
ejecuta el mismo motor importado por Streamlit, guarda tablas CSV auditables,
crea figuras PNG con una identidad visual común y escribe macros LaTeX con los
resultados de referencia.

Ejecución desde zombie_poisson_streamlit:

    python report/generate_report_assets.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


REPORT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = REPORT_DIR.parent
APP_DIR = PROJECT_DIR / "App"
ASSETS_DIR = PROJECT_DIR / "assets"
DATA_DIR = REPORT_DIR / "data"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from simulation import (
    SimulationConfig,
    compare_arrival_models,
    paired_comparison_statistics,
    simulate,
    survival_curve,
    validate_arrival_generators,
)


# Colores alineados con Streamlit y la presentación.
BACKGROUND = "#070908"
PANEL = "#101612"
GRID = "#334139"
TEXT = "#e8eee9"
MUTED = "#9aa8a0"
POISSON = "#b8f36b"
POLAR = "#68b8d8"
ORANGE = "#f3a64f"
RED = "#ff606b"


def configure_matplotlib() -> None:
    """Aplica un tema legible tanto en pantalla como en el PDF."""

    plt.rcParams.update({
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": PANEL,
        "savefig.facecolor": BACKGROUND,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "grid.color": GRID,
        "grid.alpha": 0.55,
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
    })


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Guarda con resolución suficiente y elimina márgenes innecesarios."""

    fig.savefig(ASSETS_DIR / filename, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_generator_validation(validation, config: SimulationConfig) -> None:
    """Crea seis diagnósticos sobre muestras no censuradas.

    Las dos primeras filas cubren las transformaciones y el conteo. La tercera
    audita la fuente uniforme y contrapone forma marginal contra estructura
    secuencial, que es la distinción sobre la que descansa toda la batería.
    """

    exponential = validation.samples["exponential"].to_numpy()
    lognormal = validation.samples["lognormal"].to_numpy()
    normals = validation.samples["normal_z"].to_numpy()
    counts = validation.poisson_counts["count"].to_numpy()
    reference_uniforms = validation.uniform_samples["pcg64"].to_numpy()
    own_uniforms = validation.uniform_samples["lcg_minstd"].to_numpy()
    degenerate_uniforms = validation.uniform_samples["lcg_degenerate"].to_numpy()
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 12.4))
    fig.suptitle("Validación estadística de los generadores", fontsize=18, y=0.995)

    ax = axes[0, 0]
    ax.hist(exponential, bins=42, density=True, color=POISSON, alpha=0.72)
    x = np.linspace(0, np.quantile(exponential, 0.995), 350)
    ax.plot(x, config.lambda_rate * np.exp(-config.lambda_rate * x), color=ORANGE, lw=2.6)
    ax.set(title="Interarribos exponenciales", xlabel="Delta (s)", ylabel="Densidad")

    sigma2 = np.log(1.0 + config.polar_cv**2)
    sigma = np.sqrt(sigma2)
    mu = np.log(1.0 / config.lambda_rate) - sigma2 / 2.0
    ax = axes[0, 1]
    ax.hist(lognormal, bins=42, density=True, color=POLAR, alpha=0.72)
    x = np.linspace(0.001, np.quantile(lognormal, 0.995), 350)
    density = np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma2))
    density /= x * sigma * np.sqrt(2 * np.pi)
    ax.plot(x, density, color=ORANGE, lw=2.6)
    ax.set(title="Interarribos Polar-lognormal", xlabel="Delta (s)", ylabel="Densidad")

    ax = axes[1, 0]
    probabilities = (np.arange(len(normals)) + 0.5) / len(normals)
    theoretical = stats.norm.ppf(probabilities)
    observed = np.sort(normals)
    bounds = [min(theoretical.min(), observed.min()), max(theoretical.max(), observed.max())]
    ax.scatter(theoretical, observed, s=8, color=POLAR, alpha=0.5)
    ax.plot(bounds, bounds, color=ORANGE, lw=2, ls="--")
    ax.set(
        title="Q-Q de normales generadas por Marsaglia",
        xlabel="Cuantil normal teórico",
        ylabel="Cuantil observado",
    )

    ax = axes[1, 1]
    values, frequencies = np.unique(counts, return_counts=True)
    ax.bar(values, frequencies / len(counts), color=POISSON, alpha=0.72)
    ax.plot(
        values,
        stats.poisson.pmf(values, counts.mean()),
        color=ORANGE,
        marker="o",
        ms=3,
        lw=2,
    )
    ax.set(title="Conteo Poisson en calendarios repetidos", xlabel="N(T)", ylabel="Frecuencia")

    ax = axes[2, 0]
    ax.hist(reference_uniforms, bins=20, range=(0, 1), density=True,
            color=POISSON, alpha=0.6, label="PCG64")
    ax.hist(own_uniforms, bins=20, range=(0, 1), density=True,
            color=POLAR, alpha=0.55, label="LCG propio")
    ax.axhline(1.0, color=ORANGE, lw=2.4)
    ax.legend(loc="lower right", fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    ax.set(title="Fuente uniforme: forma marginal", xlabel="u", ylabel="Densidad")

    # El diagrama de rezago uno separa lo que el histograma confunde: el control
    # degenerado es plano en una dimensión y una recta en dos.
    ax = axes[2, 1]
    ax.scatter(own_uniforms[:-1], own_uniforms[1:], s=4, color=POLAR,
               alpha=0.35, label="LCG propio")
    ax.scatter(degenerate_uniforms[:-1], degenerate_uniforms[1:], s=4, color=RED,
               alpha=0.75, label="Control degenerado")
    ax.legend(loc="lower right", fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    ax.set(title="Fuente uniforme: diagrama de rezago 1", xlabel="u(n)", ylabel="u(n+1)")

    for axis in axes.flat:
        axis.grid(True, axis="y")
    fig.tight_layout()
    save_figure(fig, "report-generator-validation.png")


def plot_model_comparison(trials, summary) -> None:
    """Visualiza probabilidad e impacto sobre concurrencia."""

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    labels = ["Poisson", "Polar-lognormal"]
    probabilities = summary["survival_probability"].to_numpy() * 100
    # El redondeo de punto flotante puede producir -1e-14 cuando p=1; Matplotlib
    # exige magnitudes de error no negativas.
    lower = np.maximum(0.0, probabilities - summary["ci_low"].to_numpy() * 100)
    upper = np.maximum(0.0, summary["ci_high"].to_numpy() * 100 - probabilities)
    axes[0].bar(labels, probabilities, color=[POISSON, POLAR], width=0.62)
    axes[0].errorbar(
        labels, probabilities, yerr=np.vstack((lower, upper)),
        fmt="none", ecolor=TEXT, capsize=6, lw=2,
    )
    axes[0].set(title="Supervivencia con IC de Wilson", ylabel="Probabilidad (%)", ylim=(0, 105))
    for index, value in enumerate(probabilities):
        axes[0].text(index, value - 8, f"{value:.1f} %", ha="center", fontweight="bold", color=BACKGROUND)

    poisson_peak = trials.loc[trials["model"] == "poisson", "max_concurrent"]
    polar_peak = trials.loc[trials["model"] == "polar", "max_concurrent"]
    box = axes[1].boxplot(
        [poisson_peak, polar_peak],
        tick_labels=labels,
        patch_artist=True,
        medianprops={"color": ORANGE, "linewidth": 2.2},
        whiskerprops={"color": TEXT},
        capprops={"color": TEXT},
    )
    for patch, color in zip(box["boxes"], (POISSON, POLAR)):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    axes[1].set(title="Concurrencia máxima por partida", ylabel="Infectados activos")
    for axis in axes:
        axis.grid(True, axis="y")
    fig.suptitle("Comparación pareada del escenario base", fontsize=18, y=1.03)
    fig.tight_layout()
    save_figure(fig, "report-model-comparison.png")


def plot_survival_curves(config: SimulationConfig) -> None:
    """Evalúa ambos modelos sobre la misma cuadrícula de tasas."""

    rates_per_minute = np.linspace(24, 72, 9)
    curves = []
    for model in ("poisson", "polar"):
        model_config = replace(config, arrival_model=model)
        curve = survival_curve(
            model_config,
            [float(rate / 60.0) for rate in rates_per_minute],
            runs_per_lambda=60,
        )
        curves.append(curve)
    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    for curve, label, color in zip(curves, ("Poisson", "Polar-lognormal"), (POISSON, POLAR)):
        x = curve["lambda"].to_numpy() * 60
        y = curve["survival_probability"].to_numpy() * 100
        low = curve["ci_low"].to_numpy() * 100
        high = curve["ci_high"].to_numpy() * 100
        ax.plot(x, y, color=color, marker="o", lw=2.5, label=label)
        ax.fill_between(x, low, high, color=color, alpha=0.15)
    ax.axhline(50, color=ORANGE, ls="--", lw=1.6, label="Umbral 50 %")
    ax.set(
        title="Sensibilidad de supervivencia frente a la tasa",
        xlabel="Llegadas esperadas por minuto",
        ylabel="Supervivencia (%)",
        ylim=(-2, 103),
    )
    ax.grid(True)
    ax.legend(frameon=False, labelcolor=TEXT)
    fig.tight_layout()
    save_figure(fig, "report-survival-curves.png")
    curves[0].assign(model="poisson").to_csv(DATA_DIR / "curve_poisson.csv", index=False)
    curves[1].assign(model="polar").to_csv(DATA_DIR / "curve_polar.csv", index=False)


def plot_timeline(config: SimulationConfig) -> None:
    """Incluye una trayectoria individual para conectar eventos y estado."""

    result = simulate(config)
    timeline = result.timeline
    fig, first = plt.subplots(figsize=(11.8, 5.2))
    second = first.twinx()
    first.plot(timeline["time"], timeline["player_hp"], color=POISSON, lw=2.6, label="HP")
    second.plot(
        timeline["time"], timeline["active_enemies"],
        color=RED, lw=2.1, alpha=0.9, label="Activos",
    )
    first.set(title="Trayectoria de una misión reproducible", xlabel="Tiempo (s)", ylabel="HP")
    second.set_ylabel("Infectados activos", color=RED)
    first.grid(True)
    lines = first.lines + second.lines
    first.legend(lines, [line.get_label() for line in lines], frameon=False, labelcolor=TEXT)
    fig.tight_layout()
    save_figure(fig, "report-timeline.png")


def latex_number(value: float, decimals: int = 4) -> str:
    """Formatea números para macros LaTeX sin notación regional."""

    return f"{value:.{decimals}f}"


def latex_pvalue(value: float) -> str:
    """Conserva valores p muy pequeños mediante notación científica LaTeX."""

    # Un cero exacto es subdesbordamiento, no una probabilidad nula: se declara
    # como cota. Un valor pequeño pero representable conserva su magnitud.
    if value <= 0.0:
        return "<10^{-16}"
    if value < 0.0001:
        exponent = int(np.floor(np.log10(value)))
        mantissa = value / (10**exponent)
        return f"{mantissa:.3f}\\times 10^{{{exponent}}}"
    return f"{value:.6f}"


def write_generated_results(validation, summary, effects, contingency) -> None:
    """Escribe macros que mantienen tablas y narrativa sincronizadas."""

    poisson = summary.loc[summary["model"] == "poisson"].iloc[0]
    polar = summary.loc[summary["model"] == "polar"].iloc[0]
    survival = effects.loc[effects["metric"] == "survived"].iloc[0]
    peak = effects.loc[effects["metric"] == "max_concurrent"].iloc[0]


    def contrast(generator: str, name: str):
        """Localiza una fila por generador y contraste, sin depender del orden."""

        rows = validation.tests
        selected = rows.loc[
            (rows["generator"] == generator) & (rows["test"] == name)
        ]
        if selected.empty:
            raise KeyError(f"No existe el contraste {generator} / {name}")
        return selected.iloc[0]

    exp_ks = contrast("Poisson-exponencial", "KS contra Exponencial")
    normal_ks = contrast("Marsaglia Polar", "KS de Z contra Normal(0,1)")
    acceptance = contrast("Marsaglia Polar", "Aceptación contra pi/4")
    lognormal_mean = contrast("Polar-lognormal", "Media contra 1/lambda")
    count_test = contrast("Conteo Poisson", "Índice de dispersión")
    count_chi = contrast("Conteo Poisson", "Ji-cuadrada contra pmf Poisson")
    uniform_ks = contrast("Fuente uniforme PCG64", "KS contra U(0,1)")
    uniform_serial = contrast("Fuente uniforme PCG64", "Serial de tercias en el cubo")
    lcg_ks = contrast("LCG propio (MINSTD)", "KS contra U(0,1)")
    lcg_serial = contrast("LCG propio (MINSTD)", "Serial de tercias en el cubo")
    control = validation.uniform_control.set_index("test")["p_value"]
    poisson_perf = validation.performance.loc[
        validation.performance["generator"] == "Poisson-exponencial"
    ].iloc[0]
    polar_perf = validation.performance.loc[
        validation.performance["generator"] == "Polar-lognormal"
    ].iloc[0]
    lines = [
        "% Archivo generado automáticamente; no editar a mano.",
        f"\\newcommand{{\\ResultsDate}}{{{date.today().isoformat()}}}",
        f"\\newcommand{{\\PoissonSurvival}}{{{poisson['survival_probability']*100:.1f}}}",
        f"\\newcommand{{\\PolarSurvival}}{{{polar['survival_probability']*100:.1f}}}",
        f"\\newcommand{{\\PoissonCILow}}{{{poisson['ci_low']*100:.1f}}}",
        f"\\newcommand{{\\PoissonCIHigh}}{{{poisson['ci_high']*100:.1f}}}",
        f"\\newcommand{{\\PolarCILow}}{{{polar['ci_low']*100:.1f}}}",
        f"\\newcommand{{\\PolarCIHigh}}{{{polar['ci_high']*100:.1f}}}",
        f"\\newcommand{{\\PoissonMeanTime}}{{{poisson['mean_survival_time']:.3f}}}",
        f"\\newcommand{{\\PolarMeanTime}}{{{polar['mean_survival_time']:.3f}}}",
        f"\\newcommand{{\\PoissonPeak}}{{{poisson['mean_max_concurrent']:.3f}}}",
        f"\\newcommand{{\\PolarPeak}}{{{polar['mean_max_concurrent']:.3f}}}",
        f"\\newcommand{{\\SurvivalDifference}}{{{survival['difference']*100:.1f}}}",
        f"\\newcommand{{\\SurvivalDiffLow}}{{{survival['ci_low']*100:.1f}}}",
        f"\\newcommand{{\\SurvivalDiffHigh}}{{{survival['ci_high']*100:.1f}}}",
        f"\\newcommand{{\\McNemarP}}{{{latex_pvalue(float(survival['p_value']))}}}",
        f"\\newcommand{{\\PeakDifference}}{{{peak['difference']:.3f}}}",
        f"\\newcommand{{\\PeakP}}{{{latex_pvalue(float(peak['p_value']))}}}",
        f"\\newcommand{{\\TestCount}}{{{len(validation.tests)}}}",
        f"\\newcommand{{\\SmallestNominalP}}{{{latex_number(float(validation.tests['p_value'].min()))}}}",
        f"\\newcommand{{\\ExponentialKSP}}{{{latex_number(float(exp_ks['p_value']))}}}",
        f"\\newcommand{{\\NormalKSP}}{{{latex_number(float(normal_ks['p_value']))}}}",
        f"\\newcommand{{\\AcceptanceRate}}{{{float(acceptance['statistic']):.4f}}}",
        f"\\newcommand{{\\AcceptanceP}}{{{latex_number(float(acceptance['p_value']))}}}",
        f"\\newcommand{{\\LognormalMeanP}}{{{latex_number(float(lognormal_mean['p_value']))}}}",
        f"\\newcommand{{\\CountDispersionP}}{{{latex_number(float(count_test['p_value']))}}}",
        f"\\newcommand{{\\CountChiP}}{{{latex_number(float(count_chi['p_value']))}}}",
        f"\\newcommand{{\\UniformKSP}}{{{latex_number(float(uniform_ks['p_value']))}}}",
        f"\\newcommand{{\\UniformSerialP}}{{{latex_number(float(uniform_serial['p_value']))}}}",
        f"\\newcommand{{\\LcgKSP}}{{{latex_number(float(lcg_ks['p_value']))}}}",
        f"\\newcommand{{\\LcgSerialP}}{{{latex_number(float(lcg_serial['p_value']))}}}",
        f"\\newcommand{{\\ControlUniformityP}}{{{latex_number(float(control['Ji-cuadrada de uniformidad']))}}}",
        f"\\newcommand{{\\ControlKSP}}{{{latex_number(float(control['KS contra U(0,1)']))}}}",
        f"\\newcommand{{\\ControlRunsP}}{{{latex_pvalue(float(control['Rachas arriba/abajo']))}}}",
        f"\\newcommand{{\\ControlLjungP}}{{{latex_pvalue(float(control['Ljung-Box rezagos 1-5']))}}}",
        f"\\newcommand{{\\ControlSerialP}}{{{latex_pvalue(float(control['Serial de tercias en el cubo']))}}}",
        f"\\newcommand{{\\PoissonMicros}}{{{float(poisson_perf['microseconds_per_value']):.3f}}}",
        f"\\newcommand{{\\PolarMicros}}{{{float(polar_perf['microseconds_per_value']):.3f}}}",
        f"\\newcommand{{\\BothSurvive}}{{{int(contingency.iloc[0]['count'])}}}",
        f"\\newcommand{{\\OnlyPoisson}}{{{int(contingency.iloc[1]['count'])}}}",
        f"\\newcommand{{\\OnlyPolar}}{{{int(contingency.iloc[2]['count'])}}}",
        f"\\newcommand{{\\BothFail}}{{{int(contingency.iloc[3]['count'])}}}",
    ]
    (REPORT_DIR / "generated_results.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Ejecuta el protocolo de referencia completo."""

    configure_matplotlib()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    config = SimulationConfig()
    validation = validate_arrival_generators(config)
    trials, summary = compare_arrival_models(config, runs=400)
    effects, contingency = paired_comparison_statistics(trials)

    validation.tests.to_csv(DATA_DIR / "generator_tests.csv", index=False)
    validation.moments.to_csv(DATA_DIR / "generator_moments.csv", index=False)
    validation.performance.to_csv(DATA_DIR / "generator_performance.csv", index=False)
    validation.uniform_control.to_csv(DATA_DIR / "uniform_control.csv", index=False)
    trials.to_csv(DATA_DIR / "paired_trials.csv", index=False)
    summary.to_csv(DATA_DIR / "comparison_summary.csv", index=False)
    effects.to_csv(DATA_DIR / "paired_effects.csv", index=False)
    contingency.to_csv(DATA_DIR / "survival_contingency.csv", index=False)

    plot_generator_validation(validation, config)
    plot_model_comparison(trials, summary)
    plot_survival_curves(config)
    plot_timeline(config)
    write_generated_results(validation, summary, effects, contingency)
    print("Activos del informe regenerados correctamente.")
    print(summary.to_string(index=False))
    print(effects.to_string(index=False))


if __name__ == "__main__":
    main()

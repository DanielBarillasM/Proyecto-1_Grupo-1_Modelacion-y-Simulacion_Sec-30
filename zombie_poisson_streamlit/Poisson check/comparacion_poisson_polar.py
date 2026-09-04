"""
Comparación Poisson vs Polar-lognormal: ¿la práctica confirma la teoría?
==========================================================================

Script autocontenido (solo requiere numpy, scipy y matplotlib) que:

1. Genera interarribos con dos modelos que comparten la misma media 1/lambda:
     - Poisson homogéneo:      Delta = -ln(U) / lambda            (transformada inversa)
     - Polar-lognormal:        Delta = exp(mu + sigma * Z)         (Z por Marsaglia Polar)
2. Contrasta momentos teóricos contra observados (media, varianza, CV).
3. Aplica pruebas formales de bondad de ajuste e independencia:
     - KS de los interarribos Poisson contra Exponencial(lambda)
     - KS de las normales de Marsaglia contra N(0,1)
     - Prueba t de la media lognormal contra 1/lambda
     - Correlación de rezago 1 (independencia) en ambos interarribos
     - Índice de dispersión y chi-cuadrada de N(T) contra Poisson(lambda*T)
   y corrige por comparaciones múltiples con Holm-Bonferroni.
4. Genera figuras comparativas (histogramas + densidad teórica, Q-Q de
   normales, trayectorias N(t) y ajuste del conteo).
5. Imprime un reporte de conclusiones y lo guarda en texto plano.

Ejecución:
    python comparacion_poisson_polar.py

Todo el experimento es reproducible: cambia SEED para obtener otra
realización, o los demás parámetros en la sección de configuración.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------
# 1. Configuración del experimento (ajusta aquí)
# ---------------------------------------------------------------------
LAMBDA_RATE = 0.93          # tasa media de llegadas, en eventos/segundo
DURATION = 90.0             # horizonte T de una "partida" / observación, en segundos
POLAR_CV = 0.60             # coeficiente de variación deseado del modelo Polar-lognormal
SAMPLE_SIZE = 5_000         # tamaño de muestra fija para comparar interarribos (sin censura)
COUNT_REPETITIONS = 2_000   # nro. de calendarios repetidos para validar el conteo N(T)
SEED = 22193
ALPHA = 0.05
OUTPUT_DIR = Path(__file__).resolve().parent / "salidas_comparacion"
OUTPUT_DIR.mkdir(exist_ok=True)

REPORT_LINES: list[str] = []


def log(line: str = "") -> None:
    """Imprime y además acumula la línea para el reporte de texto final."""
    print(line)
    REPORT_LINES.append(line)


# ---------------------------------------------------------------------
# 2. Generadores
# ---------------------------------------------------------------------
def interarribos_exponenciales(n: int, lam: float, rng: np.random.Generator) -> np.ndarray:
    """Poisson homogéneo: Delta_i = -ln(U_i) / lambda, con U_i ~ U(0,1)."""
    u = rng.uniform(np.finfo(float).eps, 1.0, size=n)
    return -np.log(u) / lam


def parametros_lognormal(lam: float, cv: float) -> tuple[float, float]:
    """mu y sigma de la normal subyacente para que E[Delta] = 1/lambda con ese CV."""
    sigma2 = math.log(1.0 + cv**2)
    return math.log(1.0 / lam) - sigma2 / 2.0, math.sqrt(sigma2)


class GeneradorMarsaglia:
    """Normales estándar por el método polar de Marsaglia, con búfer vectorizado.

    Se proponen pares (V1, V2) ~ U(-1,1)^2 y se aceptan solo si 0 < S = V1^2+V2^2 < 1.
    Cada par aceptado produce dos normales independientes. La tasa de aceptación
    teórica es pi/4 (área del círculo unitario sobre el cuadrado [-1,1]^2).
    """

    def __init__(self, rng: np.random.Generator, tam_lote: int = 4_096) -> None:
        self.rng = rng
        self.tam_lote = tam_lote
        self._buffer: list[float] = []
        self.propuestos = 0
        self.aceptados = 0

    def _rellenar(self) -> None:
        pares = self.rng.uniform(-1.0, 1.0, size=(self.tam_lote, 2))
        s = np.sum(pares**2, axis=1)
        mascara = (s > 0.0) & (s < 1.0)
        self.propuestos += self.tam_lote
        self.aceptados += int(mascara.sum())
        v1, v2, s_ok = pares[mascara, 0], pares[mascara, 1], s[mascara]
        factor = np.sqrt(-2.0 * np.log(s_ok) / s_ok)
        nuevos = np.empty(v1.size * 2)
        nuevos[0::2] = v1 * factor
        nuevos[1::2] = v2 * factor
        self._buffer.extend(nuevos.tolist())

    def siguiente(self) -> float:
        if not self._buffer:
            self._rellenar()
        return self._buffer.pop()

    def muestra(self, n: int) -> np.ndarray:
        return np.array([self.siguiente() for _ in range(n)])

    @property
    def tasa_aceptacion(self) -> float:
        return self.aceptados / self.propuestos if self.propuestos else float("nan")


def interarribos_polar(
    n: int, lam: float, cv: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, float]:
    """Polar-lognormal: Delta_i = exp(mu + sigma * Z_i), Z_i por Marsaglia Polar."""
    generador = GeneradorMarsaglia(rng)
    z = generador.muestra(n)
    mu, sigma = parametros_lognormal(lam, cv)
    delta = np.exp(mu + sigma * z)
    return delta, z, generador.tasa_aceptacion


def contar_en_horizonte(
    lam: float, duration: float, rng: np.random.Generator, modelo: str, cv: float | None = None
) -> int:
    """Genera un calendario completo hasta 'duration' y devuelve N(T)."""
    t, n = 0.0, 0
    generador = GeneradorMarsaglia(rng) if modelo == "polar" else None
    mu_sigma = parametros_lognormal(lam, cv) if modelo == "polar" else None
    while True:
        if modelo == "poisson":
            u = rng.uniform(np.finfo(float).eps, 1.0)
            delta = -math.log(u) / lam
        else:
            mu, sigma = mu_sigma
            delta = math.exp(mu + sigma * generador.siguiente())
        t += delta
        if t > duration:
            break
        n += 1
    return n


def trayectoria(lam: float, duration: float, rng: np.random.Generator, modelo: str, cv: float | None = None):
    """Devuelve los instantes de llegada t_1..t_N hasta el horizonte (para graficar N(t))."""
    tiempos, t = [], 0.0
    generador = GeneradorMarsaglia(rng) if modelo == "polar" else None
    mu_sigma = parametros_lognormal(lam, cv) if modelo == "polar" else None
    while True:
        if modelo == "poisson":
            u = rng.uniform(np.finfo(float).eps, 1.0)
            delta = -math.log(u) / lam
        else:
            mu, sigma = mu_sigma
            delta = math.exp(mu + sigma * generador.siguiente())
        t += delta
        if t > duration:
            break
        tiempos.append(t)
    return np.array(tiempos)


def holm(p_values: np.ndarray) -> np.ndarray:
    """Corrección de Holm-Bonferroni: controla el error por familia sin suponer independencia."""
    m = len(p_values)
    orden = np.argsort(p_values)
    ordenados = np.asarray(p_values, dtype=float)[orden]
    ajustados = np.maximum.accumulate((m - np.arange(m)) * ordenados)
    resultado = np.empty(m)
    resultado[orden] = np.minimum(ajustados, 1.0)
    return resultado


# ---------------------------------------------------------------------
# 3. Experimento principal
# ---------------------------------------------------------------------
def main() -> None:
    semillas = np.random.SeedSequence(SEED).spawn(6)
    rng_exp, rng_polar, rng_count_poisson, rng_count_polar, rng_traj_p, rng_traj_q = (
        np.random.default_rng(s) for s in semillas
    )

    log("=" * 78)
    log("COMPARACIÓN POISSON vs POLAR-LOGNORMAL — TEORÍA vs PRÁCTICA")
    log("=" * 78)
    log(f"lambda = {LAMBDA_RATE} eventos/s   |   1/lambda (interarribo medio) = {1/LAMBDA_RATE:.4f} s")
    log(f"Duración T = {DURATION} s   |   CV Polar = {POLAR_CV}   |   semilla = {SEED}")
    log(f"Muestra fija para interarribos = {SAMPLE_SIZE}   |   calendarios repetidos = {COUNT_REPETITIONS}")
    log()

    # --- 3.1 Muestra fija de interarribos (sin censura por horizonte) -------
    exp_delta = interarribos_exponenciales(SAMPLE_SIZE, LAMBDA_RATE, rng_exp)
    polar_delta, z_normales, tasa_aceptacion = interarribos_polar(
        SAMPLE_SIZE, LAMBDA_RATE, POLAR_CV, rng_polar
    )
    mu, sigma = parametros_lognormal(LAMBDA_RATE, POLAR_CV)

    log("-" * 78)
    log("1) MOMENTOS: teórico vs observado (muestra de tamaño fijo)")
    log("-" * 78)
    media_teo = 1.0 / LAMBDA_RATE
    var_teo_exp = 1.0 / LAMBDA_RATE**2
    var_teo_log = (POLAR_CV * media_teo) ** 2
    log(f"{'':22}{'media teo.':>12}{'media obs.':>12}{'var teo.':>12}{'var obs.':>12}{'CV teo.':>9}{'CV obs.':>9}")
    log(
        f"{'Poisson-exponencial':22}{media_teo:12.4f}{exp_delta.mean():12.4f}"
        f"{var_teo_exp:12.4f}{exp_delta.var(ddof=1):12.4f}"
        f"{1.0:9.3f}{exp_delta.std(ddof=1)/exp_delta.mean():9.3f}"
    )
    log(
        f"{'Polar-lognormal':22}{media_teo:12.4f}{polar_delta.mean():12.4f}"
        f"{var_teo_log:12.4f}{polar_delta.var(ddof=1):12.4f}"
        f"{POLAR_CV:9.3f}{polar_delta.std(ddof=1)/polar_delta.mean():9.3f}"
    )
    log(
        f"Aceptación Marsaglia Polar: {tasa_aceptacion:.4f}  "
        f"(teórica pi/4 = {math.pi/4:.4f})"
    )
    log()

    # --- 3.2 Pruebas formales -------------------------------------------
    ks_exp = stats.kstest(exp_delta, "expon", args=(0.0, 1.0 / LAMBDA_RATE))
    ks_norm = stats.kstest(z_normales, "norm")
    t_media_log = stats.ttest_1samp(polar_delta, 1.0 / LAMBDA_RATE)
    corr_exp = stats.pearsonr(exp_delta[:-1], exp_delta[1:])
    corr_polar = stats.pearsonr(polar_delta[:-1], polar_delta[1:])

    # --- 3.3 Validación del conteo N(T) sobre calendarios repetidos ------
    semillas_poisson = rng_count_poisson.integers(1, 2**31 - 1, size=COUNT_REPETITIONS)
    semillas_polar = rng_count_polar.integers(1, 2**31 - 1, size=COUNT_REPETITIONS)
    conteos_poisson = np.array([
        contar_en_horizonte(LAMBDA_RATE, DURATION, np.random.default_rng(int(s)), "poisson")
        for s in semillas_poisson
    ])
    conteos_polar = np.array([
        contar_en_horizonte(LAMBDA_RATE, DURATION, np.random.default_rng(int(s)), "polar", POLAR_CV)
        for s in semillas_polar
    ])

    esperado_lambda_t = LAMBDA_RATE * DURATION
    media_conteo_poisson = conteos_poisson.mean()
    var_conteo_poisson = conteos_poisson.var(ddof=1)
    # Índice de dispersión: bajo Poisson, (n-1)*S^2/media ~ chi2_{n-1}
    disp_stat = (COUNT_REPETITIONS - 1) * var_conteo_poisson / media_conteo_poisson
    disp_cdf = stats.chi2.cdf(disp_stat, COUNT_REPETITIONS - 1)
    disp_p = min(1.0, 2.0 * min(disp_cdf, 1.0 - disp_cdf))

    # Chi-cuadrada de bondad de ajuste de N(T) contra Poisson(lambda*T), con
    # fusión de celdas por la derecha hasta que la frecuencia esperada >= 5.
    maximo = int(conteos_poisson.max())
    soporte = np.arange(0, maximo + 1)
    probs = np.append(stats.poisson.pmf(soporte, esperado_lambda_t), stats.poisson.sf(maximo, esperado_lambda_t))
    frecs = np.append(np.bincount(conteos_poisson.astype(int), minlength=maximo + 1), 0)
    prob_acum = frec_acum = 0.0
    probs_fusion, frecs_fusion = [], []
    for p, f in zip(probs, frecs):
        prob_acum += p
        frec_acum += f
        if prob_acum * COUNT_REPETITIONS >= 5.0:
            probs_fusion.append(prob_acum)
            frecs_fusion.append(frec_acum)
            prob_acum = frec_acum = 0.0
    if probs_fusion and prob_acum > 0.0:
        probs_fusion[-1] += prob_acum
        frecs_fusion[-1] += frec_acum
    esperadas = np.array(probs_fusion) * COUNT_REPETITIONS
    esperadas *= sum(frecs_fusion) / esperadas.sum()
    chi_conteo = stats.chisquare(np.array(frecs_fusion), esperadas)

    log("-" * 78)
    log("2) PRUEBAS FORMALES (alpha = 0.05, corregidas por Holm-Bonferroni)")
    log("-" * 78)
    nombres = [
        "KS interarribos Poisson vs Exponencial(lambda)",
        "KS normales Marsaglia vs N(0,1)",
        "t: media lognormal vs 1/lambda",
        "Correlación rezago-1 (independencia) Poisson",
        "Correlación rezago-1 (independencia) Polar",
        "Indice de dispersion N(T) Poisson",
        "Chi-cuadrada N(T) vs Poisson(lambda*T)",
    ]
    estadisticos = [
        ks_exp.statistic, ks_norm.statistic, t_media_log.statistic,
        corr_exp.statistic, corr_polar.statistic, disp_stat, chi_conteo.statistic,
    ]
    p_valores = np.array([
        ks_exp.pvalue, ks_norm.pvalue, t_media_log.pvalue,
        corr_exp.pvalue, corr_polar.pvalue, disp_p, chi_conteo.pvalue,
    ])
    p_holm = holm(p_valores)

    log(f"{'Prueba':48}{'Estadístico':>12}{'p':>10}{'p (Holm)':>10}  ¿No se rechaza?")
    for nombre, est, p, ph in zip(nombres, estadisticos, p_valores, p_holm):
        veredicto = "sí" if ph >= ALPHA else "NO"
        log(f"{nombre:48}{est:12.5f}{p:10.5f}{ph:10.5f}  {veredicto}")
    log()

    log("-" * 78)
    log("3) EL CONTEO N(T): por qué Poisson y Polar-lognormal no son lo mismo")
    log("-" * 78)
    log(f"lambda*T (referencia) = {esperado_lambda_t:.2f}")
    log(f"Poisson:  media(N(T)) = {media_conteo_poisson:.3f}   var(N(T)) = {var_conteo_poisson:.3f}   "
        f"(teoría exige media = var = lambda*T)")
    log(f"Polar:    media(N(T)) = {conteos_polar.mean():.3f}   var(N(T)) = {conteos_polar.var(ddof=1):.3f}   "
        f"(es un proceso de renovación: no tiene por qué cumplir media=var=lambda*T)")
    diferencia_media = conteos_polar.mean() - esperado_lambda_t
    log(f"Diferencia Polar - lambda*T = {diferencia_media:+.3f}  "
        f"(igualar E[Delta]=1/lambda no garantiza igualar E[N(T)] en un horizonte finito)")
    log()

    log("=" * 78)
    log("CONCLUSIÓN")
    log("=" * 78)
    if bool((p_holm >= ALPHA).all()):
        log("Con la semilla usada, NINGUNA prueba corregida por Holm rechaza el modelo:")
        log("la práctica (lo simulado) es consistente con la teoría declarada para ambos")
        log("procesos: interarribos exponenciales para Poisson, normales de Marsaglia")
        log("válidas y N(T) compatible con Poisson(lambda*T). No rechazar no es una")
        log("prueba absoluta; solo indica que la muestra no contradice el modelo.")
    else:
        rechazadas = [n for n, ph in zip(nombres, p_holm) if ph < ALPHA]
        log("Al menos una prueba fue rechazada tras la corrección de Holm:")
        for n in rechazadas:
            log(f"  - {n}")
        log("Antes de concluir que el generador falla, prueba con otra semilla o una")
        log("muestra mayor: con 7 contrastes simultáneos, un rechazo aislado al 5%")
        log("puede deberse al azar (por eso se usa Holm en vez del p nominal).")
    log()
    log("Diferencia estructural (no depende de la semilla): Poisson tiene CV=1 fijo")
    log("y carece de memoria; Polar-lognormal permite fijar el CV libremente, por lo")
    log("que puede ser más regular (CV<1, menos rachas) o más disperso (CV>1) que")
    log("Poisson mantiendo la misma tasa media de llegadas.")

    # ------------------------------------------------------------------
    # 4. Gráficas
    # ------------------------------------------------------------------
    plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3})

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.hist(exp_delta, bins=50, density=True, alpha=0.6, color="#4c9a5b", label="Observado (Poisson)")
    xs = np.linspace(0, np.quantile(exp_delta, 0.995), 300)
    ax.plot(xs, LAMBDA_RATE * np.exp(-LAMBDA_RATE * xs), color="#d9622b", lw=2.5, label="Exponencial teórica")
    ax.set_title("Interarribos: Poisson-exponencial")
    ax.set_xlabel("Delta (s)"); ax.set_ylabel("Densidad"); ax.legend()

    ax = axes[0, 1]
    ax.hist(polar_delta, bins=50, density=True, alpha=0.6, color="#3d7ea6", label="Observado (Polar)")
    xs = np.linspace(0.001, np.quantile(polar_delta, 0.995), 300)
    pdf = np.exp(-((np.log(xs) - mu) ** 2) / (2 * sigma**2)) / (xs * sigma * math.sqrt(2 * math.pi))
    ax.plot(xs, pdf, color="#d9622b", lw=2.5, label="Lognormal teórica")
    ax.set_title("Interarribos: Polar-lognormal")
    ax.set_xlabel("Delta (s)"); ax.set_ylabel("Densidad"); ax.legend()

    ax = axes[1, 0]
    probabilidades = (np.arange(len(z_normales)) + 0.5) / len(z_normales)
    teorico = stats.norm.ppf(probabilidades)
    observado = np.sort(z_normales)
    limites = [min(teorico.min(), observado.min()), max(teorico.max(), observado.max())]
    ax.scatter(teorico, observado, s=6, alpha=0.4, color="#3d7ea6")
    ax.plot(limites, limites, color="#d9622b", lw=2, ls="--", label="Referencia 45°")
    ax.set_title("Q-Q: normales de Marsaglia vs N(0,1)")
    ax.set_xlabel("Cuantil teórico"); ax.set_ylabel("Cuantil observado"); ax.legend()

    ax = axes[1, 1]
    valores_conteo, frecuencias = np.unique(conteos_poisson, return_counts=True)
    empirica = frecuencias / len(conteos_poisson)
    ax.bar(valores_conteo, empirica, alpha=0.6, color="#4c9a5b", label="N(T) observado")
    ax.plot(valores_conteo, stats.poisson.pmf(valores_conteo, esperado_lambda_t),
            color="#d9622b", lw=2, marker="o", ms=3, label="PMF Poisson(lambda·T)")
    ax.set_title("Conteo N(T) repetido: Poisson")
    ax.set_xlabel("N(T)"); ax.set_ylabel("Frecuencia relativa"); ax.legend()

    fig.suptitle("Auditoría de generadores: Poisson vs Polar-lognormal", fontsize=14, y=1.00)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_auditoria_generadores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Trayectorias N(t) de una realización de cada modelo, y comparación de N(T)
    tiempos_poisson = trayectoria(LAMBDA_RATE, DURATION, rng_traj_p, "poisson")
    tiempos_polar = trayectoria(LAMBDA_RATE, DURATION, rng_traj_q, "polar", POLAR_CV)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for tiempos, color, etiqueta in (
        (tiempos_poisson, "#4c9a5b", "Poisson"), (tiempos_polar, "#3d7ea6", "Polar-lognormal")
    ):
        x = np.r_[0.0, np.repeat(tiempos, 2), DURATION]
        conteo = np.arange(1, len(tiempos) + 1)
        y = np.r_[0, np.column_stack((conteo - 1, conteo)).ravel(), len(tiempos)]
        ax.step(x, y, where="post", color=color, lw=2, label=etiqueta)
    ax.plot([0, DURATION], [0, esperado_lambda_t], color="#888", ls=":", label="lambda·T (referencia)")
    ax.set_title("Una realización: N(t) observado")
    ax.set_xlabel("t (s)"); ax.set_ylabel("N(t)"); ax.legend()

    ax = axes[1]
    ax.boxplot([conteos_poisson, conteos_polar], tick_labels=["Poisson", "Polar-lognormal"])
    ax.axhline(esperado_lambda_t, color="#888", ls=":", label="lambda·T (referencia)")
    ax.set_title(f"Distribución de N(T) en {COUNT_REPETITIONS} calendarios repetidos")
    ax.set_ylabel("N(T)"); ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_conteo_NT_poisson_vs_polar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    (OUTPUT_DIR / "reporte.txt").write_text("\n".join(REPORT_LINES), encoding="utf-8")
    log()
    log(f"Figuras y reporte guardados en: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

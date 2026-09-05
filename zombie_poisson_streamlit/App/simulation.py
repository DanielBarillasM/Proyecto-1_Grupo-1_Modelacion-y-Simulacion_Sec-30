"""Motor estocástico y capa de experimentación del proyecto.

Se comparan dos modelos con el mismo interarribo medio: un proceso de Poisson
homogéneo con interarribos exponenciales y un proceso de renovación con
interarribos lognormales generados mediante el método polar de Marsaglia.
El método polar no es un proceso de conteo; esta formulación evita presentar
como equivalentes dos conceptos matemáticos de distinta naturaleza.

El módulo no depende de Streamlit ni de Plotly. Esa separación permite ejecutar
pruebas y experimentos Monte Carlo sin levantar la interfaz web. El flujo es:

``SimulationConfig -> calendario de llegadas -> combate -> SimulationResult``.

Unidades internas
-----------------
* tiempo: segundos;
* distancia: metros;
* vida: puntos de HP;
* daño: HP por segundo;
* ``lambda_rate``: llegadas por segundo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import exp, log, sqrt
from time import perf_counter
from typing import Iterator, Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats


# A Literal catches invalid model names in static analysis while ``validate``
# protects callers at runtime (for example, dictionaries restored from cache).
ArrivalModel = Literal["poisson", "polar"]


@dataclass(frozen=True)
class SimulationConfig:
    """Configuración completa e inmutable de una partida.

    La inmutabilidad evita que una ejecución Monte Carlo cambie accidentalmente
    los parámetros compartidos. ``dataclasses.replace`` crea las variantes de
    semilla, tasa o modelo necesarias para cada corrida.
    """

    duration: float = 90.0  # Horizonte de simulación, en segundos.
    # lambda_rate, polar_cv y enemy_dps quedaron recalibrados en la auditoría: el
    # escenario original (0.65, 0.45, 9.0) saturaba Polar en 400/400
    # supervivencias, con solo 14 pares discordantes sosteniendo McNemar. Con
    # estos valores, Poisson sobrevive ~35 % y Polar ~78 % sobre 400 corridas
    # pareadas (254 pares discordantes, presentes en ambas direcciones), lejos
    # de los extremos 0 y 1 y con potencia estadística real. Ver
    # ``compare_arrival_models`` / ``paired_comparison_statistics``.
    lambda_rate: float = 0.93  # Intensidad media de llegadas por segundo.
    arrival_model: ArrivalModel = "poisson"  # Familia temporal seleccionada.
    polar_cv: float = 0.6  # Coeficiente de variación del modelo lognormal.
    player_hp: float = 110.0  # Vida inicial del protagonista.
    player_dps: float = 42.0  # Daño continuo infligido por segundo.
    weapon_range: float = 10.0  # Radio dentro del que puede atacar.
    enemy_hp: float = 42.0  # Vida base antes de la perturbación individual.
    enemy_speed: float = 1.55  # Velocidad radial base en metros/segundo.
    enemy_dps: float = 9.5  # Daño base de un atacante en contacto.
    arena_radius: float = 18.0  # Radio nominal de aparición.
    contact_radius: float = 1.35  # Distancia a la que comienza el daño.
    dt: float = 0.05  # Paso de integración del combate.
    seed: int = 22193  # Semilla maestra reproducible.

    def validate(self) -> None:
        """Valida dominio, geometría y estabilidad numérica.

        Raises:
            ValueError: si una magnitud es no positiva, el modelo no existe,
                los radios se solapan de forma imposible o ``dt`` es demasiado
                grande para la resolución admitida.
        """

        positive = {
            "duration": self.duration,
            "lambda_rate": self.lambda_rate,
            "player_hp": self.player_hp,
            "player_dps": self.player_dps,
            "weapon_range": self.weapon_range,
            "enemy_hp": self.enemy_hp,
            "enemy_speed": self.enemy_speed,
            "enemy_dps": self.enemy_dps,
            "arena_radius": self.arena_radius,
            "contact_radius": self.contact_radius,
            "dt": self.dt,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Los parámetros deben ser positivos: {', '.join(invalid)}")
        if self.arrival_model not in {"poisson", "polar"}:
            raise ValueError("arrival_model debe ser 'poisson' o 'polar'")
        if not 0.05 <= self.polar_cv <= 2.0:
            raise ValueError("polar_cv debe estar entre 0.05 y 2.0")
        if self.contact_radius >= self.weapon_range:
            raise ValueError("contact_radius debe ser menor que weapon_range")
        if self.weapon_range >= self.arena_radius:
            raise ValueError("weapon_range debe ser menor que arena_radius")
        if self.dt > 0.25:
            raise ValueError("dt debe ser menor o igual a 0.25 s")


@dataclass
class SimulationResult:
    """Resultados escalares y tablas producidas por una partida.

    ``arrivals`` contiene solo eventos que ocurrieron antes del final real de
    la partida. ``arrival_schedule`` conserva el calendario hasta el horizonte
    solicitado, incluso si el jugador murió antes. Esta separación impide que la
    interfaz contabilice como ocurridos eventos posteriores a la derrota.
    """

    config: SimulationConfig
    survived: bool
    survival_time: float
    final_hp: float
    generated: int
    scheduled: int
    eliminated: int
    remaining: int
    max_concurrent: int
    mean_interarrival: float
    expected_arrivals: float
    expected_schedule: float
    arrivals: pd.DataFrame
    arrival_schedule: pd.DataFrame
    enemies: pd.DataFrame
    timeline: pd.DataFrame


@dataclass
class GeneratorValidationResult:
    """Evidencia reproducible sobre la calidad de los generadores.

    ``tests`` contiene contrastes formales con su valor p nominal y su valor p
    corregido por multiplicidad; ``moments`` compara momentos teóricos y
    empíricos; ``samples`` conserva muestras de tamaño fijo, sin censura por
    horizonte, para histogramas y gráficos Q-Q; ``uniform_samples`` guarda los
    uniformes de cada fuente auditada; ``uniform_control`` reporta aparte el
    generador degenerado que sirve de control negativo; y ``performance``
    registra medianas de tiempo de generación. Separar estas tablas permite
    reutilizar exactamente la misma evidencia en Streamlit, el informe y los
    scripts de figuras.
    """

    tests: pd.DataFrame
    moments: pd.DataFrame
    samples: pd.DataFrame
    poisson_counts: pd.DataFrame
    performance: pd.DataFrame
    uniform_samples: pd.DataFrame
    uniform_control: pd.DataFrame


# Ambos generadores entregan el mismo esquema. Las columnas que no aplican a un
# modelo se rellenan con NaN; así, la interfaz puede alternar modelos sin adaptar
# su contrato de datos.
ARRIVAL_COLUMNS = [
    "enemy_id", "model", "u", "polar_v1", "polar_v2", "z", "delta", "arrival_time"
]


def _marsaglia_normals(rng: np.random.Generator) -> Iterator[tuple[float, float, float, float]]:
    """Produce una secuencia infinita de normales estándar por Marsaglia Polar.

    Args:
        rng: generador NumPy que aporta los uniformes independientes.

    Yields:
        Tuplas ``(z, v1, v2, s)``. Se preservan los uniformes aceptados y
        ``s = v1² + v2²`` para poder explicar y auditar la transformación.

    Notes:
        Cada punto se propone en el cuadrado (-1, 1)². Los puntos fuera del
        círculo unitario se rechazan; cada par aceptado produce dos normales.
    """

    while True:
        v1, v2 = rng.uniform(-1.0, 1.0, size=2)
        s = float(v1 * v1 + v2 * v2)
        if not 0.0 < s < 1.0:
            continue
        factor = sqrt(-2.0 * log(s) / s)
        yield float(v1 * factor), float(v1), float(v2), s
        yield float(v2 * factor), float(v1), float(v2), s


def _lognormal_parameters(lambda_rate: float, coefficient_variation: float) -> tuple[float, float]:
    """Calcula ``mu`` y ``sigma`` de una lognormal comparable con Poisson.

    La parametrización fuerza ``E[Delta] = 1 / lambda_rate`` y conserva el
    coeficiente de variación solicitado. Devuelve parámetros de la normal
    subyacente, no la media y desviación de ``Delta``.
    """

    sigma_squared = log(1.0 + coefficient_variation**2)
    return log(1.0 / lambda_rate) - sigma_squared / 2.0, sqrt(sigma_squared)


def sample_exponential_interarrivals(
    lambda_rate: float,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera una muestra fija de interarribos exponenciales.

    A diferencia del calendario limitado por una duración, esta función no
    elimina el último interarribo que cruza el horizonte. Por eso es la entrada
    correcta para pruebas de bondad de ajuste y comparación de momentos.
    """

    if lambda_rate <= 0 or sample_size <= 0:
        raise ValueError("lambda_rate y sample_size deben ser positivos")
    uniforms = rng.uniform(np.finfo(float).eps, 1.0, size=sample_size)
    return -np.log(uniforms) / lambda_rate


def sample_marsaglia_normals(
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, int]:
    """Obtiene normales estándar y registra la eficiencia de aceptación.

    Returns:
        Normales generadas, proporción de pares aceptados y cantidad de pares
        propuestos. La tasa teórica de aceptación del círculo unitario es pi/4.
    """

    if sample_size <= 0:
        raise ValueError("sample_size debe ser positivo")
    values: list[float] = []
    proposed_pairs = 0
    accepted_pairs = 0
    while len(values) < sample_size:
        proposed_pairs += 1
        v1, v2 = rng.uniform(-1.0, 1.0, size=2)
        s = float(v1 * v1 + v2 * v2)
        if not 0.0 < s < 1.0:
            continue
        accepted_pairs += 1
        factor = sqrt(-2.0 * log(s) / s)
        values.extend((float(v1 * factor), float(v2 * factor)))
    return (
        np.asarray(values[:sample_size], dtype=float),
        accepted_pairs / proposed_pairs,
        proposed_pairs,
    )


def sample_polar_lognormal_interarrivals(
    lambda_rate: float,
    coefficient_variation: float,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Genera una muestra lognormal fija a partir de Marsaglia Polar."""

    if lambda_rate <= 0 or sample_size <= 0:
        raise ValueError("lambda_rate y sample_size deben ser positivos")
    if not 0.05 <= coefficient_variation <= 2.0:
        raise ValueError("coefficient_variation debe estar entre 0.05 y 2.0")
    normals, acceptance_rate, proposed_pairs = sample_marsaglia_normals(sample_size, rng)
    mu, sigma = _lognormal_parameters(lambda_rate, coefficient_variation)
    interarrivals = np.exp(mu + sigma * normals)
    return interarrivals, normals, acceptance_rate, proposed_pairs


def sample_marsaglia_normals_vectorized(
    sample_size: int,
    rng: np.random.Generator,
    batch_scale: float = 1.35,
) -> tuple[np.ndarray, float, int]:
    """Variante vectorizada de Marsaglia Polar, con el mismo contrato de salida.

    ``sample_marsaglia_normals`` propone un par por iteración de un bucle de
    Python; esa elección de estilo, no el método de aceptación-rechazo en sí,
    era la causa de que el benchmark original penalizara a Polar. Aquí se
    propone un lote completo de pares por llamada a NumPy y se filtra con
    álgebra vectorial, igual que ``sample_exponential_interarrivals`` resuelve
    la transformada inversa en una sola pasada. El resultado es la misma
    distribución exacta; solo cambia cuánto trabajo por propuesta delega en C
    en vez de en el intérprete.

    ``batch_scale`` sobredimensiona cada lote sobre lo que exige la tasa de
    aceptación teórica (pi/4 ~ 0.785) para que, en la enorme mayoría de las
    llamadas, una sola generación de lote baste.
    """

    if sample_size <= 0:
        raise ValueError("sample_size debe ser positivo")
    needed_pairs = -(-sample_size // 2)  # techo de sample_size / 2
    values = np.empty(0, dtype=float)
    proposed_pairs = 0
    accepted_pairs = 0
    while accepted_pairs < needed_pairs:
        remaining_pairs = needed_pairs - accepted_pairs
        # pi/4 es la probabilidad teórica de aceptación; el margen evita una
        # segunda vuelta del bucle en el caso típico sin sobre-generar de más.
        batch_size = max(64, int(remaining_pairs / (np.pi / 4.0) * batch_scale))
        pairs = rng.uniform(-1.0, 1.0, size=(batch_size, 2))
        squared_radius = np.sum(pairs**2, axis=1)
        mask = (squared_radius > 0.0) & (squared_radius < 1.0)
        proposed_pairs += batch_size
        v1 = pairs[mask, 0]
        v2 = pairs[mask, 1]
        s = squared_radius[mask]
        factor = np.sqrt(-2.0 * np.log(s) / s)
        batch_values = np.empty(v1.size * 2, dtype=float)
        batch_values[0::2] = v1 * factor
        batch_values[1::2] = v2 * factor
        values = np.concatenate([values, batch_values])
        accepted_pairs += v1.size
    return (
        values[:sample_size],
        accepted_pairs / proposed_pairs,
        proposed_pairs,
    )


def sample_polar_lognormal_interarrivals_vectorized(
    lambda_rate: float,
    coefficient_variation: float,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Igual que ``sample_polar_lognormal_interarrivals`` con Polar vectorizado.

    Existe únicamente para que el benchmark compare implementaciones del mismo
    nivel de esfuerzo de vectorización; el motor de simulación y la validación
    estadística siguen usando la versión original, más legible y suficiente
    para sus tamaños de muestra.
    """

    if lambda_rate <= 0 or sample_size <= 0:
        raise ValueError("lambda_rate y sample_size deben ser positivos")
    if not 0.05 <= coefficient_variation <= 2.0:
        raise ValueError("coefficient_variation debe estar entre 0.05 y 2.0")
    normals, acceptance_rate, proposed_pairs = sample_marsaglia_normals_vectorized(
        sample_size, rng
    )
    mu, sigma = _lognormal_parameters(lambda_rate, coefficient_variation)
    interarrivals = np.exp(mu + sigma * normals)
    return interarrivals, normals, acceptance_rate, proposed_pairs


def sample_box_muller_normals(
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera normales estándar por Box-Muller trigonométrico, vectorizado.

    A diferencia de Marsaglia Polar, Box-Muller no rechaza propuestas: cada par
    de uniformes ``(U1, U2) ~ U(0,1)`` produce dos normales mediante seno y
    coseno. Es el segundo método que el enunciado pide comparar contra Polar
    para la MISMA distribución objetivo (N(0,1)), en lugar de comparar dos
    procesos de llegada distintos como hacía el benchmark original.

    ``U1`` se recorta lejos de cero para evitar ``log(0)`` sin introducir sesgo
    perceptible (la probabilidad de caer en el intervalo excluido es del orden
    de la precisión de máquina).
    """

    if sample_size <= 0:
        raise ValueError("sample_size debe ser positivo")
    pair_count = -(-sample_size // 2)
    u1 = rng.uniform(np.finfo(float).eps, 1.0, size=pair_count)
    u2 = rng.uniform(0.0, 1.0, size=pair_count)
    radius = np.sqrt(-2.0 * np.log(u1))
    angle = 2.0 * np.pi * u2
    values = np.empty(pair_count * 2, dtype=float)
    values[0::2] = radius * np.cos(angle)
    values[1::2] = radius * np.sin(angle)
    return values[:sample_size]


def compare_normal_generators(
    sample_size: int = 20_000,
    benchmark_repetitions: int = 9,
    seed: int = 77_411,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compara Marsaglia Polar contra Box-Muller para la MISMA distribución.

    Esta es la comparación de métodos que pide el enunciado: dos algoritmos
    distintos generando exactamente N(0,1) a partir de uniformes, evaluados en
    bondad de ajuste (KS), sesgo/curtosis, costo de generación y, en el caso de
    Polar, la fracción de propuestas descartadas por el rechazo.

    Returns:
        ``tests``: una fila por método con KS contra N(0,1), momentos y su
        estado de aceptación bajo ``alpha``.
        ``performance``: medianas de tiempo de generación, en microsegundos
        por valor, usando implementaciones vectorizadas equivalentes para
        ambos métodos.
    """

    if sample_size < 1_000 or benchmark_repetitions < 1:
        raise ValueError("Se requieren muestra y repeticiones suficientes")
    seed_sequence = np.random.SeedSequence(seed).spawn(3)
    polar_rng = np.random.default_rng(seed_sequence[0])
    box_rng = np.random.default_rng(seed_sequence[1])
    bench_rng = np.random.default_rng(seed_sequence[2])

    polar_values, acceptance_rate, proposed_pairs = sample_marsaglia_normals_vectorized(
        sample_size, polar_rng
    )
    box_values = sample_box_muller_normals(sample_size, box_rng)

    rows = []
    for name, values in (("Marsaglia Polar", polar_values), ("Box-Muller", box_values)):
        ks = stats.kstest(values, "norm")
        rows.append({
            "method": name,
            "ks_statistic": float(ks.statistic),
            "p_value": float(ks.pvalue),
            "passes": bool(ks.pvalue >= alpha),
            "mean": float(values.mean()),
            "variance": float(values.var(ddof=1)),
            "skewness": float(stats.skew(values)),
            "excess_kurtosis": float(stats.kurtosis(values)),
            "rejected_fraction": (
                1.0 - acceptance_rate if name == "Marsaglia Polar" else 0.0
            ),
        })
    tests = pd.DataFrame(rows)

    benchmark_rows = []
    for name, generator in (
        ("Marsaglia Polar", lambda size, r: sample_marsaglia_normals_vectorized(size, r)[0]),
        ("Box-Muller", sample_box_muller_normals),
    ):
        elapsed = []
        for _ in range(benchmark_repetitions):
            local_rng = np.random.default_rng(int(bench_rng.integers(1, 2**31 - 1)))
            start = perf_counter()
            generator(sample_size, local_rng)
            elapsed.append((perf_counter() - start) * 1_000.0)
        median_ms = float(np.median(elapsed))
        benchmark_rows.append({
            "method": name,
            "sample_size": sample_size,
            "repetitions": benchmark_repetitions,
            "median_ms": median_ms,
            "microseconds_per_value": median_ms * 1_000.0 / sample_size,
        })
    performance = pd.DataFrame(benchmark_rows)
    return tests, performance


def _pearson_lag_test(values: np.ndarray) -> tuple[float, float]:
    """Contrasta correlación lineal de rezago uno como diagnóstico de independencia."""

    result = stats.pearsonr(values[:-1], values[1:])
    return float(result.statistic), float(result.pvalue)


class LinearCongruentialGenerator:
    """Generador congruencial lineal propio: ``x_{n+1} = (a x_n + c) mod m``.

    El proyecto usa PCG64 de NumPy para producir resultados, pero un LCG escrito
    a mano permite auditar la fuente uniforme con el mismo detalle con que se
    auditan las transformaciones. Se emplea en dos papeles opuestos:

    * ``MINSTD_PARAMETERS`` es un multiplicador clásico de calidad aceptable y
      compite de igual a igual con PCG64 en la batería de contrastes;
    * ``COUNTER_PARAMETERS`` produce una rampa perfectamente uniforme pero
      totalmente predecible. Sirve como control negativo: demuestra que la
      batería detecta dependencia y no solo desviaciones de la forma uniforme.

    Notes:
        No sustituye a NumPy dentro del motor de simulación. Su ámbito es el
        laboratorio de validación, donde se compara contra la fuente real.
    """

    def __init__(self, multiplier: int, increment: int, modulus: int, seed: int) -> None:
        if modulus <= 1 or multiplier <= 0 or increment < 0:
            raise ValueError("Parámetros LCG inválidos")
        self.multiplier = int(multiplier)
        self.increment = int(increment)
        self.modulus = int(modulus)
        state = int(seed) % self.modulus
        # Un LCG multiplicativo (c = 0) se queda atrapado en el cero absorbente.
        if state == 0 and self.increment == 0:
            state = 1
        self._state = state

    def next_uniform(self) -> float:
        """Avanza el estado y devuelve un uniforme en ``[0, 1)``."""

        self._state = (self.multiplier * self._state + self.increment) % self.modulus
        return self._state / self.modulus

    def uniforms(self, size: int) -> np.ndarray:
        """Devuelve ``size`` uniformes consecutivos del mismo flujo."""

        if size <= 0:
            raise ValueError("size debe ser positivo")
        return np.asarray([self.next_uniform() for _ in range(size)], dtype=float)


# Park y Miller, ampliamente documentado y de período completo 2^31 - 2.
MINSTD_PARAMETERS = (16_807, 0, 2**31 - 1)
# Contador puro: recorre todo su período en orden. Uniforme, jamás aleatorio.
# El módulo divide los tamaños de muestra que ofrece la interfaz (1000, 2000,
# 4000 y 8000), de modo que el control recorre un número entero de períodos y
# su histograma es exactamente plano. Así, cuando falla, falla por dependencia
# y no por una cobertura incompleta del intervalo.
COUNTER_PARAMETERS = (1, 1, 500)


def _chi_square_uniformity(values: np.ndarray, cells: int = 20) -> tuple[float, float]:
    """Contrasta la forma marginal con celdas equiprobables en ``[0, 1)``.

    Responde a una pregunta que ninguna prueba de independencia contesta:
    ¿aparecen todos los tramos del intervalo con la misma frecuencia?
    """

    observed, _ = np.histogram(values, bins=cells, range=(0.0, 1.0))
    expected = np.full(cells, len(values) / cells, dtype=float)
    result = stats.chisquare(observed, expected)
    return float(result.statistic), float(result.pvalue)


def _runs_up_down(values: np.ndarray) -> tuple[float, float]:
    """Prueba de rachas ascendentes y descendentes sobre el orden observado.

    Cuenta cambios de sentido entre valores consecutivos. Bajo independencia el
    número de rachas tiene media ``(2n-1)/3`` y varianza ``(16n-29)/90``. Detecta
    tendencias y ciclos que una prueba de forma marginal no puede ver.
    """

    differences = np.diff(values)
    signs = np.sign(differences)
    signs = signs[signs != 0]
    if len(signs) < 20:
        raise ValueError("La prueba de rachas necesita más observaciones")
    runs = 1 + int(np.count_nonzero(np.diff(signs) != 0))
    size = len(signs) + 1
    mean = (2.0 * size - 1.0) / 3.0
    variance = (16.0 * size - 29.0) / 90.0
    z = (runs - mean) / sqrt(variance)
    return float(z), float(2.0 * stats.norm.sf(abs(z)))


def _ljung_box(values: np.ndarray, lags: int = 5) -> tuple[float, float]:
    """Contrasta correlación serial conjunta hasta el rezago ``lags``.

    Un solo rezago puede salir limpio mientras la estructura vive en otro. El
    estadístico de Ljung-Box acumula los primeros ``lags`` y se distribuye
    ``chi2`` con ``lags`` grados de libertad bajo independencia.
    """

    size = len(values)
    centred = values - values.mean()
    denominator = float(np.sum(centred**2))
    statistic = 0.0
    for lag in range(1, lags + 1):
        correlation = float(np.sum(centred[lag:] * centred[:-lag]) / denominator)
        statistic += correlation**2 / (size - lag)
    statistic *= size * (size + 2)
    return float(statistic), float(stats.chi2.sf(statistic, lags))


def _serial_three_dimensional(values: np.ndarray, cells_per_axis: int = 4) -> tuple[float, float]:
    """Contrasta uniformidad de tercias no solapadas dentro del cubo unitario.

    Los generadores congruenciales colocan sus puntos sobre familias de planos.
    El defecto es invisible en una dimensión y puede serlo en dos; por eso se
    revisa el cubo. Las tercias no se solapan para conservar la independencia
    que supone la ji-cuadrada.
    """

    usable = (len(values) // 3) * 3
    triples = values[:usable].reshape(-1, 3)
    indices = np.minimum((triples * cells_per_axis).astype(int), cells_per_axis - 1)
    flat = (
        indices[:, 0] * cells_per_axis**2
        + indices[:, 1] * cells_per_axis
        + indices[:, 2]
    )
    total_cells = cells_per_axis**3
    observed = np.bincount(flat, minlength=total_cells)
    expected = np.full(total_cells, len(triples) / total_cells, dtype=float)
    if expected[0] < 5.0:
        raise ValueError("Se requieren más tercias para la prueba serial")
    result = stats.chisquare(observed, expected)
    return float(result.statistic), float(result.pvalue)


def _poisson_count_chi_square(counts: np.ndarray, expected_mean: float) -> tuple[float, float, int]:
    """Contrasta la forma completa de ``N(T)`` contra ``Poisson(lambda*T)``.

    El índice de dispersión solo verifica que media y varianza coincidan; una
    distribución con esa propiedad puede no ser Poisson. Esta prueba compara
    todas las frecuencias. Las celdas se fusionan desde la izquierda hasta que
    cada frecuencia esperada supera cinco, y ``lambda*T`` es conocido, de modo
    que no se pierden grados de libertad por estimación.

    Returns:
        Estadístico, valor p y grados de libertad efectivos.
    """

    size = len(counts)
    highest = int(counts.max())
    support = np.arange(0, highest + 1)
    probabilities = np.append(
        stats.poisson.pmf(support, expected_mean),
        float(stats.poisson.sf(highest, expected_mean)),
    )
    frequencies = np.append(np.bincount(counts.astype(int), minlength=highest + 1), 0)

    merged_probabilities: list[float] = []
    merged_frequencies: list[float] = []
    probability_accumulator = 0.0
    frequency_accumulator = 0.0
    for probability, frequency in zip(probabilities, frequencies):
        probability_accumulator += float(probability)
        frequency_accumulator += float(frequency)
        if probability_accumulator * size >= 5.0:
            merged_probabilities.append(probability_accumulator)
            merged_frequencies.append(frequency_accumulator)
            probability_accumulator = 0.0
            frequency_accumulator = 0.0
    if merged_probabilities and probability_accumulator > 0.0:
        merged_probabilities[-1] += probability_accumulator
        merged_frequencies[-1] += frequency_accumulator
    if len(merged_probabilities) < 3:
        raise ValueError("Se requieren más calendarios para la ji-cuadrada del conteo")

    expected = np.asarray(merged_probabilities) * size
    expected *= sum(merged_frequencies) / expected.sum()
    result = stats.chisquare(np.asarray(merged_frequencies), expected)
    return float(result.statistic), float(result.pvalue), len(merged_probabilities) - 1


def uniform_source_tests(
    values: np.ndarray,
    label: str,
    lags: int = 5,
    cells: int = 20,
) -> list[tuple[str, str, float, float]]:
    """Aplica la batería completa a una fuente uniforme y devuelve sus filas.

    Los cinco contrastes miden propiedades distintas y no se implican entre sí:
    forma marginal por celdas, discrepancia máxima acumulada, orden secuencial,
    correlación conjunta y estructura en tres dimensiones. Un generador puede
    aprobar los dos primeros y fracasar en los tres siguientes; ese es
    exactamente el caso del control negativo incluido en la validación.
    """

    # La prueba serial reparte tercias entre 64 celdas y exige al menos cinco
    # esperadas por celda; por debajo de mil valores la batería no es aplicable.
    if len(values) < 1_000:
        raise ValueError("La auditoría de la fuente uniforme necesita al menos 1000 valores")
    kolmogorov = stats.kstest(values, "uniform")
    return [
        (label, "Ji-cuadrada de uniformidad", *_chi_square_uniformity(values, cells)),
        (label, "KS contra U(0,1)", float(kolmogorov.statistic), float(kolmogorov.pvalue)),
        (label, "Rachas arriba/abajo", *_runs_up_down(values)),
        (label, f"Ljung-Box rezagos 1-{lags}", *_ljung_box(values, lags)),
        (label, "Serial de tercias en el cubo", *_serial_three_dimensional(values)),
    ]


def holm_adjusted_p_values(p_values: np.ndarray) -> np.ndarray:
    """Corrige por multiplicidad con el método de Holm-Bonferroni.

    Con dieciocho contrastes independientes a ``alpha = 0.05``, la probabilidad
    de al menos un falso rechazo sería ``1 - 0.95**18``, cerca del 60 %. Aquí
    varios contrastes comparten muestras y no son independientes, por lo que ese
    porcentaje es solo una ilustración del problema de multiplicidad, no la FWER
    exacta de esta batería. Holm controla la tasa de error por familia sin exigir
    independencia y es uniformemente más potente que Bonferroni. El valor
    devuelto se compara contra el mismo ``alpha`` nominal.
    """

    total = len(p_values)
    order = np.argsort(p_values)
    ordered = np.asarray(p_values, dtype=float)[order]
    # El multiplicador decrece con el rango y la acumulación por máximo
    # garantiza que el ajuste conserve el orden de los valores originales.
    adjusted = np.maximum.accumulate((total - np.arange(total)) * ordered)
    result = np.empty(total, dtype=float)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def estimate_polar_acceptance(
    proposed_pairs: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Propone un número fijo de pares y cuenta cuántos caen en el círculo.

    ``sample_marsaglia_normals`` se detiene al completar la muestra pedida: el
    número de aceptaciones queda fijo y el de propuestas es aleatorio, que es lo
    contrario de lo que supone una prueba binomial. Este experimento invierte el
    diseño —``n`` fijo, éxitos aleatorios— para que el contraste contra ``pi/4``
    sea exacto.
    """

    if proposed_pairs < 100:
        raise ValueError("Se requieren al menos 100 pares propuestos")
    pairs = rng.uniform(-1.0, 1.0, size=(proposed_pairs, 2))
    squared_radius = np.sum(pairs**2, axis=1)
    accepted = int(np.count_nonzero((squared_radius > 0.0) & (squared_radius < 1.0)))
    return accepted, proposed_pairs


def validate_arrival_generators(
    config: SimulationConfig,
    sample_size: int = 4_000,
    count_repetitions: int = 600,
    benchmark_repetitions: int = 7,
    alpha: float = 0.05,
) -> GeneratorValidationResult:
    """Audita la cadena completa: fuente uniforme, transformaciones y conteo.

    La auditoría sigue el orden en que se construyen las variables. Primero se
    contrasta la fuente uniforme, porque tanto la transformada inversa como
    Marsaglia Polar son funciones de uniformes y un defecto allí contamina todo
    lo demás. Después se contrastan las distribuciones derivadas y, por último,
    el conteo ``N(T)`` que produce el calendario real.

    Cada contraste mide una propiedad que ningún otro implica. En particular, no
    se incluye una prueba de bondad de ajuste sobre la lognormal: el estadístico
    de Kolmogorov-Smirnov es invariante ante transformaciones monótonas y
    ``Delta = exp(mu + sigma Z)`` lo es, de modo que ese contraste devuelve el
    mismo número que el KS de ``Z`` y no aportaría evidencia nueva. Lo que el KS
    de ``Z`` no puede ver es si la parametrización cumple ``E[Delta] = 1/lambda``;
    para eso se contrasta la media.

    Los contrastes usan muestras de tamaño fijo para evitar la censura de un
    calendario finito. El conteo Poisson sí se valida generando calendarios
    completos repetidos, exactamente con la función usada por el simulador. Los
    tiempos de ejecución son medianas descriptivas del equipo actual y no
    participan en ninguna decisión estadística.
    """

    config.validate()
    if sample_size < 1_000 or count_repetitions < 30 or benchmark_repetitions < 1:
        raise ValueError("La validación requiere muestras y repeticiones suficientes")

    seed_sequence = np.random.SeedSequence(config.seed).spawn(6)
    exponential_rng = np.random.default_rng(seed_sequence[0])
    polar_rng = np.random.default_rng(seed_sequence[1])
    count_rng = np.random.default_rng(seed_sequence[2])
    benchmark_rng = np.random.default_rng(seed_sequence[3])
    uniform_rng = np.random.default_rng(seed_sequence[4])
    acceptance_rng = np.random.default_rng(seed_sequence[5])

    # --- Nivel 1: la fuente uniforme -------------------------------------
    # PCG64 es la fuente real del proyecto. El LCG propio se somete a la misma
    # batería para poder afirmar con evidencia, y no por autoridad, que la
    # elección de generador base no está deteriorando las variables derivadas.
    reference_uniforms = uniform_rng.uniform(0.0, 1.0, size=sample_size)
    lcg_seed = int(uniform_rng.integers(1, 2**31 - 1))
    own_lcg = LinearCongruentialGenerator(*MINSTD_PARAMETERS, seed=lcg_seed)
    own_uniforms = own_lcg.uniforms(sample_size)
    degenerate_lcg = LinearCongruentialGenerator(*COUNTER_PARAMETERS, seed=lcg_seed)
    degenerate_uniforms = degenerate_lcg.uniforms(sample_size)

    uniform_rows = (
        uniform_source_tests(reference_uniforms, "Fuente uniforme PCG64")
        + uniform_source_tests(own_uniforms, "LCG propio (MINSTD)")
    )

    # --- Nivel 2: las transformaciones -----------------------------------
    exponential = sample_exponential_interarrivals(
        config.lambda_rate, sample_size, exponential_rng
    )
    lognormal, normals, _, _ = sample_polar_lognormal_interarrivals(
        config.lambda_rate, config.polar_cv, sample_size, polar_rng
    )
    mu, sigma = _lognormal_parameters(config.lambda_rate, config.polar_cv)

    exponential_ks = stats.kstest(
        exponential, "expon", args=(0.0, 1.0 / config.lambda_rate)
    )
    normal_ks = stats.kstest(normals, "norm")
    exponential_lag = _pearson_lag_test(exponential)
    lognormal_lag = _pearson_lag_test(lognormal)
    # Contrasta la parametrización, no la forma: verifica que el par (mu, sigma)
    # traslade la lognormal a la misma tasa media que la exponencial.
    mean_test = stats.ttest_1samp(lognormal, 1.0 / config.lambda_rate)

    # El diseño con pares propuestos fijos convierte la prueba binomial en exacta.
    accepted_pairs, proposed_pairs = estimate_polar_acceptance(
        max(2_000, sample_size), acceptance_rng
    )
    acceptance_rate = accepted_pairs / proposed_pairs
    acceptance_test = stats.binomtest(
        accepted_pairs, proposed_pairs, p=np.pi / 4.0, alternative="two-sided"
    )

    # --- Nivel 3: el conteo del calendario --------------------------------
    count_seeds = count_rng.integers(1, 2**31 - 1, size=count_repetitions)
    counts = np.asarray([
        len(generate_poisson_arrivals(
            config.lambda_rate,
            config.duration,
            np.random.default_rng(int(seed)),
        ))
        for seed in count_seeds
    ], dtype=float)
    count_mean = float(counts.mean())
    count_variance = float(counts.var(ddof=1))
    dispersion_statistic = (count_repetitions - 1) * count_variance / count_mean
    dispersion_cdf = float(stats.chi2.cdf(dispersion_statistic, count_repetitions - 1))
    dispersion_p = min(1.0, 2.0 * min(dispersion_cdf, 1.0 - dispersion_cdf))
    expected_count = config.lambda_rate * config.duration
    count_chi, count_chi_p, _ = _poisson_count_chi_square(counts, expected_count)

    test_rows = uniform_rows + [
        ("Poisson-exponencial", "KS contra Exponencial", exponential_ks.statistic, exponential_ks.pvalue),
        ("Poisson-exponencial", "Correlación de rezago 1", *exponential_lag),
        ("Marsaglia Polar", "KS de Z contra Normal(0,1)", normal_ks.statistic, normal_ks.pvalue),
        ("Marsaglia Polar", "Aceptación contra pi/4", acceptance_rate, acceptance_test.pvalue),
        ("Polar-lognormal", "Media contra 1/lambda", float(mean_test.statistic), float(mean_test.pvalue)),
        ("Polar-lognormal", "Correlación de rezago 1", *lognormal_lag),
        ("Conteo Poisson", "Índice de dispersión", dispersion_statistic, dispersion_p),
        ("Conteo Poisson", "Ji-cuadrada contra pmf Poisson", count_chi, count_chi_p),
    ]
    tests = pd.DataFrame(
        test_rows, columns=["generator", "test", "statistic", "p_value"]
    )
    tests["alpha"] = alpha
    tests["passes"] = tests["p_value"] >= alpha
    # El criterio de decisión del proyecto es la columna corregida: con una
    # familia de este tamaño, leer solo el valor p nominal produciría rechazos
    # espurios con alta probabilidad.
    tests["p_holm"] = holm_adjusted_p_values(tests["p_value"].to_numpy(dtype=float))
    tests["passes_holm"] = tests["p_holm"] >= alpha

    # El control negativo se reporta aparte porque se espera que falle. Mezclarlo
    # con la tabla principal contaminaría el conteo de contrastes no rechazados.
    control_rows = uniform_source_tests(
        degenerate_uniforms, "LCG degenerado (control negativo)"
    )
    uniform_control = pd.DataFrame(
        control_rows, columns=["generator", "test", "statistic", "p_value"]
    )
    uniform_control["alpha"] = alpha
    uniform_control["passes"] = uniform_control["p_value"] >= alpha

    exponential_mean = 1.0 / config.lambda_rate
    exponential_variance = 1.0 / config.lambda_rate**2
    lognormal_variance = (config.polar_cv * exponential_mean) ** 2
    moments = pd.DataFrame([
        {
            "generator": "Poisson-exponencial",
            "mean_theoretical": exponential_mean,
            "mean_observed": float(exponential.mean()),
            "variance_theoretical": exponential_variance,
            "variance_observed": float(exponential.var(ddof=1)),
            "cv_theoretical": 1.0,
            "cv_observed": float(exponential.std(ddof=1) / exponential.mean()),
        },
        {
            "generator": "Polar-lognormal",
            "mean_theoretical": exponential_mean,
            "mean_observed": float(lognormal.mean()),
            "variance_theoretical": lognormal_variance,
            "variance_observed": float(lognormal.var(ddof=1)),
            "cv_theoretical": config.polar_cv,
            "cv_observed": float(lognormal.std(ddof=1) / lognormal.mean()),
        },
        {
            "generator": "Marsaglia Normal",
            "mean_theoretical": 0.0,
            "mean_observed": float(normals.mean()),
            "variance_theoretical": 1.0,
            "variance_observed": float(normals.var(ddof=1)),
            "cv_theoretical": float("nan"),
            "cv_observed": float("nan"),
        },
        {
            "generator": "Fuente uniforme PCG64",
            "mean_theoretical": 0.5,
            "mean_observed": float(reference_uniforms.mean()),
            "variance_theoretical": 1.0 / 12.0,
            "variance_observed": float(reference_uniforms.var(ddof=1)),
            "cv_theoretical": float("nan"),
            "cv_observed": float("nan"),
        },
        {
            "generator": "LCG propio (MINSTD)",
            "mean_theoretical": 0.5,
            "mean_observed": float(own_uniforms.mean()),
            "variance_theoretical": 1.0 / 12.0,
            "variance_observed": float(own_uniforms.var(ddof=1)),
            "cv_theoretical": float("nan"),
            "cv_observed": float("nan"),
        },
    ])

    # El benchmark usa tamaños idénticos y generadores nuevos en cada repetición.
    # Ambas rutas están vectorizadas con NumPy al mismo nivel de esfuerzo: la
    # variante original de Polar (bucle de Python, una propuesta por
    # iteración) mide estilo de código, no el método de aceptación-rechazo en
    # sí. Usar ``sample_polar_lognormal_interarrivals_vectorized`` aquí hace que
    # la comparación de tiempos sea atribuible al algoritmo.
    benchmark_rows: list[dict[str, float | int | str]] = []
    for model in ("Poisson-exponencial", "Polar-lognormal"):
        elapsed: list[float] = []
        for _ in range(benchmark_repetitions):
            local_seed = int(benchmark_rng.integers(1, 2**31 - 1))
            local_rng = np.random.default_rng(local_seed)
            start = perf_counter()
            if model == "Poisson-exponencial":
                sample_exponential_interarrivals(config.lambda_rate, sample_size, local_rng)
            else:
                sample_polar_lognormal_interarrivals_vectorized(
                    config.lambda_rate, config.polar_cv, sample_size, local_rng
                )
            elapsed.append((perf_counter() - start) * 1_000.0)
        median_ms = float(np.median(elapsed))
        benchmark_rows.append({
            "generator": model,
            "sample_size": sample_size,
            "repetitions": benchmark_repetitions,
            "median_ms": median_ms,
            "microseconds_per_value": median_ms * 1_000.0 / sample_size,
        })

    samples = pd.DataFrame({
        "exponential": exponential,
        "lognormal": lognormal,
        "normal_z": normals,
    })
    uniform_samples = pd.DataFrame({
        "pcg64": reference_uniforms,
        "lcg_minstd": own_uniforms,
        "lcg_degenerate": degenerate_uniforms,
    })
    poisson_counts = pd.DataFrame({"count": counts.astype(int)})
    return GeneratorValidationResult(
        tests=tests,
        moments=moments,
        samples=samples,
        poisson_counts=poisson_counts,
        performance=pd.DataFrame(benchmark_rows),
        uniform_samples=uniform_samples,
        uniform_control=uniform_control,
    )


def generate_poisson_arrivals(
    lambda_rate: float,
    duration: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Genera un calendario Poisson por transformada inversa.

    Args:
        lambda_rate: tasa constante de eventos por segundo.
        duration: horizonte hasta el que se conservan llegadas.
        rng: fuente de uniformes reproducible.

    Returns:
        DataFrame ordenado con una fila por llegada dentro del horizonte.

    Raises:
        ValueError: si la tasa o la duración no son positivas.
    """

    if lambda_rate <= 0 or duration <= 0:
        raise ValueError("lambda_rate y duration deben ser positivos")
    rows: list[dict[str, float | int | str]] = []
    arrival_time = 0.0
    enemy_id = 0
    while True:
        # Excluir cero evita ``log(0)``; NumPy ya excluye el extremo superior.
        u = float(rng.uniform(np.finfo(float).eps, 1.0))
        delta = float(-np.log(u) / lambda_rate)
        arrival_time += delta
        if arrival_time > duration:
            break
        enemy_id += 1
        rows.append({
            "enemy_id": enemy_id,
            "model": "poisson",
            "u": u,
            "polar_v1": np.nan,
            "polar_v2": np.nan,
            "z": np.nan,
            "delta": delta,
            "arrival_time": arrival_time,
        })
    return pd.DataFrame(rows, columns=ARRIVAL_COLUMNS)


def generate_polar_arrivals(
    lambda_rate: float,
    duration: float,
    rng: np.random.Generator,
    coefficient_variation: float = 0.45,
) -> pd.DataFrame:
    """Genera renovaciones lognormales mediante normales de Marsaglia Polar.

    La lognormal garantiza interarribos positivos. Sus parámetros se ajustan
    para que ``E[Delta] = 1/lambda`` y la comparación tenga una tasa media justa.

    Args:
        lambda_rate: frecuencia media objetivo en llegadas por segundo.
        duration: horizonte de observación.
        rng: fuente de uniformes usada por Marsaglia.
        coefficient_variation: desviación relativa de los interarribos.

    Returns:
        DataFrame con normales, uniformes aceptados e instantes acumulados.
    """

    if lambda_rate <= 0 or duration <= 0:
        raise ValueError("lambda_rate y duration deben ser positivos")
    if not 0.05 <= coefficient_variation <= 2.0:
        raise ValueError("coefficient_variation debe estar entre 0.05 y 2.0")
    mu, sigma = _lognormal_parameters(lambda_rate, coefficient_variation)
    normals = _marsaglia_normals(rng)
    rows: list[dict[str, float | int | str]] = []
    arrival_time = 0.0
    enemy_id = 0
    while True:
        z, v1, v2, _ = next(normals)
        delta = exp(mu + sigma * z)
        arrival_time += delta
        if arrival_time > duration:
            break
        enemy_id += 1
        rows.append({
            "enemy_id": enemy_id,
            "model": "polar",
            "u": np.nan,
            "polar_v1": v1,
            "polar_v2": v2,
            "z": z,
            "delta": delta,
            "arrival_time": arrival_time,
        })
    return pd.DataFrame(rows, columns=ARRIVAL_COLUMNS)


def generate_arrivals(config: SimulationConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Selecciona el generador sin duplicar lógica en ``simulate``.

    Args:
        config: configuración que contiene modelo, tasa, duración y CV.
        rng: flujo aleatorio reservado exclusivamente para llegadas.

    Returns:
        Calendario normalizado con las columnas de ``ARRIVAL_COLUMNS``.
    """

    if config.arrival_model == "poisson":
        return generate_poisson_arrivals(config.lambda_rate, config.duration, rng)
    return generate_polar_arrivals(
        config.lambda_rate, config.duration, rng, coefficient_variation=config.polar_cv
    )


def _distance(enemy: dict[str, object], time: float, contact_radius: float) -> float:
    """Calcula la distancia radial y la limita al radio de contacto.

    El límite inferior representa que un infectado no atraviesa al protagonista:
    una vez en contacto permanece allí hasta morir o finalizar la misión.
    """

    elapsed = max(0.0, time - float(enemy["spawn_time"]))
    return max(contact_radius, float(enemy["spawn_radius"]) - float(enemy["speed"]) * elapsed)


def simulate(config: SimulationConfig, keep_timeline: bool = True) -> SimulationResult:
    """Ejecuta una partida mediante integración temporal de paso fijo.

    Args:
        config: parámetros validados del escenario.
        keep_timeline: si es ``False``, omite muestras temporales para acelerar
            lotes Monte Carlo; los indicadores finales siempre se calculan.

    Returns:
        ``SimulationResult`` con indicadores, entidades y trazas de la corrida.

    Algorithm:
        1. Crear un calendario completo de llegadas.
        2. Asignar atributos independientes a cada enemigo.
        3. Activar solo entidades cuyo instante ya ocurrió.
        4. Seleccionar el objetivo, resolver ataque y sumar daño entrante.
        5. Muestrear telemetría y detener al morir o alcanzar el horizonte.
    """

    config.validate()
    # Separar los flujos evita que cambiar el algoritmo de llegadas modifique
    # también, por consumo accidental de uniformes, los atributos del enemigo i.
    arrival_seed, trait_seed = np.random.SeedSequence(config.seed).spawn(2)
    arrival_rng = np.random.default_rng(arrival_seed)
    trait_rng = np.random.default_rng(trait_seed)
    schedule = generate_arrivals(config, arrival_rng)

    # Preconstruir las entidades hace reproducible su identidad y permite que la
    # visualización consulte posiciones pasadas a partir de spawn/death_time.
    enemies: list[dict[str, object]] = []
    for row in schedule.itertuples(index=False):
        max_hp = float(config.enemy_hp * trait_rng.uniform(0.90, 1.12))
        enemies.append({
            "enemy_id": int(row.enemy_id),
            "spawn_time": float(row.arrival_time),
            "angle": float(trait_rng.uniform(0, 2 * np.pi)),
            "spawn_radius": float(config.arena_radius * trait_rng.uniform(0.94, 1.03)),
            "speed": float(config.enemy_speed * trait_rng.uniform(0.88, 1.12)),
            "max_hp": max_hp,
            "hp": max_hp,
            "dps": float(config.enemy_dps * trait_rng.uniform(0.90, 1.12)),
            "death_time": None,
            "killed": False,
        })

    player_hp = float(config.player_hp)
    eliminated = 0
    max_concurrent = 0
    spawn_cursor = 0
    active: list[int] = []
    time = 0.0
    next_sample = 0.5
    timeline_rows: list[dict[str, float | int]] = []
    survived = False

    if keep_timeline:
        # Registrar el estado inicial evita etiquetar como t=0 un estado que ya
        # recibió daño durante el primer paso numérico.
        timeline_rows.append({
            "time": 0.0,
            "player_hp": player_hp,
            "active_enemies": 0,
            "spawned": 0,
            "eliminated": 0,
            "incoming_dps": 0.0,
        })

    # ``active`` contiene índices, no copias de diccionarios. Esto reduce el
    # costo del bucle respecto a revisar todo el calendario en cada paso dt.
    while time < config.duration - 1e-12 and player_hp > 0:
        # El último intervalo puede ser menor que dt. Esta cota impide aplicar
        # daño o ataques después del horizonte [0, T].
        step = min(config.dt, config.duration - time)

        # Fase 1: materializar todas las llegadas ocurridas desde el paso previo.
        while spawn_cursor < len(enemies) and float(enemies[spawn_cursor]["spawn_time"]) <= time + 1e-12:
            active.append(spawn_cursor)
            spawn_cursor += 1

        # Fase 2: clasificar vivos, atacantes y candidato más cercano.
        target_idx: Optional[int] = None
        target_distance = float("inf")
        attackers: list[int] = []
        living: list[int] = []
        for idx in active:
            enemy = enemies[idx]
            if bool(enemy["killed"]):
                continue
            distance = _distance(enemy, time, config.contact_radius)
            living.append(idx)
            if distance <= config.contact_radius + 1e-9:
                attackers.append(idx)
            if distance <= config.weapon_range and distance < target_distance:
                target_idx = idx
                target_distance = distance
        active = living
        max_concurrent = max(max_concurrent, len(active))

        # Fase 3: el protagonista concentra todo su DPS en un único objetivo.
        if target_idx is not None:
            enemy = enemies[target_idx]
            enemy["hp"] = max(0.0, float(enemy["hp"]) - config.player_dps * step)
            if float(enemy["hp"]) <= 0:
                enemy["killed"] = True
                enemy["death_time"] = min(time + step, config.duration)
                eliminated += 1

        # Fase 4: enemigos de contacto que sobrevivieron al disparo dañan en
        # paralelo; por eso se suman sus DPS antes de integrar durante dt.
        incoming_dps = sum(
            float(enemies[idx]["dps"]) for idx in attackers if not bool(enemies[idx]["killed"])
        )
        player_hp = max(0.0, player_hp - incoming_dps * step)
        new_time = min(round(time + step, 10), config.duration)
        mission_finished = new_time >= config.duration - 1e-12
        player_fell = player_hp <= 0

        # La telemetría se reduce a 2 Hz para mantener pequeños los DataFrames y
        # las gráficas. Cada fila representa el estado al final del intervalo.
        if keep_timeline and (
            new_time >= next_sample - 1e-12 or mission_finished or player_fell
        ):
            timeline_rows.append({
                "time": round(new_time, 4),
                "player_hp": player_hp,
                "active_enemies": sum(not bool(enemies[idx]["killed"]) for idx in active),
                "spawned": spawn_cursor,
                "eliminated": eliminated,
                "incoming_dps": incoming_dps,
            })
            while next_sample <= new_time + 1e-12:
                next_sample += 0.5
        time = new_time

    survived = player_hp > 0 and time >= config.duration - 1e-12

    # Separar calendario y eventos realmente ocurridos corrige el caso en que el
    # jugador muere antes de T y el proceso tenía apariciones futuras planeadas.
    survival_time = config.duration if survived else time
    actual_arrivals = schedule[schedule["arrival_time"] <= survival_time + 1e-9].copy()
    generated = len(actual_arrivals)
    remaining = max(0, generated - eliminated)
    enemy_df = pd.DataFrame(enemies)
    if enemy_df.empty:
        enemy_df = pd.DataFrame(columns=[
            "enemy_id", "spawn_time", "angle", "spawn_radius", "speed",
            "max_hp", "hp", "dps", "death_time", "killed",
        ])
    timeline = pd.DataFrame(timeline_rows)
    if timeline.empty:
        timeline = pd.DataFrame([{
            "time": 0.0,
            "player_hp": player_hp,
            "active_enemies": 0,
            "spawned": 0,
            "eliminated": 0,
            "incoming_dps": 0.0,
        }])

    mean_interarrival = float(actual_arrivals["delta"].mean()) if not actual_arrivals.empty else float("nan")
    return SimulationResult(
        config=config,
        survived=survived,
        survival_time=float(survival_time),
        final_hp=float(player_hp),
        generated=generated,
        scheduled=len(schedule),
        eliminated=eliminated,
        remaining=remaining,
        max_concurrent=max_concurrent,
        mean_interarrival=mean_interarrival,
        # lambda*t es E[N(t)] exacto para Poisson. En una renovación lognormal
        # finita funciona únicamente como referencia de tasa, no como esperanza
        # exacta del conteo; la interfaz distingue ambos casos.
        expected_arrivals=config.lambda_rate * survival_time,
        expected_schedule=config.lambda_rate * config.duration,
        arrivals=actual_arrivals,
        arrival_schedule=schedule,
        enemies=enemy_df,
        timeline=timeline,
    )


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Calcula un intervalo bilateral de Wilson para una proporción binomial.

    Args:
        successes: número de partidas sobrevividas.
        trials: número total de partidas.
        z: cuantil normal; el valor predeterminado corresponde al 95 %.

    Returns:
        Límites inferior y superior recortados al intervalo [0, 1]. Si no hay
        ensayos, ambos límites son ``NaN``.
    """

    if trials <= 0:
        return float("nan"), float("nan")
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    centre = (proportion + z**2 / (2.0 * trials)) / denominator
    margin = z * sqrt(
        proportion * (1.0 - proportion) / trials + z**2 / (4.0 * trials**2)
    ) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def estimate_survival_probability(
    base_config: SimulationConfig,
    runs: int = 200,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Ejecuta partidas independientes y devuelve sus resultados individuales.

    ``keep_timeline=False`` evita construir datos que el resumen Monte Carlo no
    consume. La semilla maestra solo genera semillas de corrida; cada partida
    conserva internamente flujos separados para llegadas y atributos.
    """

    if runs <= 0:
        raise ValueError("runs debe ser positivo")
    master_rng = np.random.default_rng(base_config.seed if seed is None else seed)
    rows: list[dict[str, float | int | bool | str]] = []
    for run in range(1, runs + 1):
        run_seed = int(master_rng.integers(1, 2**31 - 1))
        result = simulate(replace(base_config, seed=run_seed), keep_timeline=False)
        rows.append({
            "run": run,
            "model": base_config.arrival_model,
            "seed": run_seed,
            "survived": result.survived,
            "survival_time": result.survival_time,
            "generated": result.generated,
            "eliminated": result.eliminated,
            "final_hp": result.final_hp,
            "max_concurrent": result.max_concurrent,
        })
    return pd.DataFrame(rows)


def summarize_batch(batch: pd.DataFrame, model: str) -> dict[str, float | int | str]:
    """Reduce un lote Monte Carlo a indicadores e IC de Wilson al 95 %.

    El resumen conserva el nombre del modelo para que las funciones visuales
    puedan aplicar etiquetas y colores sin inferirlos de columnas externas.
    """

    successes = int(batch["survived"].sum())
    trials = len(batch)
    ci_low, ci_high = wilson_interval(successes, trials)
    return {
        "model": model,
        "runs": trials,
        "survival_probability": successes / trials,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "mean_survival_time": float(batch["survival_time"].mean()),
        "mean_generated": float(batch["generated"].mean()),
        "mean_eliminated": float(batch["eliminated"].mean()),
        "mean_max_concurrent": float(batch["max_concurrent"].mean()),
    }


def compare_arrival_models(
    base_config: SimulationConfig,
    runs: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compara Poisson y Polar mediante un diseño de semillas pareadas.

    Returns:
        Una tupla con el detalle de las ``2 * runs`` partidas y una tabla de dos
        filas con los indicadores agregados de cada modelo.

    Notes:
        Emparejar semillas reduce ruido ajeno al modelo. No convierte las
        observaciones en idénticas: cada algoritmo consume sus uniformes de
        manera distinta, que es precisamente la diferencia estudiada.
    """

    if runs <= 0:
        raise ValueError("runs debe ser positivo")
    master_rng = np.random.default_rng(base_config.seed)
    seeds = [int(value) for value in master_rng.integers(1, 2**31 - 1, size=runs)]
    batches: list[pd.DataFrame] = []
    summaries: list[dict[str, float | int | str]] = []
    for model in ("poisson", "polar"):
        rows = []
        config = replace(base_config, arrival_model=model)
        for run, run_seed in enumerate(seeds, start=1):
            result = simulate(replace(config, seed=run_seed), keep_timeline=False)
            rows.append({
                "run": run,
                "model": model,
                "seed": run_seed,
                "survived": result.survived,
                "survival_time": result.survival_time,
                "generated": result.generated,
                "eliminated": result.eliminated,
                "final_hp": result.final_hp,
                "max_concurrent": result.max_concurrent,
            })
        batch = pd.DataFrame(rows)
        batches.append(batch)
        summaries.append(summarize_batch(batch, model))
    return pd.concat(batches, ignore_index=True), pd.DataFrame(summaries)


def paired_comparison_statistics(
    trials: pd.DataFrame,
    bootstrap_repetitions: int = 4_000,
    confidence: float = 0.95,
    seed: int = 91_337,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cuantifica diferencias pareadas entre Poisson y Polar-lognormal.

    La dirección de todos los efectos es Poisson menos Polar. Supervivencia usa
    la prueba exacta de McNemar porque la respuesta es binaria y las partidas
    comparten índice de corrida. Las métricas continuas usan Wilcoxon pareado.
    En todos los casos el intervalo del efecto se obtiene con bootstrap de pares,
    conservando juntos los dos resultados pertenecientes a una corrida.
    """

    required = {
        "run", "model", "survived", "survival_time", "generated", "max_concurrent"
    }
    missing = required.difference(trials.columns)
    if missing:
        raise ValueError(f"Faltan columnas para el análisis pareado: {sorted(missing)}")
    if bootstrap_repetitions < 200:
        raise ValueError("bootstrap_repetitions debe ser al menos 200")
    if not 0.80 <= confidence < 1.0:
        raise ValueError("confidence debe pertenecer a [0.80, 1.0)")

    ordered = trials.sort_values(["run", "model"])
    if set(ordered["model"]) != {"poisson", "polar"}:
        raise ValueError("Se requieren observaciones de Poisson y Polar")

    alpha = 1.0 - confidence
    rng = np.random.default_rng(seed)
    effect_rows: list[dict[str, float | str]] = []
    paired_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for metric in ("survived", "survival_time", "generated", "max_concurrent"):
        pivot = ordered.pivot(index="run", columns="model", values=metric).dropna()
        if len(pivot) == 0:
            raise ValueError(f"No existen pares completos para {metric}")
        poisson = pivot["poisson"].to_numpy(dtype=float)
        polar = pivot["polar"].to_numpy(dtype=float)
        paired_values[metric] = (poisson, polar)

    survival_poisson, survival_polar = paired_values["survived"]
    poisson_only = int(np.sum((survival_poisson == 1) & (survival_polar == 0)))
    polar_only = int(np.sum((survival_poisson == 0) & (survival_polar == 1)))
    discordant = poisson_only + polar_only
    mcnemar_p = (
        float(stats.binomtest(poisson_only, discordant, p=0.5).pvalue)
        if discordant
        else 1.0
    )

    for metric, (poisson, polar) in paired_values.items():
        differences = poisson - polar
        indices = rng.integers(
            0, len(differences), size=(bootstrap_repetitions, len(differences))
        )
        boot_means = differences[indices].mean(axis=1)
        ci_low, ci_high = np.quantile(
            boot_means, [alpha / 2.0, 1.0 - alpha / 2.0]
        )
        if metric == "survived":
            p_value = mcnemar_p
            test_name = "McNemar exacta"
        elif np.allclose(differences, 0.0):
            p_value = 1.0
            test_name = "Wilcoxon pareada"
        else:
            p_value = float(
                stats.wilcoxon(
                    differences,
                    zero_method="wilcox",
                    alternative="two-sided",
                    method="auto",
                ).pvalue
            )
            test_name = "Wilcoxon pareada"
        effect_rows.append({
            "metric": metric,
            "poisson_mean": float(poisson.mean()),
            "polar_mean": float(polar.mean()),
            "difference": float(differences.mean()),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "p_value": p_value,
            "test": test_name,
        })

    contingency = pd.DataFrame([
        {"outcome": "Ambos sobreviven", "count": int(np.sum((survival_poisson == 1) & (survival_polar == 1)))},
        {"outcome": "Solo Poisson sobrevive", "count": poisson_only},
        {"outcome": "Solo Polar sobrevive", "count": polar_only},
        {"outcome": "Ambos fallan", "count": int(np.sum((survival_poisson == 0) & (survival_polar == 0)))},
    ])
    return pd.DataFrame(effect_rows), contingency


def sweep_extreme_scenarios(
    base_config: SimulationConfig,
    lambda_rates: tuple[float, ...] = (0.45, 0.93, 1.60),
    coefficients_variation: tuple[float, ...] = (0.15, 0.60, 1.20),
    runs: int = 250,
) -> pd.DataFrame:
    """Recorre combinaciones extremas de tasa y CV con error Monte Carlo.

    Para cada punto de la rejilla ``lambda_rates x coefficients_variation`` se
    ejecuta una comparación pareada Poisson-Polar completa (mismas semillas
    entre modelos) y se reportan ambas probabilidades de supervivencia con su
    intervalo de Wilson al 95 %, además del número de pares discordantes que
    sostiene la prueba de McNemar en ese punto. El ancho del intervalo de
    Wilson es el error Monte Carlo del punto: con ``runs`` fijo, ese ancho es
    la evidencia de cuánta confianza merece cada celda de la rejilla, no solo
    su estimación puntual.
    """

    if runs <= 0:
        raise ValueError("runs debe ser positivo")
    rows: list[dict[str, float | int]] = []
    for lambda_rate in lambda_rates:
        for cv in coefficients_variation:
            cfg = replace(base_config, lambda_rate=lambda_rate, polar_cv=cv)
            cfg.validate()
            trials, summary = compare_arrival_models(cfg, runs=runs)
            _, contingency = paired_comparison_statistics(trials)
            discordant = int(
                contingency.loc[
                    contingency["outcome"].isin(
                        ["Solo Poisson sobrevive", "Solo Polar sobrevive"]
                    ),
                    "count",
                ].sum()
            )
            poisson_row = summary.loc[summary["model"] == "poisson"].iloc[0]
            polar_row = summary.loc[summary["model"] == "polar"].iloc[0]
            rows.append({
                "lambda_rate": lambda_rate,
                "arrivals_per_minute": lambda_rate * 60.0,
                "polar_cv": cv,
                "runs": runs,
                "poisson_survival": float(poisson_row["survival_probability"]),
                "poisson_ci_width": float(
                    poisson_row["ci_high"] - poisson_row["ci_low"]
                ),
                "polar_survival": float(polar_row["survival_probability"]),
                "polar_ci_width": float(
                    polar_row["ci_high"] - polar_row["ci_low"]
                ),
                "discordant_pairs": discordant,
            })
    return pd.DataFrame(rows)


DECISION_MATRIX_CRITERIA: tuple[tuple[str, float], ...] = (
    # (criterio, peso). Pesos para decidir entre dos métodos que generan
    # EXACTAMENTE la misma distribución objetivo, N(0,1): Marsaglia Polar y
    # Box-Muller (ver ``compare_normal_generators``). El ajuste distribucional
    # pesa más porque es la única razón de ser del generador; el costo y la
    # eficiencia de propuestas pesan más que la estabilidad numérica porque,
    # en este proyecto, ambos métodos ya evitan sus respectivos riesgos
    # conocidos (Polar rechaza en vez de invertir una raíz negativa; Box-Muller
    # recorta ``U1`` lejos de cero). Los pesos suman 1.0.
    ("Ajuste a la distribución objetivo (KS)", 0.30),
    ("Costo computacional (µs/valor, misma vectorización)", 0.25),
    ("Eficiencia y previsibilidad del costo (fracción no rechazada)", 0.20),
    ("Robustez numérica (funciones trascendentales, casos de borde)", 0.15),
    ("Sencillez de implementación y auditoría", 0.10),
)


def build_decision_matrix(
    generator_scores: dict[str, dict[str, float]],
    criteria: tuple[tuple[str, float], ...] = DECISION_MATRIX_CRITERIA,
) -> pd.DataFrame:
    """Combina puntajes por criterio (1-5) con pesos justificados en un ranking.

    Args:
        generator_scores: ``{nombre_del_metodo: {criterio: puntaje_1_a_5}}``.
            Los criterios cuantificables deben proceder de evidencia generada
            por el módulo (KS, benchmark y aceptación). Cuando un criterio sea
            necesariamente cualitativo, como facilidad de auditoría, su escala
            y justificación deben quedar declaradas junto a la matriz.
        criteria: pares ``(criterio, peso)``; los pesos deben sumar 1.0.

    Returns:
        Una fila por método con su puntaje ponderado total y el desglose por
        criterio, ordenada de mayor a menor puntaje.
    """

    total_weight = sum(weight for _, weight in criteria)
    if not np.isclose(total_weight, 1.0, atol=1e-6):
        raise ValueError(f"Los pesos deben sumar 1.0, suman {total_weight}")

    rows = []
    for method, scores in generator_scores.items():
        missing = [name for name, _ in criteria if name not in scores]
        if missing:
            raise ValueError(f"Faltan puntajes para {method}: {missing}")
        row: dict[str, float | str] = {"method": method}
        weighted_total = 0.0
        for name, weight in criteria:
            score = float(scores[name])
            if not 1.0 <= score <= 5.0:
                raise ValueError(f"{method}/{name} fuera de rango [1, 5]: {score}")
            row[name] = score
            weighted_total += weight * score
        row["weighted_total"] = weighted_total
        rows.append(row)
    return pd.DataFrame(rows).sort_values("weighted_total", ascending=False).reset_index(drop=True)


def normal_generator_decision_matrix(
    tests: pd.DataFrame,
    performance: pd.DataFrame,
) -> pd.DataFrame:
    """Construye la matriz de decisión Marsaglia Polar frente a Box-Muller.

    Los valores p se usan únicamente para decidir si cada método es compatible
    con ``N(0,1)``; un valor p mayor no se interpreta como mejor ajuste. El costo
    se puntúa de forma relativa respecto del método más rápido y la eficiencia
    usa la fracción de propuestas aprovechadas. Robustez y sencillez son
    calificaciones cualitativas explícitas: Polar evita funciones trigonométricas
    pero necesita rechazo y control de lotes; Box-Muller es directo, aunque debe
    proteger ``log(U_1)`` cuando ``U_1`` se aproxima a cero.

    La matriz es una ayuda multicriterio reproducible, no una prueba inferencial.
    Ambos métodos siguen siendo válidos si sus contrastes no se rechazan.
    """

    required_tests = {"method", "passes", "rejected_fraction"}
    required_performance = {"method", "microseconds_per_value"}
    if missing := required_tests.difference(tests.columns):
        raise ValueError(f"Faltan columnas de calidad: {sorted(missing)}")
    if missing := required_performance.difference(performance.columns):
        raise ValueError(f"Faltan columnas de rendimiento: {sorted(missing)}")

    evidence = tests.merge(
        performance[["method", "microseconds_per_value"]],
        on="method",
        how="inner",
        validate="one_to_one",
    )
    expected_methods = {"Marsaglia Polar", "Box-Muller"}
    if set(evidence["method"]) != expected_methods:
        raise ValueError("La decisión requiere evidencia de Polar y Box-Muller")
    if (evidence["microseconds_per_value"] <= 0).any():
        raise ValueError("Los tiempos del benchmark deben ser positivos")

    fastest = float(evidence["microseconds_per_value"].min())
    qualitative = {
        "Marsaglia Polar": {
            "Robustez numérica (funciones trascendentales, casos de borde)": 4.5,
            "Sencillez de implementación y auditoría": 3.5,
        },
        "Box-Muller": {
            "Robustez numérica (funciones trascendentales, casos de borde)": 4.0,
            "Sencillez de implementación y auditoría": 4.5,
        },
    }
    scores: dict[str, dict[str, float]] = {}
    for row in evidence.itertuples(index=False):
        method = str(row.method)
        cost = float(row.microseconds_per_value)
        accepted_fraction = 1.0 - float(row.rejected_fraction)
        scores[method] = {
            "Ajuste a la distribución objetivo (KS)": 5.0 if bool(row.passes) else 1.0,
            "Costo computacional (µs/valor, misma vectorización)": max(
                1.0, 5.0 * fastest / cost
            ),
            "Eficiencia y previsibilidad del costo (fracción no rechazada)": max(
                1.0, 5.0 * accepted_fraction
            ),
            **qualitative[method],
        }
    return build_decision_matrix(scores)


def survival_curve(
    base_config: SimulationConfig,
    lambdas: list[float],
    runs_per_lambda: int = 80,
) -> pd.DataFrame:
    """Estima supervivencia e incertidumbre sobre una cuadrícula de tasas.

    Cada tasa recibe un desplazamiento determinista de semilla para que los
    puntos sean reproducibles y no reutilicen exactamente el mismo lote.
    """

    rows: list[dict[str, float | str]] = []
    for index, lambda_rate in enumerate(lambdas):
        config = replace(base_config, lambda_rate=float(lambda_rate))
        batch = estimate_survival_probability(
            config, runs=runs_per_lambda, seed=base_config.seed + index * 1009
        )
        summary = summarize_batch(batch, config.arrival_model)
        rows.append({
            "model": config.arrival_model,
            "lambda": float(lambda_rate),
            "survival_probability": float(summary["survival_probability"]),
            "ci_low": float(summary["ci_low"]),
            "ci_high": float(summary["ci_high"]),
            "mean_survival_time": float(summary["mean_survival_time"]),
            "mean_generated": float(summary["mean_generated"]),
        })
    return pd.DataFrame(rows)


def config_as_dict(config: SimulationConfig) -> dict[str, object]:
    """Convierte la configuración inmutable en un diccionario serializable.

    Streamlit puede calcular una clave de caché estable para tipos simples; la
    conversión también facilita guardar el escenario en ``session_state``.
    """

    return asdict(config)

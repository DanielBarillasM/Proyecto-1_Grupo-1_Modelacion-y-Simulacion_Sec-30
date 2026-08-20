"""Motor estocástico del simulador de supervivencia.

Se comparan dos modelos con el mismo interarribo medio: un proceso de Poisson
homogéneo con interarribos exponenciales y un proceso de renovación con
interarribos lognormales generados mediante el método polar de Marsaglia.
El método polar no es un proceso de conteo; esta formulación evita presentar
como equivalentes dos conceptos matemáticos de distinta naturaleza.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import exp, log, sqrt
from typing import Iterator, Literal, Optional

import numpy as np
import pandas as pd


ArrivalModel = Literal["poisson", "polar"]


@dataclass(frozen=True)
class SimulationConfig:
    """Parámetros de una partida expresados en segundos, metros y puntos de vida."""

    duration: float = 90.0
    lambda_rate: float = 0.65
    arrival_model: ArrivalModel = "poisson"
    polar_cv: float = 0.45
    player_hp: float = 110.0
    player_dps: float = 42.0
    weapon_range: float = 10.0
    enemy_hp: float = 42.0
    enemy_speed: float = 1.55
    enemy_dps: float = 9.0
    arena_radius: float = 18.0
    contact_radius: float = 1.35
    dt: float = 0.05
    seed: int = 22193

    def validate(self) -> None:
        """Valida dominio, geometría y estabilidad numérica."""

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
    """Resultados, eventos y trazas de una partida."""

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


ARRIVAL_COLUMNS = [
    "enemy_id", "model", "u", "polar_v1", "polar_v2", "z", "delta", "arrival_time"
]


def _marsaglia_normals(rng: np.random.Generator) -> Iterator[tuple[float, float, float, float]]:
    """Produce normales estándar y conserva los uniformes que las originan."""

    while True:
        v1, v2 = rng.uniform(-1.0, 1.0, size=2)
        s = float(v1 * v1 + v2 * v2)
        if not 0.0 < s < 1.0:
            continue
        factor = sqrt(-2.0 * log(s) / s)
        yield float(v1 * factor), float(v1), float(v2), s
        yield float(v2 * factor), float(v1), float(v2), s


def _lognormal_parameters(lambda_rate: float, coefficient_variation: float) -> tuple[float, float]:
    """Calcula parámetros logarítmicos con media 1/lambda y CV indicado."""

    sigma_squared = log(1.0 + coefficient_variation**2)
    return log(1.0 / lambda_rate) - sigma_squared / 2.0, sqrt(sigma_squared)


def generate_poisson_arrivals(
    lambda_rate: float,
    duration: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Genera llegadas Poisson mediante ``Delta = -ln(U) / lambda``."""

    if lambda_rate <= 0 or duration <= 0:
        raise ValueError("lambda_rate y duration deben ser positivos")
    rows: list[dict[str, float | int | str]] = []
    arrival_time = 0.0
    enemy_id = 0
    while True:
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
    """Selecciona el generador de llegadas configurado."""

    if config.arrival_model == "poisson":
        return generate_poisson_arrivals(config.lambda_rate, config.duration, rng)
    return generate_polar_arrivals(
        config.lambda_rate, config.duration, rng, coefficient_variation=config.polar_cv
    )


def _distance(enemy: dict[str, object], time: float, contact_radius: float) -> float:
    elapsed = max(0.0, time - float(enemy["spawn_time"]))
    return max(contact_radius, float(enemy["spawn_radius"]) - float(enemy["speed"]) * elapsed)


def simulate(config: SimulationConfig, keep_timeline: bool = True) -> SimulationResult:
    """Ejecuta una partida con listas activas para evitar recorridos innecesarios."""

    config.validate()
    arrival_seed, trait_seed = np.random.SeedSequence(config.seed).spawn(2)
    arrival_rng = np.random.default_rng(arrival_seed)
    trait_rng = np.random.default_rng(trait_seed)
    schedule = generate_arrivals(config, arrival_rng)

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
    next_sample = 0.0
    timeline_rows: list[dict[str, float | int]] = []
    survived = False

    while True:
        while spawn_cursor < len(enemies) and float(enemies[spawn_cursor]["spawn_time"]) <= time + 1e-12:
            active.append(spawn_cursor)
            spawn_cursor += 1

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

        if target_idx is not None:
            enemy = enemies[target_idx]
            enemy["hp"] = max(0.0, float(enemy["hp"]) - config.player_dps * config.dt)
            if float(enemy["hp"]) <= 0:
                enemy["killed"] = True
                enemy["death_time"] = min(time + config.dt, config.duration)
                eliminated += 1

        incoming_dps = sum(
            float(enemies[idx]["dps"]) for idx in attackers if not bool(enemies[idx]["killed"])
        )
        player_hp = max(0.0, player_hp - incoming_dps * config.dt)
        mission_finished = time >= config.duration - 1e-12
        player_fell = player_hp <= 0

        if keep_timeline and (time >= next_sample - 1e-12 or mission_finished or player_fell):
            timeline_rows.append({
                "time": round(time, 4),
                "player_hp": player_hp,
                "active_enemies": sum(not bool(enemies[idx]["killed"]) for idx in active),
                "spawned": spawn_cursor,
                "eliminated": eliminated,
                "incoming_dps": incoming_dps,
            })
            next_sample += 0.5

        if player_fell:
            break
        if mission_finished:
            survived = True
            break
        time = min(round(time + config.dt, 10), config.duration)

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
        expected_arrivals=config.lambda_rate * survival_time,
        expected_schedule=config.lambda_rate * config.duration,
        arrivals=actual_arrivals,
        arrival_schedule=schedule,
        enemies=enemy_df,
        timeline=timeline,
    )


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Intervalo de Wilson bilateral para una proporción binomial."""

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
    """Ejecuta partidas independientes para estimar la supervivencia."""

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
    """Resume una muestra Monte Carlo e incluye IC de Wilson al 95 %."""

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
    """Compara Poisson y Polar usando las mismas semillas de corrida."""

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


def survival_curve(
    base_config: SimulationConfig,
    lambdas: list[float],
    runs_per_lambda: int = 80,
) -> pd.DataFrame:
    """Estima la curva de supervivencia con incertidumbre para un modelo."""

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
    """Representación serializable usada por la caché de Streamlit."""

    return asdict(config)

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    duration: float = 90.0
    lambda_rate: float = 0.75
    player_hp: float = 100.0
    player_dps: float = 38.0
    weapon_range: float = 10.0
    enemy_hp: float = 42.0
    enemy_speed: float = 1.55
    enemy_dps: float = 9.0
    arena_radius: float = 18.0
    contact_radius: float = 1.35
    dt: float = 0.05
    seed: int = 22193


@dataclass
class SimulationResult:
    config: SimulationConfig
    survived: bool
    survival_time: float
    final_hp: float
    generated: int
    eliminated: int
    remaining: int
    max_concurrent: int
    mean_interarrival: float
    expected_arrivals: float
    arrivals: pd.DataFrame
    enemies: pd.DataFrame
    timeline: pd.DataFrame


def generate_poisson_arrivals(
    lambda_rate: float,
    duration: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate homogeneous Poisson-process arrivals via exponential interarrivals.

    If U ~ Uniform(0, 1), then Delta = -ln(U)/lambda is Exponential(lambda).
    Arrival times are t_i = sum_{j<=i} Delta_j.
    """
    if lambda_rate <= 0:
        raise ValueError("lambda_rate must be > 0")
    if duration <= 0:
        raise ValueError("duration must be > 0")

    rows: list[dict[str, float | int]] = []
    t = 0.0
    i = 0
    while True:
        # Avoid exactly 0 so log(U) is always finite.
        u = float(rng.uniform(np.finfo(float).eps, 1.0))
        delta = float(-np.log(u) / lambda_rate)
        t += delta
        if t > duration:
            break
        i += 1
        rows.append({"enemy_id": i, "u": u, "delta": delta, "arrival_time": t})

    return pd.DataFrame(rows, columns=["enemy_id", "u", "delta", "arrival_time"])


def _enemy_position_at(row: pd.Series, t: float, contact_radius: float) -> tuple[float, float, float]:
    elapsed = max(0.0, t - float(row["spawn_time"]))
    radius = max(contact_radius, float(row["spawn_radius"]) - float(row["speed"]) * elapsed)
    angle = float(row["angle"])
    return radius * np.cos(angle), radius * np.sin(angle), 0.45


def simulate(config: SimulationConfig, keep_timeline: bool = True) -> SimulationResult:
    rng = np.random.default_rng(config.seed)
    arrivals = generate_poisson_arrivals(config.lambda_rate, config.duration, rng)

    enemies: list[dict[str, float | int | bool | None]] = []
    for row in arrivals.itertuples(index=False):
        angle = float(rng.uniform(0, 2 * np.pi))
        spawn_radius = float(config.arena_radius * rng.uniform(0.92, 1.04))
        speed = float(config.enemy_speed * rng.uniform(0.82, 1.18))
        hp = float(config.enemy_hp * rng.uniform(0.84, 1.18))
        dps = float(config.enemy_dps * rng.uniform(0.82, 1.20))
        enemies.append(
            {
                "enemy_id": int(row.enemy_id),
                "spawn_time": float(row.arrival_time),
                "angle": angle,
                "spawn_radius": spawn_radius,
                "speed": speed,
                "max_hp": hp,
                "hp": hp,
                "dps": dps,
                "death_time": None,
                "killed": False,
            }
        )

    player_hp = float(config.player_hp)
    eliminated = 0
    max_concurrent = 0
    t = 0.0
    timeline_rows: list[dict[str, float | int]] = []
    next_sample = 0.0

    # Pointer allows us to avoid scanning enemies that have not spawned yet for some operations.
    while t <= config.duration + 1e-9 and player_hp > 0:
        alive_indices: list[int] = []
        attackers: list[int] = []
        target_idx: Optional[int] = None
        target_distance = float("inf")

        for idx, enemy in enumerate(enemies):
            if enemy["spawn_time"] > t:
                continue
            if enemy["killed"]:
                continue

            elapsed = max(0.0, t - float(enemy["spawn_time"]))
            radius = max(
                config.contact_radius,
                float(enemy["spawn_radius"]) - float(enemy["speed"]) * elapsed,
            )
            alive_indices.append(idx)

            if radius <= config.contact_radius + 1e-9:
                attackers.append(idx)

            if radius <= config.weapon_range and radius < target_distance:
                target_idx = idx
                target_distance = radius

        concurrent = len(alive_indices)
        max_concurrent = max(max_concurrent, concurrent)

        # Player focuses one nearest target at a time.
        if target_idx is not None:
            enemy = enemies[target_idx]
            enemy["hp"] = float(enemy["hp"]) - config.player_dps * config.dt
            if float(enemy["hp"]) <= 0:
                enemy["hp"] = 0.0
                enemy["killed"] = True
                enemy["death_time"] = min(t + config.dt, config.duration)
                eliminated += 1

        # Contact enemies damage the player concurrently.
        if attackers:
            incoming_dps = sum(float(enemies[idx]["dps"]) for idx in attackers if not enemies[idx]["killed"])
            player_hp -= incoming_dps * config.dt
            player_hp = max(0.0, player_hp)

        if keep_timeline and (t + 1e-9 >= next_sample or player_hp <= 0):
            alive_now = sum(
                1
                for e in enemies
                if float(e["spawn_time"]) <= t and not bool(e["killed"])
            )
            spawned_now = int((arrivals["arrival_time"] <= t).sum()) if not arrivals.empty else 0
            timeline_rows.append(
                {
                    "time": round(t, 4),
                    "player_hp": player_hp,
                    "active_enemies": alive_now,
                    "spawned": spawned_now,
                    "eliminated": eliminated,
                }
            )
            next_sample += 0.5

        if player_hp <= 0:
            break
        t += config.dt

    survival_time = min(t, config.duration)
    survived = player_hp > 0 and survival_time >= config.duration - config.dt

    enemy_df = pd.DataFrame(enemies)
    if enemy_df.empty:
        enemy_df = pd.DataFrame(
            columns=[
                "enemy_id", "spawn_time", "angle", "spawn_radius", "speed",
                "max_hp", "hp", "dps", "death_time", "killed"
            ]
        )

    generated = int((arrivals["arrival_time"] <= survival_time).sum()) if not arrivals.empty else 0
    remaining = max(0, generated - eliminated)

    timeline_df = pd.DataFrame(timeline_rows)
    if timeline_df.empty:
        timeline_df = pd.DataFrame(
            [{"time": 0.0, "player_hp": player_hp, "active_enemies": 0, "spawned": 0, "eliminated": 0}]
        )

    mean_interarrival = float(arrivals["delta"].mean()) if not arrivals.empty else float("nan")

    return SimulationResult(
        config=config,
        survived=survived,
        survival_time=float(survival_time),
        final_hp=float(player_hp),
        generated=generated,
        eliminated=eliminated,
        remaining=remaining,
        max_concurrent=max_concurrent,
        mean_interarrival=mean_interarrival,
        expected_arrivals=config.lambda_rate * min(config.duration, survival_time),
        arrivals=arrivals,
        enemies=enemy_df,
        timeline=timeline_df,
    )


def estimate_survival_probability(
    base_config: SimulationConfig,
    runs: int = 200,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    if runs <= 0:
        raise ValueError("runs must be > 0")

    base_seed = base_config.seed if seed is None else seed
    master_rng = np.random.default_rng(base_seed)
    rows: list[dict[str, float | int | bool]] = []

    for i in range(runs):
        run_seed = int(master_rng.integers(0, 2**31 - 1))
        cfg = SimulationConfig(**{**base_config.__dict__, "seed": run_seed})
        result = simulate(cfg, keep_timeline=False)
        rows.append(
            {
                "run": i + 1,
                "survived": result.survived,
                "survival_time": result.survival_time,
                "generated": result.generated,
                "eliminated": result.eliminated,
                "final_hp": result.final_hp,
                "max_concurrent": result.max_concurrent,
            }
        )

    return pd.DataFrame(rows)


def survival_curve(
    base_config: SimulationConfig,
    lambdas: list[float],
    runs_per_lambda: int = 80,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for idx, lam in enumerate(lambdas):
        cfg = SimulationConfig(**{**base_config.__dict__, "lambda_rate": float(lam)})
        batch = estimate_survival_probability(cfg, runs=runs_per_lambda, seed=base_config.seed + idx * 1009)
        rows.append(
            {
                "lambda": float(lam),
                "survival_probability": float(batch["survived"].mean()),
                "mean_survival_time": float(batch["survival_time"].mean()),
                "mean_generated": float(batch["generated"].mean()),
            }
        )
    return pd.DataFrame(rows)

"""Pruebas de regresión y propiedades estadísticas del motor.

El conjunto comprueba tanto invariantes deterministas (reproducibilidad,
validación y emparejamiento de semillas) como propiedades que solo pueden
evaluarse sobre una muestra grande (media y varianza de los procesos).

Ejecución desde ``zombie_poisson_streamlit``::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

# The production modules live in ``App`` after the project reorganization.
# Insert that directory explicitly so the suite behaves the same when launched
# from the project folder, an IDE, or an automated CI runner.
APP_DIR = Path(__file__).resolve().parents[1] / "App"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from simulation import (
    ARRIVAL_COLUMNS,
    SimulationConfig,
    compare_arrival_models,
    generate_poisson_arrivals,
    generate_polar_arrivals,
    paired_comparison_statistics,
    simulate,
    validate_arrival_generators,
    wilson_interval,
)


class ArrivalGenerationTests(unittest.TestCase):
    """Valida los generadores antes de involucrar el motor de combate."""

    def test_poisson_is_reproducible(self) -> None:
        first = generate_poisson_arrivals(0.7, 90.0, np.random.default_rng(17))
        second = generate_poisson_arrivals(0.7, 90.0, np.random.default_rng(17))
        self.assertTrue(first.equals(second))

    def test_poisson_count_mean_and_variance_match_lambda_t(self) -> None:
        # El tamaño de muestra equilibra estabilidad estadística y velocidad.
        # Las tolerancias son más estrechas que la desviación típica esperada
        # del estimador, pero evitan una prueba frágil ante fluctuaciones válidas.
        lambda_rate, duration, repetitions = 0.4, 60.0, 1200
        master = np.random.default_rng(8102)
        counts = np.array([
            len(generate_poisson_arrivals(lambda_rate, duration, np.random.default_rng(int(seed))))
            for seed in master.integers(1, 2**31 - 1, size=repetitions)
        ])
        expected = lambda_rate * duration
        self.assertLess(abs(float(counts.mean()) - expected), 0.6)
        self.assertLess(abs(float(counts.var(ddof=1)) - expected), 2.2)

    def test_polar_lognormal_has_target_mean(self) -> None:
        # Un horizonte amplio ofrece suficientes interarribos para contrastar
        # la parametrización teórica E[Delta] = 1 / lambda.
        lambda_rate = 0.8
        arrivals = generate_polar_arrivals(
            lambda_rate,
            duration=50_000.0,
            rng=np.random.default_rng(91),
            coefficient_variation=0.45,
        )
        self.assertAlmostEqual(float(arrivals["delta"].mean()), 1 / lambda_rate, delta=0.025)
        self.assertTrue((arrivals["delta"] > 0).all())

    def test_formal_validation_accepts_reference_generators(self) -> None:
        # Una semilla fija convierte la auditoría estadística en una prueba de
        # regresión reproducible sin sustituir la interpretación de valores p.
        validation = validate_arrival_generators(
            SimulationConfig(seed=22193),
            sample_size=2_000,
            count_repetitions=300,
            benchmark_repetitions=1,
        )
        self.assertTrue(bool(validation.tests["passes"].all()))
        self.assertEqual(len(validation.tests), 7)
        self.assertEqual(set(validation.moments["generator"]), {
            "Poisson-exponencial", "Polar-lognormal", "Marsaglia Normal"
        })


class SimulationTests(unittest.TestCase):
    """Comprueba reglas de negocio e invariantes de una partida completa."""

    def test_full_simulation_is_reproducible(self) -> None:
        config = SimulationConfig(seed=3344)
        first, second = simulate(config), simulate(config)
        self.assertEqual(first.survived, second.survived)
        self.assertEqual(first.generated, second.generated)
        self.assertAlmostEqual(first.final_hp, second.final_hp)
        self.assertTrue(first.timeline.equals(second.timeline))

    def test_arrivals_after_player_death_are_not_counted_as_occurred(self) -> None:
        config = SimulationConfig(
            lambda_rate=2.2,
            player_hp=20,
            player_dps=10,
            enemy_hp=100,
            enemy_speed=3,
            enemy_dps=20,
            seed=22193,
        )
        result = simulate(config)
        self.assertFalse(result.survived)
        self.assertEqual(result.generated, len(result.arrivals))
        self.assertGreaterEqual(result.scheduled, result.generated)
        self.assertTrue((result.arrivals["arrival_time"] <= result.survival_time + 1e-9).all())

    def test_invalid_geometry_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            simulate(SimulationConfig(weapon_range=19.0, arena_radius=18.0))

    def test_comparison_uses_paired_seeds(self) -> None:
        trials, summary = compare_arrival_models(SimulationConfig(), runs=8)
        poisson_seeds = trials.loc[trials["model"] == "poisson", "seed"].tolist()
        polar_seeds = trials.loc[trials["model"] == "polar", "seed"].tolist()
        self.assertEqual(poisson_seeds, polar_seeds)
        self.assertEqual(set(summary["model"]), {"poisson", "polar"})

    def test_smaller_dt_gives_a_close_result_in_simple_scenario(self) -> None:
        # Reducir dt no tiene por qué producir igualdad bit a bit, pero sí una
        # conclusión y un HP final cercanos en un escenario de baja presión.
        config = SimulationConfig(lambda_rate=0.2, seed=19)
        coarse = simulate(replace(config, dt=0.05))
        fine = simulate(replace(config, dt=0.025))
        self.assertEqual(coarse.survived, fine.survived)
        self.assertLess(abs(coarse.final_hp - fine.final_hp), 3.0)

    @patch("simulation.generate_arrivals")
    def test_no_damage_is_applied_after_mission_horizon(self, mocked_arrivals) -> None:
        # El enemigo alcanza contacto exactamente en T. Debe contarse como
        # llegada, pero ya no dispone de un intervalo posterior para atacar.
        mocked_arrivals.return_value = pd.DataFrame([{
            "enemy_id": 1,
            "model": "poisson",
            "u": 0.5,
            "polar_v1": np.nan,
            "polar_v2": np.nan,
            "z": np.nan,
            "delta": 0.0,
            "arrival_time": 0.0,
        }], columns=ARRIVAL_COLUMNS)
        config = SimulationConfig(
            duration=0.05,
            dt=0.05,
            player_hp=10.0,
            player_dps=0.1,
            weapon_range=1.8,
            contact_radius=1.7,
            arena_radius=2.0,
            enemy_hp=100.0,
            enemy_speed=100.0,
            enemy_dps=1_000.0,
        )
        result = simulate(config)
        self.assertTrue(result.survived)
        self.assertAlmostEqual(result.final_hp, config.player_hp)
        self.assertAlmostEqual(result.survival_time, config.duration)

    def test_paired_statistics_preserve_direction_and_pairs(self) -> None:
        # Datos sintéticos permiten verificar exactamente la dirección
        # Poisson - Polar y el conteo usado por McNemar.
        rows = []
        poisson_survival = [1, 0, 0, 1, 0, 1]
        polar_survival = [1, 1, 1, 1, 1, 1]
        for run, (poisson, polar) in enumerate(
            zip(poisson_survival, polar_survival), start=1
        ):
            for model, survived in (("poisson", poisson), ("polar", polar)):
                rows.append({
                    "run": run,
                    "model": model,
                    "survived": bool(survived),
                    "survival_time": 90.0 if survived else 70.0,
                    "generated": 55 if model == "poisson" else 56,
                    "max_concurrent": 9 if model == "poisson" else 6,
                })
        effects, contingency = paired_comparison_statistics(
            pd.DataFrame(rows), bootstrap_repetitions=400, seed=7
        )
        survival = effects.loc[effects["metric"] == "survived"].iloc[0]
        self.assertAlmostEqual(float(survival["difference"]), -0.5)
        self.assertEqual(str(survival["test"]), "McNemar exacta")
        polar_only = contingency.loc[
            contingency["outcome"] == "Solo Polar sobrevive", "count"
        ].iloc[0]
        self.assertEqual(int(polar_only), 3)


class ConfidenceIntervalTests(unittest.TestCase):
    """Verifica límites básicos del intervalo binomial de Wilson."""

    def test_wilson_interval_contains_observed_proportion(self) -> None:
        low, high = wilson_interval(63, 100)
        self.assertLess(low, 0.63)
        self.assertGreater(high, 0.63)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)


if __name__ == "__main__":
    unittest.main()

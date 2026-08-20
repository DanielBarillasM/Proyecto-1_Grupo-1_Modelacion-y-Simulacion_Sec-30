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

import numpy as np

# The production modules live in ``App`` after the project reorganization.
# Insert that directory explicitly so the suite behaves the same when launched
# from the project folder, an IDE, or an automated CI runner.
APP_DIR = Path(__file__).resolve().parents[1] / "App"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from simulation import (
    SimulationConfig,
    compare_arrival_models,
    generate_poisson_arrivals,
    generate_polar_arrivals,
    simulate,
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

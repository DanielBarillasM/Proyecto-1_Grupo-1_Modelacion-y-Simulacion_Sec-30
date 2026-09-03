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
    COUNTER_PARAMETERS,
    DECISION_MATRIX_CRITERIA,
    MINSTD_PARAMETERS,
    LinearCongruentialGenerator,
    SimulationConfig,
    _poisson_count_chi_square,
    build_decision_matrix,
    compare_arrival_models,
    compare_normal_generators,
    estimate_polar_acceptance,
    generate_poisson_arrivals,
    generate_polar_arrivals,
    holm_adjusted_p_values,
    paired_comparison_statistics,
    sample_marsaglia_normals_vectorized,
    simulate,
    sweep_extreme_scenarios,
    uniform_source_tests,
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
        # El criterio es la columna corregida por Holm: con dieciocho contrastes
        # simultáneos, exigir que ninguno baje de 0.05 sería exigir suerte.
        validation = validate_arrival_generators(
            SimulationConfig(seed=22193),
            sample_size=2_000,
            count_repetitions=300,
            benchmark_repetitions=1,
        )
        self.assertTrue(bool(validation.tests["passes_holm"].all()))
        self.assertEqual(len(validation.tests), 18)
        self.assertEqual(set(validation.moments["generator"]), {
            "Poisson-exponencial", "Polar-lognormal", "Marsaglia Normal",
            "Fuente uniforme PCG64", "LCG propio (MINSTD)",
        })

    def test_no_two_contrasts_report_the_same_statistic(self) -> None:
        # Regresión del defecto corregido: el KS contra la lognormal devolvía
        # exactamente el mismo estadístico que el KS de Z, porque KS es
        # invariante ante la transformación monótona que las relaciona.
        validation = validate_arrival_generators(
            SimulationConfig(seed=22193),
            sample_size=2_000,
            count_repetitions=300,
            benchmark_repetitions=1,
        )
        statistics = validation.tests["statistic"].round(12)
        self.assertFalse(bool(statistics.duplicated().any()))
        self.assertNotIn("KS contra Lognormal", set(validation.tests["test"]))

    def test_negative_control_is_reported_outside_the_main_table(self) -> None:
        # El control negativo debe fallar; mezclarlo con la tabla principal
        # convertiría una demostración de potencia en un contraste rechazado.
        validation = validate_arrival_generators(
            SimulationConfig(seed=22193),
            sample_size=2_000,
            count_repetitions=300,
            benchmark_repetitions=1,
        )
        self.assertNotIn(
            "LCG degenerado (control negativo)", set(validation.tests["generator"])
        )
        self.assertFalse(bool(validation.uniform_control["passes"].all()))
        self.assertEqual(
            set(validation.uniform_samples.columns),
            {"pcg64", "lcg_minstd", "lcg_degenerate"},
        )


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


class UniformSourceTests(unittest.TestCase):
    """Audita la fuente uniforme, que es la entrada de todos los métodos."""

    def test_own_lcg_is_reproducible_and_stays_in_the_unit_interval(self) -> None:
        first = LinearCongruentialGenerator(*MINSTD_PARAMETERS, seed=4321).uniforms(500)
        second = LinearCongruentialGenerator(*MINSTD_PARAMETERS, seed=4321).uniforms(500)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(((first >= 0.0) & (first < 1.0)).all())

    def test_battery_accepts_the_reference_source(self) -> None:
        values = np.random.default_rng(515).uniform(0.0, 1.0, size=4_000)
        rows = uniform_source_tests(values, "PCG64")
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row[3] >= 0.05 for row in rows))

    def test_battery_separates_uniformity_from_independence(self) -> None:
        # El contador recorre su período en orden: es perfectamente uniforme y
        # perfectamente predecible. Si la batería tuviera potencia solo para la
        # forma marginal, este generador la aprobaría entera.
        values = LinearCongruentialGenerator(*COUNTER_PARAMETERS, seed=7).uniforms(4_000)
        results = {row[1]: row[3] for row in uniform_source_tests(values, "Contador")}
        self.assertGreaterEqual(results["Ji-cuadrada de uniformidad"], 0.05)
        self.assertGreaterEqual(results["KS contra U(0,1)"], 0.05)
        self.assertLess(results["Rachas arriba/abajo"], 0.001)
        self.assertLess(results["Ljung-Box rezagos 1-5"], 0.001)
        self.assertLess(results["Serial de tercias en el cubo"], 0.001)


class MultiplicityAndCountTests(unittest.TestCase):
    """Comprueba la corrección por familia y la bondad de ajuste del conteo."""

    def test_holm_matches_the_manual_computation(self) -> None:
        # Con m = 4 los factores son 4, 3, 2 y 1 sobre los valores ordenados:
        # 0.01*4 = 0.04, 0.03*3 = 0.09 y 0.04*2 = 0.08. La acumulación por
        # máximo eleva ese 0.08 hasta 0.09 para que el ajuste no invierta el
        # orden original de los valores p.
        adjusted = holm_adjusted_p_values(np.array([0.01, 0.04, 0.03, 0.60]))
        np.testing.assert_allclose(adjusted, [0.04, 0.09, 0.09, 0.60], atol=1e-12)

    def test_holm_never_exceeds_one_and_preserves_order(self) -> None:
        raw = np.array([0.2, 0.5, 0.9, 0.02, 0.33])
        adjusted = holm_adjusted_p_values(raw)
        self.assertTrue((adjusted <= 1.0).all())
        self.assertTrue((adjusted >= raw).all())

    def test_fixed_proposal_design_recovers_the_circle_acceptance(self) -> None:
        accepted, proposed = estimate_polar_acceptance(
            20_000, np.random.default_rng(63)
        )
        self.assertEqual(proposed, 20_000)
        self.assertAlmostEqual(accepted / proposed, np.pi / 4.0, delta=0.01)

    def test_count_chi_square_detects_a_wrong_mean(self) -> None:
        # La prueba debe aceptar la media correcta y rechazar una desplazada;
        # sin esa segunda mitad no habría evidencia de que tiene potencia.
        counts = np.random.default_rng(404).poisson(58.5, size=600).astype(float)
        _, correct_p, degrees = _poisson_count_chi_square(counts, 58.5)
        _, shifted_p, _ = _poisson_count_chi_square(counts, 66.0)
        self.assertGreaterEqual(correct_p, 0.05)
        self.assertLess(shifted_p, 0.01)
        self.assertGreater(degrees, 2)


class ConfidenceIntervalTests(unittest.TestCase):
    """Verifica límites básicos del intervalo binomial de Wilson."""

    def test_wilson_interval_contains_observed_proportion(self) -> None:
        low, high = wilson_interval(63, 100)
        self.assertLess(low, 0.63)
        self.assertGreater(high, 0.63)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)


class MethodComparisonTests(unittest.TestCase):
    """P2: comparación de métodos de generación para la misma distribución."""

    def test_vectorized_polar_matches_the_theoretical_acceptance_rate(self) -> None:
        # La vectorización por lotes no debe alterar la probabilidad de
        # aceptación teórica (pi/4): solo cambia cuánto trabajo por propuesta
        # se delega a NumPy en vez de al bucle de Python.
        values, acceptance_rate, proposed = sample_marsaglia_normals_vectorized(
            20_000, np.random.default_rng(11)
        )
        self.assertEqual(len(values), 20_000)
        self.assertAlmostEqual(acceptance_rate, np.pi / 4.0, delta=0.02)
        # Se proponen pares (dos normales por par aceptado), así que el total
        # de pares propuestos ronda la mitad de la muestra dividida por pi/4.
        self.assertGreater(proposed, 10_000)

    def test_polar_and_box_muller_both_fit_the_same_target_distribution(self) -> None:
        # Ambos métodos generan N(0,1): a diferencia del benchmark de procesos
        # de llegada, aquí sí se comparan dos algoritmos para un mismo objetivo.
        tests, performance = compare_normal_generators(sample_size=8_000, benchmark_repetitions=3)
        self.assertEqual(set(tests["method"]), {"Marsaglia Polar", "Box-Muller"})
        self.assertTrue(tests["passes"].all())
        box_muller_row = tests.loc[tests["method"] == "Box-Muller"].iloc[0]
        polar_row = tests.loc[tests["method"] == "Marsaglia Polar"].iloc[0]
        # Box-Muller no rechaza propuestas; Polar rechaza cerca de 1 - pi/4.
        self.assertEqual(box_muller_row["rejected_fraction"], 0.0)
        self.assertAlmostEqual(polar_row["rejected_fraction"], 1.0 - np.pi / 4.0, delta=0.03)
        self.assertEqual(len(performance), 2)
        self.assertTrue((performance["microseconds_per_value"] > 0).all())

    def test_decision_matrix_requires_weights_summing_to_one(self) -> None:
        bad_criteria = (("único criterio", 0.5),)
        with self.assertRaises(ValueError):
            build_decision_matrix({"A": {"único criterio": 3.0}}, criteria=bad_criteria)

    def test_decision_matrix_ranks_the_higher_scoring_method_first(self) -> None:
        criteria_names = [name for name, _ in DECISION_MATRIX_CRITERIA]
        scores = {
            "Método fuerte": {name: 5.0 for name in criteria_names},
            "Método débil": {name: 2.0 for name in criteria_names},
        }
        matrix = build_decision_matrix(scores)
        self.assertEqual(matrix.iloc[0]["method"], "Método fuerte")
        self.assertGreater(matrix.iloc[0]["weighted_total"], matrix.iloc[1]["weighted_total"])


class ScenarioCalibrationTests(unittest.TestCase):
    """P2: el escenario base ya no debe saturar la comparación pareada."""

    def test_default_scenario_is_not_saturated_at_the_ceiling(self) -> None:
        # Antes de la recalibración, Polar sobrevivía 400/400 corridas y solo
        # 14 pares discordantes sostenían McNemar. Con la configuración por
        # defecto actual, ambos modelos deben quedar lejos de 0 % y 100 %, con
        # bastantes más pares discordantes para que la prueba tenga potencia.
        cfg = SimulationConfig()
        trials, summary = compare_arrival_models(cfg, runs=150)
        _, contingency = paired_comparison_statistics(trials)
        poisson_survival = summary.loc[summary["model"] == "poisson", "survival_probability"].iloc[0]
        polar_survival = summary.loc[summary["model"] == "polar", "survival_probability"].iloc[0]
        discordant = int(
            contingency.loc[
                contingency["outcome"].isin(["Solo Poisson sobrevive", "Solo Polar sobrevive"]),
                "count",
            ].sum()
        )
        self.assertTrue(0.05 < poisson_survival < 0.95)
        self.assertTrue(0.05 < polar_survival < 0.95)
        self.assertGreater(discordant, 40)

    def test_sweep_reports_wider_confidence_intervals_at_the_calibrated_rate(self) -> None:
        # El punto elegido como escenario base debe ser, precisamente, donde el
        # barrido muestra más incertidumbre Monte Carlo (mayor ancho de Wilson)
        # y más pares discordantes: es el punto de mayor información, no un
        # punto arbitrario dentro de la rejilla.
        cfg = SimulationConfig()
        sweep = sweep_extreme_scenarios(
            cfg,
            lambda_rates=(0.65, 0.93, 1.25),
            coefficients_variation=(0.60,),
            runs=80,
        )
        calibrated = sweep.loc[sweep["lambda_rate"] == 0.93].iloc[0]
        extremes = sweep.loc[sweep["lambda_rate"] != 0.93]
        self.assertGreater(calibrated["discordant_pairs"], extremes["discordant_pairs"].max())


if __name__ == "__main__":
    unittest.main()

import io
import re
import warnings
import unittest
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

import functions


def _build_core_fixture():
    champions = ["A", "B", "C"]
    data = pd.DataFrame(
        [
            [np.nan, 52.0, 30.0],
            [60.0, np.nan, 49.0],
            [46.0, 58.0, np.nan],
        ],
        index=champions,
        columns=champions,
        dtype=np.float64,
    )
    games = pd.DataFrame(
        [
            [np.nan, 1.0, 9.0],
            [2.0, np.nan, 1.0],
            [9.0, 1.0, np.nan],
        ],
        index=champions,
        columns=champions,
        dtype=np.float64,
    )
    return data, games


def _legacy_column_mae(data, baseline=50.0):
    baseline_series = pd.Series(float(baseline), index=data.columns, dtype=np.float64)
    return data.sub(baseline_series, axis=1).abs().mean(axis=0, skipna=True).sort_values(ascending=False)


def _legacy_column_equalized_mae(data, games_data, baseline=50.0):
    games_numeric = games_data.where(games_data > 0)
    column_totals = games_numeric.sum(axis=0, skipna=True)
    pick_rates = column_totals / column_totals.sum(skipna=True)
    linear_pick_rates = pick_rates / pick_rates.max(skipna=True)
    linear_weights = pd.DataFrame(
        np.broadcast_to(linear_pick_rates.to_numpy(dtype=np.float64), games_numeric.shape),
        index=games_numeric.index,
        columns=games_numeric.columns,
        dtype=np.float64,
    ).where(games_numeric.notna())
    equalized = linear_weights.div(linear_weights.mean(axis=0, skipna=True), axis=1)
    baseline_series = pd.Series(float(baseline), index=data.columns, dtype=np.float64)
    weighted_absolute_errors = data.sub(baseline_series, axis=1).abs() * equalized
    return weighted_absolute_errors.mean(axis=0, skipna=True).sort_values(ascending=False)


class MetricMathTests(unittest.TestCase):
    def test_mae_matches_manual_formula(self):
        data, _ = _build_core_fixture()
        result = functions.calculate_champ_mae(data, baseline=50.0)

        self.assertAlmostEqual(result.loc["A"], 11.0, places=9)
        self.assertAlmostEqual(result.loc["B"], 5.5, places=9)
        self.assertAlmostEqual(result.loc["C"], 6.0, places=9)

    def test_weighted_and_equalized_mae_match_manual_weighted_formula(self):
        data, games = _build_core_fixture()
        plain = functions.calculate_champ_mae(data, baseline=50.0)
        weighted = functions.calculate_champ_mae_games_weighed(data, games, baseline=50.0)
        equalized = functions.calculate_champ_mae_games_weighed_equalized(data, games, baseline=50.0)

        self.assertAlmostEqual(weighted.loc["A"], 18.2, places=9)
        self.assertAlmostEqual(weighted.loc["B"], 7.0, places=9)
        self.assertAlmostEqual(weighted.loc["C"], 4.4, places=9)
        pd.testing.assert_series_equal(weighted.sort_index(), equalized.sort_index(), rtol=1e-12, atol=1e-12)
        self.assertGreater(abs(equalized.loc["A"] - plain.loc["A"]), 1e-9)

    def test_rowwise_rescaling_invariance_for_equalized_mae(self):
        data, games = _build_core_fixture()
        scaled_games = games.copy()
        scaled_games.loc["A"] = scaled_games.loc["A"] * 13.0
        scaled_games.loc["C"] = scaled_games.loc["C"] * 0.25

        baseline = 50.0
        original = functions.calculate_champ_mae_games_weighed_equalized(data, games, baseline=baseline)
        scaled = functions.calculate_champ_mae_games_weighed_equalized(data, scaled_games, baseline=baseline)
        pd.testing.assert_series_equal(original.sort_index(), scaled.sort_index(), rtol=1e-12, atol=1e-12)

    def test_weight_orientation_is_row_based(self):
        champions = ["A", "B", "C"]
        data = pd.DataFrame(
            [
                [np.nan, 51.0, 41.0],
                [55.0, np.nan, 45.0],
                [59.0, 51.0, np.nan],
            ],
            index=champions,
            columns=champions,
            dtype=np.float64,
        )
        games = pd.DataFrame(
            [
                [np.nan, 100.0, 1.0],
                [1.0, np.nan, 100.0],
                [50.0, 2.0, np.nan],
            ],
            index=champions,
            columns=champions,
            dtype=np.float64,
        )
        result = functions.calculate_champ_mae_games_weighed(data, games, baseline=50.0)

        self.assertAlmostEqual(result.loc["A"], (100.0 * 1.0 + 1.0 * 9.0) / 101.0, places=9)
        self.assertAlmostEqual(result.loc["B"], 5.0, places=9)
        self.assertAlmostEqual(result.loc["C"], (50.0 * 9.0 + 2.0 * 1.0) / 52.0, places=9)

    def test_masking_and_missing_weight_rows_return_nan_with_warning(self):
        champions = ["A", "B", "C"]
        data = pd.DataFrame(
            [
                [np.nan, 55.0, np.nan],
                [52.0, np.nan, 48.0],
                [50.0, 60.0, np.nan],
            ],
            index=champions,
            columns=champions,
            dtype=np.float64,
        )
        games = pd.DataFrame(
            [
                [np.nan, 10.0, 5.0],
                [np.nan, np.nan, np.nan],
                [4.0, 6.0, np.nan],
            ],
            index=champions,
            columns=champions,
            dtype=np.float64,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = functions.calculate_champ_mae_games_weighed_equalized(data, games, baseline=50.0)

        self.assertAlmostEqual(result.loc["A"], 5.0, places=9)
        self.assertAlmostEqual(result.loc["C"], 6.0, places=9)
        self.assertTrue(np.isnan(result.loc["B"]))
        self.assertTrue(any("Returning NaN" in str(item.message) for item in caught))

    def test_metric_entrypoints_use_shared_helper_pipeline(self):
        data, games = _build_core_fixture()
        baseline = 50.0
        metric_cases = [
            (
                functions.calculate_champ_rmse,
                {"error_kind": "squared", "reduction": "mean", "weighted": False, "equalize": False},
            ),
            (
                functions.calculate_champ_rmedse,
                {"error_kind": "squared", "reduction": "median", "weighted": False, "equalize": False},
            ),
            (
                functions.calculate_champ_mae,
                {"error_kind": "absolute", "reduction": "mean", "weighted": False, "equalize": False},
            ),
            (
                functions.calculate_champ_medae,
                {"error_kind": "absolute", "reduction": "median", "weighted": False, "equalize": False},
            ),
            (
                functions.calculate_champ_rmse_games_weighed,
                {"error_kind": "squared", "reduction": "mean", "weighted": True, "equalize": False},
            ),
            (
                functions.calculate_champ_rmedse_games_weighed,
                {"error_kind": "squared", "reduction": "median", "weighted": True, "equalize": False},
            ),
            (
                functions.calculate_champ_mae_games_weighed,
                {"error_kind": "absolute", "reduction": "mean", "weighted": True, "equalize": False},
            ),
            (
                functions.calculate_champ_medae_games_weighed,
                {"error_kind": "absolute", "reduction": "median", "weighted": True, "equalize": False},
            ),
            (
                functions.calculate_champ_rmse_games_weighed_equalized,
                {"error_kind": "squared", "reduction": "mean", "weighted": True, "equalize": True},
            ),
            (
                functions.calculate_champ_rmedse_games_weighed_equalized,
                {"error_kind": "squared", "reduction": "median", "weighted": True, "equalize": True},
            ),
            (
                functions.calculate_champ_mae_games_weighed_equalized,
                {"error_kind": "absolute", "reduction": "mean", "weighted": True, "equalize": True},
            ),
            (
                functions.calculate_champ_medae_games_weighed_equalized,
                {"error_kind": "absolute", "reduction": "median", "weighted": True, "equalize": True},
            ),
        ]

        for public_fn, helper_kwargs in metric_cases:
            with self.subTest(metric=public_fn.__name__):
                expected = functions._calculate_champion_error_metric(
                    data,
                    baseline=baseline,
                    games_data=games,
                    metric_name="test_metric",
                    **helper_kwargs,
                )
                actual = public_fn(data, games, baseline=baseline) if helper_kwargs["weighted"] else public_fn(data, baseline=baseline)
                pd.testing.assert_series_equal(actual.sort_index(), expected.sort_index(), rtol=1e-12, atol=1e-12)

    def test_visualize_entrypoint_matches_calculator(self):
        data, games = _build_core_fixture()
        expected = functions.calculate_champ_mae_games_weighed_equalized(data, games, baseline=50.0)
        expected_ci_by_champ = functions._calculate_matchup_ci_by_champion(data, confidence_level=0.95)

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            functions.visualize_mae_games_weighed_equalized(data, games, baseline=50.0)

        output_lines = [line.strip() for line in buffer.getvalue().splitlines() if line.strip()]
        first_ranked = output_lines[1]
        match = re.match(r"^\d+\.\s+([^:]+):\s+(-?\d+(?:\.\d+)?)%\s+\((-?\d+(?:\.\d+)?)%\s+-\s+(-?\d+(?:\.\d+)?)%\)$", first_ranked)
        self.assertIsNotNone(match, f"Unexpected visualize output format: {first_ranked}")

        champion = match.group(1).lower()
        score = float(match.group(2))
        ci_low = float(match.group(3))
        ci_high = float(match.group(4))

        expected_ci_low, expected_ci_high = expected_ci_by_champ[expected.index[0]]

        self.assertEqual(champion, expected.index[0].lower())
        self.assertAlmostEqual(score, round(expected.iloc[0], 2), places=9)
        self.assertAlmostEqual(ci_low, round(expected_ci_low, 2), places=9)
        self.assertAlmostEqual(ci_high, round(expected_ci_high, 2), places=9)

    def test_regression_prior_equalized_bug(self):
        data, games = _build_core_fixture()
        legacy_plain = _legacy_column_mae(data, baseline=50.0)
        legacy_equalized = _legacy_column_equalized_mae(data, games, baseline=50.0)
        pd.testing.assert_series_equal(legacy_plain.sort_index(), legacy_equalized.sort_index(), rtol=1e-12, atol=1e-12)

        corrected_plain = functions.calculate_champ_mae(data, baseline=50.0)
        corrected_equalized = functions.calculate_champ_mae_games_weighed_equalized(data, games, baseline=50.0)
        deltas = (corrected_equalized - corrected_plain).abs()
        self.assertTrue((deltas > 1e-9).any())

    def test_missing_baseline_raises_value_error(self):
        data, _ = _build_core_fixture()
        with self.assertRaisesRegex(ValueError, "Baseline win rates are required"):
            functions.calculate_champ_mae(data, baseline=None)

    def test_incomplete_baseline_alignment_raises_value_error(self):
        data, _ = _build_core_fixture()
        partial_baseline = pd.Series({"A": 50.0, "B": 50.0}, dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "Baseline is missing for one or more champions"):
            functions.calculate_champ_mae(data, baseline=partial_baseline)

    def test_procedural_baseline_from_bulk_matchup_data_uses_games_weighted_mean(self):
        raw_data = pd.DataFrame(
            {
                "source_champion": ["A", "A", "B", "B"],
                "opponent_champion": ["X", "Y", "X", "Y"],
                "win_rate": [0.60, 0.40, 55.0, 45.0],
                "games": [10, 30, 20, 20],
            }
        )
        baselines = functions._derive_champion_baselines_from_raw_data(raw_data, ["A", "B", "C"])

        self.assertAlmostEqual(float(baselines.loc["A"]), 45.0, places=9)
        self.assertAlmostEqual(float(baselines.loc["B"]), 50.0, places=9)
        self.assertTrue(np.isnan(baselines.loc["C"]))

    def test_source_page_baseline_takes_precedence_when_present(self):
        raw_data = pd.DataFrame(
            {
                "source_champion": ["A", "A", "B"],
                "opponent_champion": ["X", "Y", "X"],
                "source_page_win_rate": [0.52, 0.52, 0.48],
                "win_rate": [90.0, 10.0, 90.0],
                "games": [100, 100, 100],
            }
        )
        baselines = functions._derive_champion_baselines_from_raw_data(raw_data, ["A", "B"])

        self.assertAlmostEqual(float(baselines.loc["A"]), 52.0, places=9)
        self.assertAlmostEqual(float(baselines.loc["B"]), 48.0, places=9)

    def test_source_page_baseline_missing_values_fall_back_to_bulk_matchup_data(self):
        raw_data = pd.DataFrame(
            {
                "source_champion": ["A", "A", "B", "B"],
                "opponent_champion": ["X", "Y", "X", "Y"],
                "source_page_win_rate": [0.52, 0.52, np.nan, np.nan],
                "win_rate": [90.0, 10.0, 0.60, 0.40],
                "games": [100, 100, 10, 30],
            }
        )
        baselines = functions._derive_champion_baselines_from_raw_data(raw_data, ["A", "B"])

        self.assertAlmostEqual(float(baselines.loc["A"]), 52.0, places=9)
        self.assertAlmostEqual(float(baselines.loc["B"]), 45.0, places=9)


class PoolAnalysisTests(unittest.TestCase):
    def _build_pool_fixture(self):
        champions = ["A", "B", "C", "D"]
        data = pd.DataFrame(
            [
                [np.nan, 52.0, 48.0, 50.0],
                [40.0, np.nan, 55.0, 60.0],
                [45.0, 65.0, np.nan, 35.0],
                [70.0, 30.0, 50.0, np.nan],
            ],
            index=champions,
            columns=champions,
            dtype=np.float64,
        )
        return data

    def test_best_champ_additions_with_ci_uses_matchup_distribution_bounds(self):
        data = self._build_pool_fixture()
        result = functions.calculate_best_champ_additions_with_ci(
            data,
            current_pool=["A"],
            confidence_level=0.95,
        )

        by_champion = {champ: (score, low, high, midpoint) for champ, score, low, high, midpoint in result}
        score, low, high, midpoint = by_champion["B"]

        expected_distribution = data[["A", "B"]].max(axis=1, skipna=True)
        expected_score = float(expected_distribution.mean(skipna=True))
        expected_low = float(expected_distribution.quantile(0.025))
        expected_high = float(expected_distribution.quantile(0.975))
        expected_midpoint = (expected_low + expected_high) / 2.0

        self.assertAlmostEqual(score, expected_score, places=9)
        self.assertAlmostEqual(low, expected_low, places=9)
        self.assertAlmostEqual(high, expected_high, places=9)
        self.assertAlmostEqual(midpoint, expected_midpoint, places=9)
        self.assertAlmostEqual(midpoint, (low + high) / 2.0, places=12)

    def test_best_champ_additions_backward_compatible_with_legacy_shape(self):
        data = self._build_pool_fixture()
        legacy = functions.calculate_best_champ_additions(data, current_pool=["A"])
        with_ci = functions.calculate_best_champ_additions_with_ci(data, current_pool=["A"], confidence_level=0.95)

        self.assertEqual([champ for champ, _ in legacy], [champ for champ, *_ in with_ci])
        for (_, legacy_score), (_, score, *_rest) in zip(legacy, with_ci):
            self.assertAlmostEqual(legacy_score, score, places=9)

    def test_best_champ_additions_with_ci_rejects_invalid_confidence_level(self):
        data = self._build_pool_fixture()
        with self.assertRaisesRegex(ValueError, "confidence_level must be between 0 and 1"):
            functions.calculate_best_champ_additions_with_ci(data, current_pool=["A"], confidence_level=1.0)


class ChampPickAnalysisTests(unittest.TestCase):
    def _build_fixture(self):
        champions = ["A", "B", "C"]
        return pd.DataFrame(
            [
                [np.nan, 10.0, 20.0],
                [90.0, np.nan, 30.0],
                [80.0, 70.0, np.nan],
            ],
            index=champions,
            columns=champions,
            dtype=np.float64,
        )

    def _table_rows_by_label(self, table):
        lines = [line for line in table.splitlines() if line.startswith("|")]
        header_cells = [cell.strip() for cell in lines[0].split("|")[1:-1]]
        rows = {}
        for line in lines[1:]:
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            rows[cells[0]] = dict(zip(header_cells[1:], cells[1:]))
        return rows

    def test_champ_pick_uses_column_champ_winrate_against_row_champ(self):
        data = self._build_fixture()
        table = functions.analyze_champ_picks(
            data,
            current_pool=["B", "C"],
            input_champs=["A"],
        )
        rows = self._table_rows_by_label(table)

        self.assertEqual(rows["A"]["B"], "90.00% -> 55.00% (norm)")
        self.assertEqual(rows["A"]["C"], "80.00% -> 45.00% (norm)")
        self.assertEqual(rows["A"]["Best Raw Champ"], "B")
        self.assertEqual(rows["A"]["Best Norm Champ"], "B")

    def test_champ_pick_min_rows_match_column_against_row_orientation(self):
        data = self._build_fixture()
        table = functions.analyze_champ_picks(
            data,
            current_pool=["B", "C"],
            input_champs=["A"],
        )
        rows = self._table_rows_by_label(table)

        self.assertEqual(rows["Raw Min"]["B"], "30.00%")
        self.assertEqual(rows["Raw Min"]["C"], "70.00%")
        self.assertEqual(rows["Norm Min"]["B"], "55.00%")
        self.assertEqual(rows["Norm Min"]["C"], "45.00%")


if __name__ == "__main__":
    unittest.main()

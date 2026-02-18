import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import re
import numpy as np
import warnings

CHAMPION_BASELINE_ATTR = "champion_baseline_winrate"
OPPONENT_PICKRATE_ATTR = "opponent_role_pickrate"

def print_break():
    """Print a standardized separator for console output."""
    print("---------------------------------------------------------------")

# Configuration Management
def load_config(config_file_path):
    """Load configuration from the specified JSON file."""
    with open(config_file_path, "r") as config_file:
        return json.load(config_file)

def save_config(config, config_file_path):
    """Save configuration back to the specified JSON file."""
    with open(config_file_path, "w") as config_file:
        json.dump(config, config_file, indent=4)

# Data Loading and Normalization
def load_data(file_path):
    """Load data from a TSV file and handle missing or invalid entries."""
    data = pd.read_csv(os.path.join(os.getcwd(), file_path), sep='\t', index_col=0)
    data.replace("/", pd.NA, inplace=True)
    numeric_data = data.apply(pd.to_numeric)
    baseline_series = load_source_page_winrates(file_path, numeric_data.columns)
    if baseline_series is not None:
        numeric_data.attrs[CHAMPION_BASELINE_ATTR] = baseline_series
    return numeric_data

def load_matchup_games_data(file_path, matrix_index, matrix_columns):
    """Load matchup game counts from a sibling CSV and align to matrix labels."""
    csv_file_path = os.path.splitext(os.path.join(os.getcwd(), file_path))[0] + ".csv"
    if not os.path.exists(csv_file_path):
        return None

    raw_games_data = pd.read_csv(csv_file_path)
    required_columns = {"source_champion", "opponent_champion", "games"}
    if not required_columns.issubset(raw_games_data.columns):
        return None

    source_mapping = create_name_mapping(list(matrix_index))
    opponent_mapping = create_name_mapping(list(matrix_columns))

    games_matrix = pd.DataFrame(np.nan, index=matrix_index, columns=matrix_columns, dtype=float)
    opponent_game_totals = pd.Series(0.0, index=matrix_columns, dtype=float)
    total_games = 0.0
    raw_games_data["games"] = pd.to_numeric(raw_games_data["games"], errors="coerce")
    raw_games_data = raw_games_data.dropna(subset=["games"])

    for _, row in raw_games_data.iterrows():
        source_normalized = normalize_champ_name(str(row["source_champion"]))
        opponent_normalized = normalize_champ_name(str(row["opponent_champion"]))
        source_champ = source_mapping.get(source_normalized)
        opponent_champ = opponent_mapping.get(opponent_normalized)

        if source_champ is None or opponent_champ is None:
            continue

        games_value = float(row["games"])
        existing_games_value = games_matrix.loc[source_champ, opponent_champ]
        if pd.isna(existing_games_value):
            games_matrix.loc[source_champ, opponent_champ] = games_value
        else:
            games_matrix.loc[source_champ, opponent_champ] = float(existing_games_value) + games_value
        opponent_game_totals.loc[opponent_champ] += games_value
        total_games += games_value

    if total_games > 0:
        games_matrix.attrs[OPPONENT_PICKRATE_ATTR] = (opponent_game_totals / total_games).reindex(matrix_columns)

    return games_matrix

def _load_matchups_raw_data(file_path):
    """Load raw matchup rows from a sibling Excel file (preferred) or CSV fallback."""
    base_path = os.path.splitext(os.path.join(os.getcwd(), file_path))[0]
    excel_file_path = base_path + ".xlsx"
    csv_file_path = base_path + ".csv"

    if os.path.exists(excel_file_path):
        try:
            return pd.read_excel(excel_file_path, sheet_name="Matchups_Raw")
        except ValueError:
            # Some exports may only contain one sheet or a different name.
            try:
                return pd.read_excel(excel_file_path)
            except Exception:
                pass
        except Exception:
            pass

    if os.path.exists(csv_file_path):
        return pd.read_csv(csv_file_path)

    return None

def _coerce_winrate_to_percent(values):
    """Convert win-rate values to percentages, handling [0, 1] and [0, 100] inputs."""
    series = pd.to_numeric(values, errors="coerce")
    if not series.notna().any():
        return series

    finite = series.dropna()
    if finite.max(skipna=True) <= 1.0:
        series = series * 100.0
    elif (
        finite.min(skipna=True) >= 0.0
        and finite.max(skipna=True) <= 100.0
        and (finite <= 1.0).any()
        and (finite > 1.0).any()
    ):
        # Handle mixed scales by promoting fractional entries to percentage space.
        series = series.where(series.isna() | (series > 1.0), series * 100.0)
    return series

def _derive_champion_baselines_from_raw_data(raw_data, matrix_columns):
    """
    Derive champion-level baselines aligned to matrix columns.

    Priority:
    1. Source-page champion win-rate columns, when present.
    2. Procedural baseline from bulk matchup rows using source champion matchup win rates.
       - Weighted mean by games when positive games exist.
       - Unweighted mean otherwise.

    If source-page win rates are available but incomplete, missing champions are
    backfilled procedurally from matchup rows.
    """
    if raw_data is None or "source_champion" not in raw_data.columns:
        return None

    column_mapping = create_name_mapping(list(matrix_columns))
    baseline_candidates = [
        "source_page_win_rate",
        "source_win_rate",
        "source_champion_win_rate",
        "overall_win_rate",
        "overall_winrate",
    ]
    baseline_column = next((candidate for candidate in baseline_candidates if candidate in raw_data.columns), None)

    source_page_baselines = None
    if baseline_column is not None:
        source_baselines = _coerce_winrate_to_percent(raw_data[baseline_column])
        baseline_rows = raw_data.assign(_baseline=source_baselines).dropna(subset=["_baseline"])
        baseline_rows = baseline_rows.copy()
        baseline_rows["_source_norm"] = baseline_rows["source_champion"].map(
            lambda value: normalize_champ_name(str(value))
        )
        if "patch" in baseline_rows.columns:
            baseline_rows["_patch_group"] = baseline_rows["patch"].fillna("").astype(str)
            baseline_rows = (
                baseline_rows.groupby(["_source_norm", "_patch_group"], as_index=False)["_baseline"].mean()
            )
        else:
            baseline_rows = baseline_rows.groupby(["_source_norm"], as_index=False)["_baseline"].mean()
        grouped_values = {}

        for _, row in baseline_rows.iterrows():
            source_normalized = str(row["_source_norm"])
            source_champ = column_mapping.get(source_normalized)
            if source_champ is None:
                continue
            grouped_values.setdefault(source_champ, []).append(float(row["_baseline"]))

        if grouped_values:
            source_page_baselines = pd.Series(
                {champ: float(np.mean(values)) for champ, values in grouped_values.items()},
                dtype=float,
            ).reindex(matrix_columns)

    if source_page_baselines is not None and not source_page_baselines.isna().any():
        return source_page_baselines

    procedural_baselines = None
    if "win_rate" in raw_data.columns:
        matchup_winrates = _coerce_winrate_to_percent(raw_data["win_rate"])
        if "games" in raw_data.columns:
            matchup_games = pd.to_numeric(raw_data["games"], errors="coerce")
        else:
            matchup_games = pd.Series(np.nan, index=raw_data.index, dtype=float)

        matchup_rows = raw_data.assign(_wr=matchup_winrates, _games=matchup_games).dropna(subset=["_wr"])
        if not matchup_rows.empty:
            grouped_values = {}
            grouped_weights = {}
            for _, row in matchup_rows.iterrows():
                source_normalized = normalize_champ_name(str(row["source_champion"]))
                source_champ = column_mapping.get(source_normalized)
                if source_champ is None:
                    continue

                wr_value = float(row["_wr"])
                game_value = float(row["_games"]) if pd.notna(row["_games"]) else np.nan
                grouped_values.setdefault(source_champ, []).append(wr_value)
                grouped_weights.setdefault(source_champ, []).append(game_value)

            if grouped_values:
                baselines = {}
                for champ, values in grouped_values.items():
                    value_array = np.asarray(values, dtype=np.float64)
                    weight_array = np.asarray(grouped_weights.get(champ, []), dtype=np.float64)
                    valid_weights = np.isfinite(weight_array) & (weight_array > 0)
                    if np.any(valid_weights):
                        baselines[champ] = float(np.average(value_array[valid_weights], weights=weight_array[valid_weights]))
                    else:
                        baselines[champ] = float(np.mean(value_array))

                procedural_baselines = pd.Series(baselines, dtype=float).reindex(matrix_columns)

    if source_page_baselines is not None:
        if procedural_baselines is not None:
            return source_page_baselines.fillna(procedural_baselines)
        return source_page_baselines

    return procedural_baselines

def load_source_page_winrates(file_path, matrix_columns):
    """
    Load champion baseline win rates aligned to matrix columns.

    Uses source-page champion win-rate columns when available, otherwise derives
    baselines procedurally from bulk matchup win-rate rows.
    """
    raw_data = _load_matchups_raw_data(file_path)
    return _derive_champion_baselines_from_raw_data(raw_data, matrix_columns)

def normalize_champ_name(name):
    """Normalize champion names for consistent matching."""
    import unicodedata
    # Strip accents, remove special characters, and lowercase
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
    name = re.sub(r"[^\w\s]", "", name)  # Remove punctuation
    name = re.sub(r"\s+", " ", name)  # Normalize spaces
    return name.strip().lower()

def create_name_mapping(columns):
    """Create a mapping from normalized names to original column names."""
    mapping = {normalize_champ_name(col): col for col in columns}
    if len(mapping) != len(columns):
        print("Warning: Some champion names may have duplicate normalized forms!")
    return mapping

# Data Analysis
def calculate_champ_std_devs(data):
    """Calculate the standard deviation of winrates for each champion."""
    return data.std(axis=0).sort_values(ascending=False)

def calculate_champ_mean_corrs(data):
    """Calculate the mean correlation of each champion with others."""
    return data.corr().mean(axis=0).sort_values(ascending=False)

def _ensure_float64_frame(data, frame_name):
    """Return a float64 DataFrame with deterministic labels."""
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"{frame_name} must be a pandas DataFrame.")
    numeric = data.apply(pd.to_numeric, errors="coerce").astype(np.float64)
    numeric.attrs = dict(getattr(data, "attrs", {}))
    if numeric.index.has_duplicates:
        raise ValueError(f"{frame_name} index contains duplicate champion labels.")
    if numeric.columns.has_duplicates:
        raise ValueError(f"{frame_name} columns contain duplicate champion labels.")
    return numeric

def _assert_aligned_matrix(candidate, reference, candidate_name):
    """Assert a matrix matches the reference matchup matrix exactly."""
    if candidate.shape != reference.shape:
        raise ValueError(
            f"{candidate_name} shape {candidate.shape} does not match matchup data shape {reference.shape}."
        )
    if not candidate.index.equals(reference.index):
        raise ValueError(f"{candidate_name} index does not align to matchup data index.")
    if not candidate.columns.equals(reference.columns):
        raise ValueError(f"{candidate_name} columns do not align to matchup data columns.")

def _resolve_baseline_by_champion(data, baseline):
    """Normalize scalar/series/dict baselines into a per-row baseline series."""
    if baseline is None:
        baseline = data.attrs.get(CHAMPION_BASELINE_ATTR)
        if baseline is None:
            raise ValueError(
                "Baseline win rates are required for error metrics. "
                f"Pass `baseline`, or attach `{CHAMPION_BASELINE_ATTR}` to the input data."
            )

    if np.isscalar(baseline):
        baseline_series = pd.Series(float(baseline), index=data.index, dtype=np.float64)
    elif isinstance(baseline, dict):
        baseline_series = pd.Series(baseline, dtype=np.float64).reindex(data.index)
    elif isinstance(baseline, pd.Series):
        baseline_series = pd.to_numeric(baseline, errors="coerce").astype(np.float64).reindex(data.index)
    else:
        raise ValueError("Baseline must be a scalar, dict, pandas Series, or None.")

    missing = baseline_series[baseline_series.isna()].index.tolist()
    if missing:
        missing_preview = ", ".join(map(str, missing[:5]))
        if len(missing) > 5:
            missing_preview += ", ..."
        raise ValueError(
            "Baseline is missing for one or more champions after alignment: "
            f"{missing_preview}."
        )

    return baseline_series

def _build_matchup_valid_mask(data):
    """Build a shared validity mask for matchup win rates used by all error metrics."""
    valid_mask = data.notna()
    shared_labels = data.index.intersection(data.columns)
    for label in shared_labels:
        valid_mask.loc[label, label] = False
    return valid_mask

def _build_games_proxy_weights(data, games_data):
    """Build nonnegative weights w_ij aligned to (champion i row, matchup j column)."""
    if games_data is None:
        raise ValueError("Matchup game counts are unavailable for games-weighted metrics.")

    aligned_games = _ensure_float64_frame(games_data, "games_data").reindex(index=data.index, columns=data.columns)
    _assert_aligned_matrix(aligned_games, data, "games_data")
    weights = aligned_games.where(aligned_games > 0)
    if weights.notna().sum().sum() == 0:
        raise ValueError("No positive games-based matchup weights were found for weighted metrics.")
    _assert_aligned_matrix(weights, data, "weight matrix")
    return weights

def _warn_weight_policy(metric_name, message, champions):
    """Emit a deterministic warning describing rows that cannot use weighted math."""
    if not champions:
        return
    sorted_champs = ", ".join(sorted(map(str, champions)))
    warnings.warn(
        f"{metric_name}: {message}: {sorted_champs}. Returning NaN for those champions.",
        RuntimeWarning,
        stacklevel=3,
    )

def _equalize_weights_by_row(weights, metric_name):
    """Equalize by champion row: w_hat_ij = w_ij / mean_i(w_ij over valid matchups)."""
    row_mean_weights = weights.mean(axis=1, skipna=True)
    valid_row_means = row_mean_weights.where(row_mean_weights > 0)
    invalid_rows = list(valid_row_means[valid_row_means.isna()].index)
    _warn_weight_policy(
        metric_name,
        "unable to row-equalize because mean weight is missing or nonpositive",
        invalid_rows,
    )
    return weights.div(valid_row_means, axis=0)

def _reduce_unweighted_errors(errors, reduction):
    """Reduce error magnitudes by champion row."""
    if reduction == "mean":
        return errors.mean(axis=1, skipna=True)
    if reduction == "median":
        return errors.median(axis=1, skipna=True)
    raise ValueError(f"Unsupported reduction '{reduction}'. Use 'mean' or 'median'.")

def _weighted_row_median(values, weights):
    """Compute the weighted median for one champion row."""
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan

    row_values = values[valid]
    row_weights = weights[valid]
    sort_idx = np.argsort(row_values, kind="mergesort")
    row_values = row_values[sort_idx]
    row_weights = row_weights[sort_idx]
    cumulative = np.cumsum(row_weights)
    cutoff = 0.5 * row_weights.sum()
    position = int(np.searchsorted(cumulative, cutoff, side="left"))
    return float(row_values[min(position, len(row_values) - 1)])

def _reduce_weighted_errors(errors, weights, reduction, metric_name):
    """Reduce weighted errors by champion row with shared masking and NaN policy."""
    _assert_aligned_matrix(weights, errors, "weight matrix")
    if reduction == "mean":
        numerator = (errors * weights).sum(axis=1, skipna=True, min_count=1)
        denominator = weights.sum(axis=1, skipna=True, min_count=1)
        reduced = numerator / denominator
        reduced = reduced.where(denominator > 0)
    elif reduction == "median":
        reduced_values = [
            _weighted_row_median(
                errors.loc[champ].to_numpy(dtype=np.float64),
                weights.loc[champ].to_numpy(dtype=np.float64),
            )
            for champ in errors.index
        ]
        reduced = pd.Series(reduced_values, index=errors.index, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported reduction '{reduction}'. Use 'mean' or 'median'.")

    missing_rows = list(reduced[reduced.isna()].index)
    _warn_weight_policy(
        metric_name,
        "no valid positive matchup weights after masking",
        missing_rows,
    )
    return reduced

def _calculate_champion_error_metric(
    data,
    baseline,
    error_kind,
    reduction,
    weighted=False,
    games_data=None,
    equalize=False,
    metric_name="metric",
):
    """Shared source-of-truth pipeline for all analyze-champ error metrics."""
    if equalize and not weighted:
        raise ValueError("Equalization can only be enabled for weighted metrics.")

    matchup_data = _ensure_float64_frame(data, "data")
    baseline_series = _resolve_baseline_by_champion(matchup_data, baseline)
    valid_matchup_mask = _build_matchup_valid_mask(matchup_data)

    raw_errors = matchup_data.sub(baseline_series, axis=0)
    if error_kind == "absolute":
        errors = raw_errors.abs()
    elif error_kind == "squared":
        errors = raw_errors.pow(2)
    else:
        raise ValueError(f"Unsupported error kind '{error_kind}'. Use 'absolute' or 'squared'.")

    errors = errors.where(valid_matchup_mask)
    if weighted:
        weights = _build_games_proxy_weights(matchup_data, games_data)
        weights = weights.where(valid_matchup_mask)
        if equalize:
            weights = _equalize_weights_by_row(weights, metric_name)
        reduced = _reduce_weighted_errors(errors, weights, reduction, metric_name)
    else:
        reduced = _reduce_unweighted_errors(errors, reduction)

    if error_kind == "squared":
        reduced = np.sqrt(reduced)
    return reduced.sort_values(ascending=False, kind="mergesort")

def calculate_champ_rmse(data, baseline=None):
    """Calculate RMSE_i = sqrt(mean_j((wr_ij - b_i)^2)) over valid matchups."""
    return _calculate_champion_error_metric(
        data,
        baseline=baseline,
        error_kind="squared",
        reduction="mean",
        metric_name="rmse",
    )

def calculate_champ_rmedse(data, baseline=None):
    """Calculate RMEDSE_i = sqrt(median_j((wr_ij - b_i)^2)) over valid matchups."""
    return _calculate_champion_error_metric(
        data,
        baseline=baseline,
        error_kind="squared",
        reduction="median",
        metric_name="rmedse",
    )

def calculate_champ_mae(data, baseline=None):
    """Calculate MAE_i = mean_j(|wr_ij - b_i|) over valid matchups."""
    return _calculate_champion_error_metric(
        data,
        baseline=baseline,
        error_kind="absolute",
        reduction="mean",
        metric_name="mae",
    )

def calculate_champ_medae(data, baseline=None):
    """Calculate MedAE_i = median_j(|wr_ij - b_i|) over valid matchups."""
    return _calculate_champion_error_metric(
        data,
        baseline=baseline,
        error_kind="absolute",
        reduction="median",
        metric_name="medae",
    )

def calculate_champ_rmse_games_weighed(data, games_data, baseline=None):
    """Calculate games-weighted RMSE using positive matchup-likelihood proxy weights."""
    return _calculate_champion_error_metric(
        data,
        baseline=baseline,
        error_kind="squared",
        reduction="mean",
        weighted=True,
        games_data=games_data,
        metric_name="rmse_games_weighed",
    )

def calculate_champ_rmedse_games_weighed(data, games_data, baseline=None):
    """Calculate games-weighted RMEDSE using weighted medians of squared errors."""
    return _calculate_champion_error_metric(
        data,
        baseline=baseline,
        error_kind="squared",
        reduction="median",
        weighted=True,
        games_data=games_data,
        metric_name="rmedse_games_weighed",
    )

def calculate_champ_mae_games_weighed(data, games_data, baseline=None):
    """Calculate WMAE_i = sum_j(w_ij*|wr_ij-b_i|)/sum_j(w_ij) over valid matchups."""
    return _calculate_champion_error_metric(
        data,
        baseline=baseline,
        error_kind="absolute",
        reduction="mean",
        weighted=True,
        games_data=games_data,
        metric_name="mae_games_weighed",
    )

def calculate_champ_medae_games_weighed(data, games_data, baseline=None):
    """Calculate games-weighted MedAE using weighted medians of absolute errors."""
    return _calculate_champion_error_metric(
        data,
        baseline=baseline,
        error_kind="absolute",
        reduction="median",
        weighted=True,
        games_data=games_data,
        metric_name="medae_games_weighed",
    )

def calculate_champ_rmse_games_weighed_equalized(data, games_data, baseline=None):
    """Calculate games-weighted RMSE with row-wise champion equalization."""
    return _calculate_champion_error_metric(
        data,
        baseline=baseline,
        error_kind="squared",
        reduction="mean",
        weighted=True,
        games_data=games_data,
        equalize=True,
        metric_name="rmse_games_weighed_equalized",
    )

def calculate_champ_rmedse_games_weighed_equalized(data, games_data, baseline=None):
    """Calculate games-weighted RMEDSE with row-wise champion equalization."""
    return _calculate_champion_error_metric(
        data,
        baseline=baseline,
        error_kind="squared",
        reduction="median",
        weighted=True,
        games_data=games_data,
        equalize=True,
        metric_name="rmedse_games_weighed_equalized",
    )

def calculate_champ_mae_games_weighed_equalized(data, games_data, baseline=None):
    """Calculate MAE_GAMES_EQUALIZED_i with row-wise equalization of w_ij before reduction."""
    return _calculate_champion_error_metric(
        data,
        baseline=baseline,
        error_kind="absolute",
        reduction="mean",
        weighted=True,
        games_data=games_data,
        equalize=True,
        metric_name="mae_games_weighed_equalized",
    )

def calculate_champ_medae_games_weighed_equalized(data, games_data, baseline=None):
    """Calculate games-weighted MedAE with row-wise champion equalization."""
    return _calculate_champion_error_metric(
        data,
        baseline=baseline,
        error_kind="absolute",
        reduction="median",
        weighted=True,
        games_data=games_data,
        equalize=True,
        metric_name="medae_games_weighed_equalized",
    )

def calculate_champ_rmse_norm_games(data, games_data, baseline=None):
    """Backward-compatible alias for calculate_champ_rmse_games_weighed."""
    return calculate_champ_rmse_games_weighed(data, games_data, baseline=baseline)

def calculate_champ_rmedse_norm_games(data, games_data, baseline=None):
    """Backward-compatible alias for calculate_champ_rmedse_games_weighed."""
    return calculate_champ_rmedse_games_weighed(data, games_data, baseline=baseline)

def calculate_champ_mae_norm_games(data, games_data, baseline=None):
    """Backward-compatible alias for calculate_champ_mae_games_weighed."""
    return calculate_champ_mae_games_weighed(data, games_data, baseline=baseline)

def calculate_champ_medae_norm_games(data, games_data, baseline=None):
    """Backward-compatible alias for calculate_champ_medae_games_weighed."""
    return calculate_champ_medae_games_weighed(data, games_data, baseline=baseline)

def calculate_champ_rmse_norm_games_equalized(data, games_data, baseline=None):
    """Alias for calculate_champ_rmse_games_weighed_equalized."""
    return calculate_champ_rmse_games_weighed_equalized(data, games_data, baseline=baseline)

def calculate_champ_rmedse_norm_games_equalized(data, games_data, baseline=None):
    """Alias for calculate_champ_rmedse_games_weighed_equalized."""
    return calculate_champ_rmedse_games_weighed_equalized(data, games_data, baseline=baseline)

def calculate_champ_mae_norm_games_equalized(data, games_data, baseline=None):
    """Alias for calculate_champ_mae_games_weighed_equalized."""
    return calculate_champ_mae_games_weighed_equalized(data, games_data, baseline=baseline)

def calculate_champ_medae_norm_games_equalized(data, games_data, baseline=None):
    """Alias for calculate_champ_medae_games_weighed_equalized."""
    return calculate_champ_medae_games_weighed_equalized(data, games_data, baseline=baseline)

def normalize_data(data):
    """Normalize win rates to have a mean of 50 and preserve relative variability."""
    normalized_data = data.copy()
    col_means = data.mean()

    normalized_data = (data - col_means) + 50

    return normalized_data

def _calculate_matchup_ci_by_champion(data, confidence_level=0.95):
    """Return matchup-based CI bounds per champion row as {champ: (low, high)}."""
    matchup_data = _ensure_float64_frame(data, "data")
    valid_matchup_mask = _build_matchup_valid_mask(matchup_data)

    ci_by_champion = {}
    for champion in matchup_data.index:
        row_values = matchup_data.loc[champion].where(valid_matchup_mask.loc[champion])
        _, ci_low, ci_high = _calculate_distribution_confidence_interval(
            row_values,
            confidence_level=confidence_level,
        )
        ci_by_champion[champion] = (ci_low, ci_high)
    return ci_by_champion

def _print_ranked_metric_with_ci(metric_scores, data, value_suffix="", confidence_level=0.95):
    """Print ranked metric lines with matchup-based confidence interval suffix."""
    ci_by_champion = _calculate_matchup_ci_by_champion(data, confidence_level=confidence_level)
    formatted_rows = []

    for i, (champ, score) in enumerate(metric_scores.items(), start=1):
        if pd.isna(score):
            score_text = "/"
        else:
            score_text = f"{score:.2f}{value_suffix}"

        ci_low, ci_high = ci_by_champion.get(champ, (np.nan, np.nan))
        if np.isfinite(ci_low) and np.isfinite(ci_high):
            ci_text = f"({ci_low:.2f}% - {ci_high:.2f}%)"
        else:
            ci_text = "(/ - /)"

        left_text = f"{i}. {champ.title()}: {score_text}"
        formatted_rows.append((left_text, ci_text))

    if not formatted_rows:
        return

    ci_column = max(len(left_text) for left_text, _ in formatted_rows) + 6
    for left_text, ci_text in formatted_rows:
        print(f"{left_text.ljust(ci_column)}{ci_text}")

def visualize_std_devs(data):
    """Display champions sorted by the standard deviation of their winrates."""
    champ_std_devs = calculate_champ_std_devs(data)
    print("Champs sorted by average winrate standard deviation:")
    _print_ranked_metric_with_ci(champ_std_devs, data, value_suffix="%")

def visualize_mean_corrs(data):
    """Display champions sorted by their mean correlation with others."""
    champ_means = calculate_champ_mean_corrs(data)
    print("Champs sorted by average correlation with others in terms of which champs they counter:")
    _print_ranked_metric_with_ci(champ_means, data)

def visualize_rmse(data, baseline=None):
    """Display champions sorted by RMSE of matchup winrates from baseline."""
    champ_rmses = calculate_champ_rmse(data, baseline=baseline)
    print("Champs sorted by matchup RMSE from baseline winrate:")
    _print_ranked_metric_with_ci(champ_rmses, data, value_suffix="%")

def visualize_rmedse(data, baseline=None):
    """Display champions sorted by RMEDSE of matchup winrates from baseline."""
    champ_rmedses = calculate_champ_rmedse(data, baseline=baseline)
    print("Champs sorted by matchup RMEDSE from baseline winrate:")
    _print_ranked_metric_with_ci(champ_rmedses, data, value_suffix="%")

def visualize_mae(data, baseline=None):
    """Display champions sorted by MAE of matchup winrates from baseline."""
    champ_maes = calculate_champ_mae(data, baseline=baseline)
    print("Champs sorted by matchup MAE from baseline winrate:")
    _print_ranked_metric_with_ci(champ_maes, data, value_suffix="%")

def visualize_medae(data, baseline=None):
    """Display champions sorted by MedAE of matchup winrates from baseline."""
    champ_medaes = calculate_champ_medae(data, baseline=baseline)
    print("Champs sorted by matchup MedAE from baseline winrate:")
    _print_ranked_metric_with_ci(champ_medaes, data, value_suffix="%")

def visualize_rmse_games_weighed(data, games_data, baseline=None):
    """Display champions sorted by games-weighted RMSE."""
    champ_rmses_games_weighed = calculate_champ_rmse_games_weighed(data, games_data, baseline=baseline)
    print("Champs sorted by matchup RMSE (games-weighted):")
    _print_ranked_metric_with_ci(champ_rmses_games_weighed, data, value_suffix="%")

def visualize_rmedse_games_weighed(data, games_data, baseline=None):
    """Display champions sorted by games-weighted RMEDSE."""
    champ_rmedses_games_weighed = calculate_champ_rmedse_games_weighed(data, games_data, baseline=baseline)
    print("Champs sorted by matchup RMEDSE (games-weighted):")
    _print_ranked_metric_with_ci(champ_rmedses_games_weighed, data, value_suffix="%")

def visualize_mae_games_weighed(data, games_data, baseline=None):
    """Display champions sorted by games-weighted MAE."""
    champ_maes_games_weighed = calculate_champ_mae_games_weighed(data, games_data, baseline=baseline)
    print("Champs sorted by matchup MAE (games-weighted):")
    _print_ranked_metric_with_ci(champ_maes_games_weighed, data, value_suffix="%")

def visualize_medae_games_weighed(data, games_data, baseline=None):
    """Display champions sorted by games-weighted MedAE."""
    champ_medaes_games_weighed = calculate_champ_medae_games_weighed(data, games_data, baseline=baseline)
    print("Champs sorted by matchup MedAE (games-weighted):")
    _print_ranked_metric_with_ci(champ_medaes_games_weighed, data, value_suffix="%")

def visualize_rmse_games_weighed_equalized(data, games_data, baseline=None):
    """Display games-weighted RMSE with row-wise equalization by champion."""
    champ_scores = calculate_champ_rmse_games_weighed_equalized(data, games_data, baseline=baseline)
    print("Champs sorted by matchup RMSE (games-weighted, row-equalized):")
    _print_ranked_metric_with_ci(champ_scores, data, value_suffix="%")

def visualize_rmedse_games_weighed_equalized(data, games_data, baseline=None):
    """Display games-weighted RMEDSE with row-wise equalization by champion."""
    champ_scores = calculate_champ_rmedse_games_weighed_equalized(data, games_data, baseline=baseline)
    print("Champs sorted by matchup RMEDSE (games-weighted, row-equalized):")
    _print_ranked_metric_with_ci(champ_scores, data, value_suffix="%")

def visualize_mae_games_weighed_equalized(data, games_data, baseline=None):
    """Display games-weighted MAE with row-wise equalization by champion."""
    champ_scores = calculate_champ_mae_games_weighed_equalized(data, games_data, baseline=baseline)
    print("Champs sorted by matchup MAE (games-weighted, row-equalized):")
    _print_ranked_metric_with_ci(champ_scores, data, value_suffix="%")

def visualize_medae_games_weighed_equalized(data, games_data, baseline=None):
    """Display games-weighted MedAE with row-wise equalization by champion."""
    champ_scores = calculate_champ_medae_games_weighed_equalized(data, games_data, baseline=baseline)
    print("Champs sorted by matchup MedAE (games-weighted, row-equalized):")
    _print_ranked_metric_with_ci(champ_scores, data, value_suffix="%")

def visualize_rmse_norm_games(data, games_data, baseline=None):
    """Backward-compatible alias for visualize_rmse_games_weighed."""
    visualize_rmse_games_weighed(data, games_data, baseline=baseline)

def visualize_rmedse_norm_games(data, games_data, baseline=None):
    """Backward-compatible alias for visualize_rmedse_games_weighed."""
    visualize_rmedse_games_weighed(data, games_data, baseline=baseline)

def visualize_mae_norm_games(data, games_data, baseline=None):
    """Backward-compatible alias for visualize_mae_games_weighed."""
    visualize_mae_games_weighed(data, games_data, baseline=baseline)

def visualize_medae_norm_games(data, games_data, baseline=None):
    """Backward-compatible alias for visualize_medae_games_weighed."""
    visualize_medae_games_weighed(data, games_data, baseline=baseline)

def visualize_rmse_norm_games_equalized(data, games_data, baseline=None):
    """Alias for visualize_rmse_games_weighed_equalized."""
    visualize_rmse_games_weighed_equalized(data, games_data, baseline=baseline)

def visualize_rmedse_norm_games_equalized(data, games_data, baseline=None):
    """Alias for visualize_rmedse_games_weighed_equalized."""
    visualize_rmedse_games_weighed_equalized(data, games_data, baseline=baseline)

def visualize_mae_norm_games_equalized(data, games_data, baseline=None):
    """Alias for visualize_mae_games_weighed_equalized."""
    visualize_mae_games_weighed_equalized(data, games_data, baseline=baseline)

def visualize_medae_norm_games_equalized(data, games_data, baseline=None):
    """Alias for visualize_medae_games_weighed_equalized."""
    visualize_medae_games_weighed_equalized(data, games_data, baseline=baseline)

# Pool Analysis
def calculate_pool_wr_stats(data, current_pool):
    """Calculate the maximum average winrate for the current champion pool."""
    if current_pool:
        pool_data = data[current_pool]
        return pool_data.max(axis=1).mean()
    return None

def _calculate_distribution_confidence_interval(values, confidence_level=0.95):
    """Calculate a two-sided empirical confidence interval from matchup values."""
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1 (exclusive).")

    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return np.nan, np.nan, np.nan

    alpha = 1.0 - confidence_level
    lower_quantile = alpha / 2.0
    upper_quantile = 1.0 - lower_quantile

    low = float(series.quantile(lower_quantile))
    high = float(series.quantile(upper_quantile))
    midpoint = (low + high) / 2.0
    return midpoint, low, high

def calculate_best_champ_additions_with_ci(data, current_pool, confidence_level=0.95):
    """Identify champion additions and include matchup-based confidence intervals."""
    name_mapping = create_name_mapping(data.columns)
    current_pool = [name_mapping.get(normalize_champ_name(champ), None) for champ in current_pool]
    current_pool = [champ for champ in current_pool if champ is not None]

    remaining_champs = [champ for champ in data.columns if champ not in current_pool]
    if not remaining_champs:
        return []

    champ_addition_impact = []
    for champ in remaining_champs:
        temp_pool = current_pool + [champ]
        temp_data = data[temp_pool]
        matchup_distribution = pd.to_numeric(temp_data.max(axis=1, skipna=True), errors="coerce")
        score = float(matchup_distribution.mean(skipna=True))
        ci_midpoint, ci_low, ci_high = _calculate_distribution_confidence_interval(
            matchup_distribution,
            confidence_level=confidence_level,
        )
        champ_addition_impact.append((champ, score, ci_low, ci_high, ci_midpoint))

    return sorted(champ_addition_impact, key=lambda x: x[1], reverse=True)

def calculate_best_champ_additions(data, current_pool):
    """Identify champions that would most improve the pool's max winrate."""
    additions_with_ci = calculate_best_champ_additions_with_ci(
        data,
        current_pool,
        confidence_level=0.95,
    )
    return [(champ, score) for champ, score, _, _, _ in additions_with_ci]

def calculate_worst_champ_removals(data, current_pool):
    """Identify champions whose removal would least impact the pool's max winrate."""
    champ_removal_impact = {}

    for champ in current_pool:
        temp_pool = [c for c in current_pool if c != champ]
        if not temp_pool:
            continue
        temp_data = data[temp_pool]
        champ_removal_impact[champ] = temp_data.max(axis=1).mean()

    return sorted(champ_removal_impact.items(), key=lambda x: x[1])

def validate_input_champs(data, input_champs, name_mapping):
    """Validate and normalize input champions."""
    normalized_input = [normalize_champ_name(champ) for champ in input_champs]
    valid_champs = [name_mapping.get(champ) for champ in normalized_input if champ in name_mapping]
    invalid_champs = [champ for champ, mapped in zip(input_champs, valid_champs) if mapped is None]
    return valid_champs, invalid_champs

def analyze_champ_picks(data, current_pool, input_champs):
    """Analyze matchups for the provided input champions against the current pool, excluding input champions as output options."""

    # Normalize the dataset
    normalized_data = normalize_data(data)

    # Read matchup values as: column champion's win rate versus row champion.
    def get_intersection_value(matrix, column_champ, row_champ):
        if column_champ not in matrix.index or row_champ not in matrix.columns:
            return np.nan
        return matrix.loc[column_champ, row_champ]

    def calculate_average(values):
        finite_values = [float(value) for value in values if pd.notna(value)]
        if not finite_values:
            return np.nan
        return float(np.mean(finite_values))

    def pick_best_champion(score_by_champion):
        finite_scores = {
            champion: float(score)
            for champion, score in score_by_champion.items()
            if pd.notna(score)
        }
        if not finite_scores:
            return "/"
        return max(finite_scores, key=finite_scores.get)

    def format_matchup_percent(raw_value, norm_value):
        if pd.isna(raw_value) or pd.isna(norm_value):
            return "/"
        return f"{float(raw_value):.2f}% -> {float(norm_value):.2f}% (norm)"

    def format_percent(value):
        if pd.isna(value):
            return "/"
        return f"{float(value):.2f}%"

    # Ensure input champions exist in the dataset as row labels.
    valid_input_champs = [champ for champ in input_champs if champ in data.columns]

    # Warn if no valid input champions are found
    if not valid_input_champs:
        raise ValueError("None of the provided input champions exist in the dataset.")

    # Exclude input champions from output options and keep champions available as column labels.
    filtered_pool = [
        champ
        for champ in current_pool
        if champ not in input_champs and champ in data.index
    ]

    # Results storage
    results_raw = {}
    results_normalized = {}

    # Collect win rates (column champion vs each row champion).
    for pool_champ in filtered_pool:
        winrates_raw = [
            get_intersection_value(data, pool_champ, input_champ)
            for input_champ in valid_input_champs
        ]
        winrates_normalized = [
            get_intersection_value(normalized_data, pool_champ, input_champ)
            for input_champ in valid_input_champs
        ]
        results_raw[pool_champ] = winrates_raw
        results_normalized[pool_champ] = winrates_normalized

    # Calculate averages for normalized and non-normalized
    averages_raw = {
        champ: calculate_average(winrates)
        for champ, winrates in results_raw.items()
    }
    averages_normalized = {
        champ: calculate_average(winrates)
        for champ, winrates in results_normalized.items()
    }

    # Determine best champs
    best_raw = pick_best_champion(averages_raw)
    best_normalized = pick_best_champion(averages_normalized)

    champion_col_width = max(
        15,
        len("Champion"),
        len("Raw Min"),
        len("Norm Min"),
        max((len(champ.title()) for champ in valid_input_champs), default=0),
    )
    matchup_col_width = max(31, len("100.00% -> 100.00% (norm)"))
    best_col_width = max(
        20,
        len("Best Raw Champ"),
        len("Best Norm Champ"),
        max((len(champ) for champ in filtered_pool), default=0),
    )

    columns = [("Champion", champion_col_width)]
    columns.extend((champ, matchup_col_width) for champ in filtered_pool)
    columns.extend(
        [
            ("Best Raw Champ", best_col_width),
            ("Best Norm Champ", best_col_width),
        ]
    )

    def build_separator():
        return "+" + "+".join("-" * (width + 2) for _, width in columns) + "+\n"

    def build_row(values):
        return "|" + "|".join(
            f" {str(value):<{width}} "
            for value, (_, width) in zip(values, columns)
        ) + "|\n"

    # Build the output table
    table = build_separator()
    table += build_row([column_name for column_name, _ in columns])
    table += build_separator()

    for input_champ in valid_input_champs:
        raw_scores_for_input = {
            champ: get_intersection_value(data, champ, input_champ)
            for champ in filtered_pool
        }
        norm_scores_for_input = {
            champ: get_intersection_value(normalized_data, champ, input_champ)
            for champ in filtered_pool
        }
        best_non_norm = pick_best_champion(raw_scores_for_input)
        best_norm = pick_best_champion(norm_scores_for_input)

        row_values = [input_champ.title()]
        for champ in filtered_pool:
            row_values.append(
                format_matchup_percent(
                    raw_scores_for_input.get(champ, np.nan),
                    norm_scores_for_input.get(champ, np.nan),
                )
            )
        row_values.extend([best_non_norm, best_norm])
        table += build_row(row_values)

    # Add minimum non-normalized row
    min_row_values = ["Raw Min"]
    for champ in filtered_pool:
        min_val = data.loc[champ].min(skipna=True) if champ in data.index else np.nan
        min_row_values.append(format_percent(min_val))
    min_row_values.extend([best_raw, "/"])
    table += build_row(min_row_values)

    # Add minimum normalized row
    min_row_values = ["Norm Min"]
    for champ in filtered_pool:
        min_val = normalized_data.loc[champ].min(skipna=True) if champ in normalized_data.index else np.nan
        min_row_values.append(format_percent(min_val))
    min_row_values.extend(["/", best_normalized])
    table += build_row(min_row_values)

    table += build_separator()

    return table

# Visualization
def generate_heatmap(correlation_matrix, settings):
    """Generate a heatmap from the correlation matrix using provided settings."""
    figsize = settings.get("figsize", (10, 8))
    annot = settings.get("annot", True)
    cmap = settings.get("cmap", "coolwarm")
    linewidths = settings.get("linewidths", 0.5)
    fmt = settings.get("fmt", ".2f")
    title = settings.get("title", "Heatmap")
    textsize = settings.get("textsize", 10)

    correlation_matrix.index = [col.title() for col in correlation_matrix.index]
    correlation_matrix.columns = [col.title() for col in correlation_matrix.columns]

    if not correlation_matrix.empty and not correlation_matrix.isna().all().all():
        plt.figure(figsize=figsize)
        sns.heatmap(
            correlation_matrix,
            annot=annot,
            cmap=cmap,
            linewidths=linewidths,
            fmt=fmt,
            annot_kws={"size": textsize}
        )
        plt.title(title)
        plt.show()
    else:
        print("Correlation matrix is empty or contains only NaN values. Heatmap cannot be generated.")

# Champ selection
def select_champs(data, current_pool, prompt="Choose champion selection option"):
    """Prompt user to select champions to include or exclude for analysis."""
    print(prompt)
    print("1. All champions")
    print("2. Current pool")
    print("3. Only include specific champions")
    print("4. Exclude specific champions")

    option = input("Enter your choice (1-4): ").strip()

    if option == "1":
        return data
    elif option == "2":
        return data[current_pool]
    elif option == "3":
        champs_to_include = input("Enter champions to include, separated by commas: ").strip().split(",")
        champs_to_include = [normalize_champ_name(champ) for champ in champs_to_include]
        included_champs = [col for col in data.columns if normalize_champ_name(col) in champs_to_include]
        return data[included_champs]
    elif option == "4":
        champs_to_exclude = input("Enter champions to exclude, separated by commas: ").strip().split(",")
        champs_to_exclude = [normalize_champ_name(champ) for champ in champs_to_exclude]
        remaining_champs = [col for col in data.columns if normalize_champ_name(col) not in champs_to_exclude]
        return data[remaining_champs]
    else:
        raise ValueError("Invalid champion selection option. Expected 1, 2, 3, or 4.")

def generate_pool_stats(correlation_matrix, num_champs, top_pools, max_global_appearances):
    """Generate pools of champions with the lowest average correlation."""

    # Extract champion names from the correlation matrix
    champ_names = correlation_matrix.columns.tolist()

    # Generate all combinations of `num_champs`
    all_combinations = list(combinations(champ_names, num_champs))

    # Calculate the average correlation for each combination
    pool_scores = []
    for combo in all_combinations:
        # Extract the sub-matrix for the current combination
        sub_matrix = correlation_matrix.loc[combo, combo]

        # Calculate the average correlation (excluding self-correlations)
        avg_correlation = np.mean(sub_matrix.values[np.triu_indices(len(combo), k=1)])
        pool_scores.append((combo, avg_correlation))

    # Sort pools by lowest average correlation
    pool_scores = sorted(pool_scores, key=lambda x: x[1])

    # Enforce the global max appearances constraint
    champ_global_count = {champ: 0 for champ in champ_names}
    valid_pools = []

    for pool, score in pool_scores:
        if all(champ_global_count[champ] < max_global_appearances for champ in pool):
            valid_pools.append((pool, score))
            for champ in pool:
                champ_global_count[champ] += 1
        if len(valid_pools) >= top_pools:
            break

    # Clean up the output to make it user-friendly
    clean_pools = [
        (", ".join(pool), round(score, 4)) for pool, score in valid_pools
    ]

    return clean_pools

def calculate_best_bans(data, champ_or_pool, n_bans):
    """Calculate the best bans based on lowest win rates for a given champion or champion pool."""
    if isinstance(champ_or_pool, str):
        # Single champion
        if champ_or_pool not in data.columns:
            raise ValueError(f"Champion {champ_or_pool} not found in the dataset.")
        
        winrates = data[champ_or_pool]
        best_bans = winrates.nsmallest(n_bans)
    else:
        # Champion pool
        valid_champs = [champ for champ in champ_or_pool if champ in data.columns]
        if not valid_champs:
            raise ValueError("No valid champions from the pool found in the dataset.")

        # Calculate the average win rates against the pool
        avg_winrates = data[valid_champs].mean(axis=1)
        best_bans = avg_winrates.nsmallest(n_bans)

    return best_bans

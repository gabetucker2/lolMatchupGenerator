#!/usr/bin/env python3
"""Generate per-lane metric rankings (with matchup CI) into two text reports.

Default metric:
- mae_games_weighed_equalized

Outputs from one run:
- <stem>_normal.txt
- <stem>_inline.txt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import functions


LANE_FILES: Dict[str, str] = {
    "top": "matchups_top_expansive.tsv",
    "jungle": "matchups_jungle_expansive.tsv",
    "mid": "matchups_mid_expansive.tsv",
    "adc": "matchups_adc_expansive.tsv",
    "support": "matchups_support_expansive.tsv",
}

LANE_HEADERS: Dict[str, str] = {
    "top": "TOP LANE",
    "jungle": "JUNGLE",
    "mid": "MID LANE",
    "adc": "ADC",
    "support": "SUPPORT",
}

LANE_ORDER = ("top", "jungle", "mid", "adc", "support")

MetricFn = Callable[..., pd.Series]

METRIC_SPECS: Dict[str, Dict[str, object]] = {
    "std_devs": {"fn": functions.calculate_champ_std_devs, "requires_games": False, "requires_baseline": False},
    "mean_corrs": {"fn": functions.calculate_champ_mean_corrs, "requires_games": False, "requires_baseline": False},
    "rmse": {"fn": functions.calculate_champ_rmse, "requires_games": False, "requires_baseline": True},
    "rmedse": {"fn": functions.calculate_champ_rmedse, "requires_games": False, "requires_baseline": True},
    "mae": {"fn": functions.calculate_champ_mae, "requires_games": False, "requires_baseline": True},
    "medae": {"fn": functions.calculate_champ_medae, "requires_games": False, "requires_baseline": True},
    "rmse_games_weighed": {
        "fn": functions.calculate_champ_rmse_games_weighed,
        "requires_games": True,
        "requires_baseline": True,
    },
    "rmedse_games_weighed": {
        "fn": functions.calculate_champ_rmedse_games_weighed,
        "requires_games": True,
        "requires_baseline": True,
    },
    "mae_games_weighed": {
        "fn": functions.calculate_champ_mae_games_weighed,
        "requires_games": True,
        "requires_baseline": True,
    },
    "medae_games_weighed": {
        "fn": functions.calculate_champ_medae_games_weighed,
        "requires_games": True,
        "requires_baseline": True,
    },
    "rmse_games_weighed_equalized": {
        "fn": functions.calculate_champ_rmse_games_weighed_equalized,
        "requires_games": True,
        "requires_baseline": True,
    },
    "rmedse_games_weighed_equalized": {
        "fn": functions.calculate_champ_rmedse_games_weighed_equalized,
        "requires_games": True,
        "requires_baseline": True,
    },
    "mae_games_weighed_equalized": {
        "fn": functions.calculate_champ_mae_games_weighed_equalized,
        "requires_games": True,
        "requires_baseline": True,
    },
    "medae_games_weighed_equalized": {
        "fn": functions.calculate_champ_medae_games_weighed_equalized,
        "requires_games": True,
        "requires_baseline": True,
    },
}


def _format_champion_name(name: str) -> str:
    """Mirror existing display behavior used in the CLI visualizers."""
    return str(name).title()


def _metric_value_suffix(metric_name: str) -> str:
    """Return display suffix for metric magnitudes."""
    if metric_name == "mean_corrs":
        return ""
    return "%"


def _extract_ranked_entries(ranked_metric: pd.Series) -> List[Tuple[str, float]]:
    """Drop NaN rows and preserve sorted order from the ranked metric series."""
    entries: List[Tuple[str, float]] = []
    for champ, value in ranked_metric.items():
        if np.isnan(value):
            continue
        entries.append((str(champ), float(value)))
    return entries


def _format_ranked_entry(
    idx: int,
    champ: str,
    score: float,
    ci_low: float,
    ci_high: float,
    total_entries: int,
    value_suffix: str,
) -> str:
    """Format one ranked champion line with endpoint annotations and CI."""
    if np.isfinite(ci_low) and np.isfinite(ci_high):
        ci_text = f"({ci_low:.2f}% - {ci_high:.2f}%)"
    else:
        ci_text = "(/ - /)"

    note = ""
    if idx == 1:
        note = " (best counterpick, worst blindpick)"
    elif idx == total_entries:
        note = " (best blindpick, worst counterpick)"

    return f"{idx}. {_format_champion_name(champ)}: {score:.2f}{value_suffix} {ci_text}{note}"


def _build_ranked_line_parts(
    entries: Sequence[Tuple[str, float]],
    ci_by_champion: Dict[str, Tuple[float, float]],
    value_suffix: str,
) -> List[Tuple[str, str, str]]:
    """Build (left_text, ci_text, note_text) parts for ranked output lines."""
    parts: List[Tuple[str, str, str]] = []
    total_entries = len(entries)

    for idx, (champ, score) in enumerate(entries, start=1):
        left_text = f"{idx}. {_format_champion_name(champ)}: {score:.2f}{value_suffix}"
        ci_low, ci_high = ci_by_champion.get(champ, (np.nan, np.nan))
        if np.isfinite(ci_low) and np.isfinite(ci_high):
            ci_text = f"({ci_low:.2f}% - {ci_high:.2f}%)"
        else:
            ci_text = "(/ - /)"

        note_text = ""
        if idx == 1:
            note_text = " (best counterpick, worst blindpick)"
        elif idx == total_entries:
            note_text = " (best blindpick, worst counterpick)"

        parts.append((left_text, ci_text, note_text))

    return parts


def _load_lane_dataset(
    lane: str, base_dir: Path
) -> Optional[Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.Series]]]:
    """Load per-lane data, games matrix, and baseline series."""
    lane_file = LANE_FILES[lane]
    lane_path = base_dir / lane_file
    if not lane_path.exists():
        print(f"Warning: skipped lane '{lane}' because file was not found: {lane_path}")
        return None

    data = functions.load_data(str(lane_path))
    games_data = functions.load_matchup_games_data(str(lane_path), data.index, data.columns)
    baseline = data.attrs.get(functions.CHAMPION_BASELINE_ATTR)
    if baseline is None:
        baseline = functions.load_source_page_winrates(str(lane_path), data.columns)

    return data, games_data, baseline


def _calculate_lane_metric(
    metric_name: str,
    data: pd.DataFrame,
    games_data: Optional[pd.DataFrame],
    baseline: Optional[pd.Series],
) -> Optional[pd.Series]:
    """Calculate one ranked metric series for a lane."""
    spec = METRIC_SPECS.get(metric_name)
    if spec is None:
        raise ValueError(f"Unsupported metric '{metric_name}'.")

    metric_fn = spec["fn"]
    requires_games = bool(spec["requires_games"])
    requires_baseline = bool(spec["requires_baseline"])

    if requires_games and games_data is None:
        return None
    if requires_baseline and baseline is None:
        return None

    try:
        if requires_games and requires_baseline:
            return metric_fn(data, games_data, baseline=baseline)
        if requires_games:
            return metric_fn(data, games_data)
        if requires_baseline:
            return metric_fn(data, baseline=baseline)
        return metric_fn(data)
    except ValueError:
        return None


def _calculate_lane_ci(
    data: pd.DataFrame, confidence_level: float
) -> Dict[str, Tuple[float, float]]:
    """Calculate matchup-based confidence intervals per champion row."""
    return functions._calculate_matchup_ci_by_champion(data, confidence_level=confidence_level)


def _load_lane_metric_with_ci(
    lane: str,
    base_dir: Path,
    metric_name: str,
    confidence_level: float,
) -> Optional[Tuple[pd.Series, Dict[str, Tuple[float, float]]]]:
    """Load one lane and return ranked metric with CI bounds."""
    loaded = _load_lane_dataset(lane, base_dir)
    if loaded is None:
        return None
    data, games_data, baseline = loaded

    ranked_metric = _calculate_lane_metric(metric_name, data, games_data, baseline)
    if ranked_metric is None:
        print(f"Warning: skipped lane '{lane}' because metric '{metric_name}' could not be calculated.")
        return None

    ci_by_champion = _calculate_lane_ci(data, confidence_level=confidence_level)
    return ranked_metric, ci_by_champion


def _format_lane_block_normal(
    lane: str,
    ranked_metric: pd.Series,
    ci_by_champion: Dict[str, Tuple[float, float]],
    value_suffix: str,
) -> List[str]:
    """Format one lane section as multi-line ranked output with CI column alignment."""
    lines: List[str] = [LANE_HEADERS[lane], ""]
    entries = _extract_ranked_entries(ranked_metric)

    if not entries:
        lines.append("No ranked champions were available for this lane.")
        lines.append("")
        return lines

    line_parts = _build_ranked_line_parts(entries, ci_by_champion, value_suffix)
    ci_column = max(len(left_text) for left_text, _, _ in line_parts) + 2
    for left_text, ci_text, note_text in line_parts:
        lines.append(f"{left_text.ljust(ci_column)}{ci_text}{note_text}")

    lines.append("")
    return lines


def _format_lane_block_inline(
    lane: str,
    ranked_metric: pd.Series,
    ci_by_champion: Dict[str, Tuple[float, float]],
    value_suffix: str,
) -> List[str]:
    """Format one lane section as an inline single-line ranked output."""
    lines: List[str] = [LANE_HEADERS[lane], ""]
    entries = _extract_ranked_entries(ranked_metric)

    if not entries:
        lines.append("No ranked champions were available for this lane.")
        lines.append("")
        return lines

    inline_parts: List[str] = []
    for idx, (champ, score) in enumerate(entries, start=1):
        ci_low, ci_high = ci_by_champion.get(champ, (np.nan, np.nan))
        inline_parts.append(
            _format_ranked_entry(
                idx,
                champ,
                score,
                ci_low,
                ci_high,
                len(entries),
                value_suffix,
            )
        )

    lines.append(" | ".join(inline_parts))
    lines.append("")
    return lines


def _build_report_lines(
    metric_name: str,
    base_dir: Path,
    formatter,
    confidence_level: float,
) -> List[str]:
    """Build full report lines for all lanes using the provided formatter."""
    all_lines: List[str] = []

    value_suffix = _metric_value_suffix(metric_name)

    for lane in LANE_ORDER:
        loaded = _load_lane_metric_with_ci(
            lane=lane,
            base_dir=base_dir,
            metric_name=metric_name,
            confidence_level=confidence_level,
        )
        if loaded is None:
            continue
        ranked_metric, ci_by_champion = loaded
        all_lines.extend(formatter(lane, ranked_metric, ci_by_champion, value_suffix))

    if not all_lines:
        raise RuntimeError("No lane sections were generated. Check input files and metric requirements.")

    return all_lines


def _write_report(output_path: Path, lines: Sequence[str]) -> Path:
    """Write report lines to disk with a stable trailing newline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def make_metric_reports(
    metric_name: str,
    output_stem: Path,
    base_dir: Path,
    confidence_level: float = 0.95,
) -> Tuple[Path, Path]:
    """Build and write both normal and inline report variants for a metric."""
    normal_lines = _build_report_lines(
        metric_name=metric_name,
        base_dir=base_dir,
        formatter=_format_lane_block_normal,
        confidence_level=confidence_level,
    )
    inline_lines = _build_report_lines(
        metric_name=metric_name,
        base_dir=base_dir,
        formatter=_format_lane_block_inline,
        confidence_level=confidence_level,
    )

    normal_path = output_stem.parent / f"{output_stem.stem}_normal.txt"
    inline_path = output_stem.parent / f"{output_stem.stem}_inline.txt"
    return _write_report(normal_path, normal_lines), _write_report(inline_path, inline_lines)


def make_mae_games_weighed_equalized_reports(
    output_stem: Path,
    base_dir: Path,
    confidence_level: float = 0.95,
) -> Tuple[Path, Path]:
    """Backward-compatible wrapper for the default metric."""
    return make_metric_reports(
        metric_name="mae_games_weighed_equalized",
        output_stem=output_stem,
        base_dir=base_dir,
        confidence_level=confidence_level,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate lane ranking reports for a selected metric and include matchup-based "
            "confidence intervals in both normal and inline outputs."
        )
    )
    parser.add_argument(
        "--metric",
        default="mae_games_weighed_equalized",
        choices=sorted(METRIC_SPECS.keys()),
        help="Metric used for ranking champions (default: mae_games_weighed_equalized).",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Two-sided confidence level for matchup CI bounds (default: 0.95).",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help=(
            "Output filename stem. If omitted, defaults to the metric name. "
            "Writes <stem>_normal.txt and <stem>_inline.txt."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Legacy alias for --output-stem. Writes <stem>_normal.txt and <stem>_inline.txt.",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Directory containing matchup TSV/CSV/XLSX lane files (default: current directory).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(os.path.abspath(args.base_dir))

    if not 0 < args.confidence_level < 1:
        raise SystemExit("confidence-level must be between 0 and 1 (exclusive).")

    output_input = args.output if args.output is not None else (args.output_stem or args.metric)
    output_stem = Path(output_input)
    if not output_stem.is_absolute():
        output_stem = Path(os.path.abspath(output_input))

    normal_path, inline_path = make_metric_reports(
        metric_name=args.metric,
        output_stem=output_stem,
        base_dir=base_dir,
        confidence_level=args.confidence_level,
    )
    print(f"Metric: {args.metric}")
    print(f"Report written: {normal_path}")
    print(f"Report written: {inline_path}")


if __name__ == "__main__":
    main()

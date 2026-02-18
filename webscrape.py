#!/usr/bin/env python3
"""Expansive OP.GG matchup scraper across all ranked lanes.

By default this script crawls all 5 lanes:
- top
- jungle
- mid
- adc
- support

Each lane writes to its own CSV/TSV/XLSX set:
- matchups_<lane>_expansive.csv
- matchups_<lane>_expansive.tsv
- matchups_<lane>_expansive.xlsx

Rows include `lane` and `patch` fields so data is categorized per role and patch.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import re
import time
import unicodedata
import zipfile
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urlencode, urlparse
from xml.sax.saxutils import escape as xml_escape

import requests


LANE_CONFIG: Dict[str, Dict[str, str]] = {
    "top": {"path": "top", "seed": "Camille"},
    "jungle": {"path": "jungle", "seed": "Elise"},
    "mid": {"path": "mid", "seed": "Zed"},
    "adc": {"path": "adc", "seed": "Jinx"},
    "support": {"path": "support", "seed": "Braum"},
}

LANE_ALIASES: Dict[str, str] = {
    "top": "top",
    "jungle": "jungle",
    "jg": "jungle",
    "mid": "mid",
    "middle": "mid",
    "adc": "adc",
    "bot": "adc",
    "bottom": "adc",
    "support": "support",
    "sup": "support",
}

LANE_ORDER = ["top", "jungle", "mid", "adc", "support"]
BASE_URL_TEMPLATE = "https://op.gg/lol/champions/{slug}/counters/{lane_path}"
DEFAULT_PATCHES = ["16.03", "16.02"]

OUTPUT_FIELDS = [
    "lane",
    "patch",
    "source_champion",
    "source_slug",
    "source_page_win_rate",
    "opponent_champion",
    "opponent_slug",
    "win_rate",
    "games",
    "url",
    "scraped_at_utc",
]

UL_PATTERN = re.compile(r'<ul class="[^"]*border-t-gray-200[^"]*">(.*?)</ul>', re.S | re.I)
LI_PATTERN = re.compile(r"<li\b.*?</li>", re.S | re.I)
ALT_PATTERN = re.compile(r'<img[^>]*alt="([^"]+)"', re.S | re.I)
SRC_PATTERN = re.compile(r'<img[^>]*src="([^"]+)"', re.S | re.I)
CHAMPION_KEY_PATTERN = re.compile(r"/champion/([^/.?]+)\.png", re.I)
WIN_RATE_PATTERN = re.compile(r">(\d{1,3}(?:\.\d+)?)\s*(?:<!-- -->)?\s*%\s*<")
GAMES_SPAN_PATTERN = re.compile(r'<span class="text-xs text-gray-600">([^<]+)</span>', re.S | re.I)
TITLE_PATTERN = re.compile(r"<title>(.*?)</title>", re.S | re.I)
SOURCE_PAGE_WIN_RATE_PATTERN = re.compile(
    r"<em[^>]*>\s*Win\s*rate\s*</em>\s*<b[^>]*>\s*(\d{1,3}(?:\.\d+)?)\s*(?:<!-- -->)?\s*%",
    re.S | re.I,
)

SLUG_ALIASES = {
    "wukong": "monkeyking",
}


def normalize_lane_choice(value: str) -> Optional[str]:
    token = value.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    return LANE_ALIASES.get(token)


def parse_lane_selection(value: str) -> List[str]:
    if value.strip().lower() == "all":
        return LANE_ORDER.copy()

    lanes: List[str] = []
    raw_parts = [part.strip() for part in value.split(",") if part.strip()]
    if not raw_parts:
        raise ValueError("No lane values provided. Use 'all' or lane names.")

    for part in raw_parts:
        lane = normalize_lane_choice(part)
        if lane is None:
            raise ValueError(
                f"Unsupported lane '{part}'. Use top, jungle/jg, mid, adc/bot, support/sup, or all."
            )
        if lane not in lanes:
            lanes.append(lane)

    return lanes


def normalize_patch_token(value: str) -> Optional[str]:
    token = value.strip()
    if not token:
        return None
    if not re.fullmatch(r"\d+\.\d{2}", token):
        return None
    return token


def parse_patch_selection(value: str) -> List[str]:
    raw_parts = [part.strip() for part in value.split(",") if part.strip()]
    if not raw_parts:
        raise ValueError("No patch values provided.")

    return _normalize_patch_parts(raw_parts)


def _normalize_patch_parts(raw_parts: List[str]) -> List[str]:
    if not raw_parts:
        raise ValueError("No patch values provided.")

    patches: List[str] = []
    for part in raw_parts:
        normalized = normalize_patch_token(part)
        if normalized is None:
            raise ValueError(
                f"Unsupported patch '{part}'. Use comma-separated values like 16.03,16.02."
            )
        if normalized not in patches:
            patches.append(normalized)
    return patches


def parse_patch_config_value(value: object) -> Optional[List[str]]:
    """Parse patch config values from JSON list or comma-separated string."""
    if value is None:
        return None
    if isinstance(value, str):
        return parse_patch_selection(value)
    if isinstance(value, list):
        raw_parts = [str(part).strip() for part in value if str(part).strip()]
        return _normalize_patch_parts(raw_parts)
    raise ValueError("config patches must be a string or a list.")


def load_patches_from_config(config_path: Path) -> Optional[List[str]]:
    """Load patch list from config.json key `patches`, when available."""
    if not config_path.exists():
        return None

    try:
        with config_path.open("r", encoding="utf-8") as infile:
            config_data = json.load(infile)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as error:
        raise ValueError(f"Unable to parse config file '{config_path}': {error}") from error

    if not isinstance(config_data, dict):
        raise ValueError(f"Config file '{config_path}' must contain a JSON object.")

    return parse_patch_config_value(config_data.get("patches"))


def build_counters_url(slug: str, lane_path: str, patch: str) -> str:
    base_url = BASE_URL_TEMPLATE.format(slug=slug, lane_path=lane_path)
    return f"{base_url}?{urlencode({'patch': patch})}"


def normalize_slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]", "", normalized.lower())
    return SLUG_ALIASES.get(slug, slug)


def parse_source_name(page_html: str, fallback: str) -> str:
    match = TITLE_PATTERN.search(page_html)
    if not match:
        return fallback
    title = html.unescape(match.group(1)).strip()
    if " Counters" in title:
        return title.split(" Counters", 1)[0].strip() or fallback
    return fallback


def parse_source_page_win_rate(page_html: str) -> Optional[float]:
    """Parse the source champion's page-level win rate from the stats summary."""
    match = SOURCE_PAGE_WIN_RATE_PATTERN.search(page_html)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_games(value: str) -> Optional[int]:
    text = value.strip().replace(",", "")
    number_match = re.fullmatch(r"(\d+(?:\.\d+)?)([KkMm]?)", text)
    if not number_match:
        return None
    amount = float(number_match.group(1))
    suffix = number_match.group(2).lower()
    if suffix == "k":
        amount *= 1_000
    elif suffix == "m":
        amount *= 1_000_000
    return int(round(amount))


def extract_champion_slug_from_image_src(image_src: str) -> Optional[str]:
    key_match = CHAMPION_KEY_PATTERN.search(image_src)
    if not key_match:
        return None
    key = html.unescape(key_match.group(1)).strip()
    if not key:
        return None
    return normalize_slug(key)


def parse_matchups_from_html(page_html: str) -> List[Dict[str, object]]:
    ul_match = UL_PATTERN.search(page_html)
    if not ul_match:
        return []

    ul_html = ul_match.group(1)
    li_blocks = LI_PATTERN.findall(ul_html)
    parsed_rows: List[Dict[str, object]] = []

    for li_html in li_blocks:
        alt_match = ALT_PATTERN.search(li_html)
        if not alt_match:
            continue

        opponent_name = html.unescape(alt_match.group(1)).strip()
        if not opponent_name:
            continue

        src_match = SRC_PATTERN.search(li_html)
        opponent_slug = None
        if src_match:
            opponent_slug = extract_champion_slug_from_image_src(src_match.group(1))
        if not opponent_slug:
            opponent_slug = normalize_slug(opponent_name)

        win_rate = None
        win_rate_match = WIN_RATE_PATTERN.search(li_html)
        if win_rate_match:
            win_rate = float(win_rate_match.group(1))

        games_raw = None
        games_match = GAMES_SPAN_PATTERN.findall(li_html)
        if games_match:
            games_raw = html.unescape(games_match[-1]).strip()
        games = parse_games(games_raw) if games_raw else None

        parsed_rows.append(
            {
                "opponent_name": opponent_name,
                "opponent_slug": opponent_slug,
                "win_rate": win_rate,
                "games": games,
            }
        )

    return parsed_rows


def is_valid_counters_url(url: str, lane_path: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    return path.startswith("/lol/champions/") and path.endswith(f"/counters/{lane_path}")


def fetch_champion_page(
    session: requests.Session,
    slug: str,
    lane_path: str,
    patch: str,
    timeout: float,
    retries: int,
    retry_delay: float,
) -> Tuple[Optional[str], str]:
    target_url = build_counters_url(slug=slug, lane_path=lane_path, patch=patch)
    last_reason = ""

    for attempt in range(1, retries + 1):
        try:
            response = session.get(target_url, timeout=timeout, allow_redirects=True)
        except requests.RequestException as exc:
            last_reason = f"request error: {exc}"
            if attempt < retries:
                time.sleep(retry_delay * attempt)
            continue

        if response.status_code != 200:
            last_reason = f"unexpected status code {response.status_code}"
            if attempt < retries:
                time.sleep(retry_delay * attempt)
            continue

        if not is_valid_counters_url(response.url, lane_path):
            last_reason = f"resolved to non-champion counters page: {response.url}"
            return None, last_reason

        return response.text, ""

    return None, last_reason or "unknown fetch failure"


def load_existing_state(
    output_path: Path,
    patches: List[str],
) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str, str]], Dict[str, str]]:
    processed_source_patches: Set[Tuple[str, str]] = set()
    known_edges: Set[Tuple[str, str, str]] = set()
    slug_to_name: Dict[str, str] = {}

    if not output_path.exists() or output_path.stat().st_size == 0:
        return processed_source_patches, known_edges, slug_to_name

    with output_path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            source_slug = (row.get("source_slug") or "").strip()
            source_name = (row.get("source_champion") or "").strip()
            opponent_slug = (row.get("opponent_slug") or "").strip()
            opponent_name = (row.get("opponent_champion") or "").strip()
            patch = infer_row_patch(row)
            if patch not in patches:
                patch = patches[0]

            if source_slug:
                processed_source_patches.add((source_slug, patch))
                if source_name and source_slug not in slug_to_name:
                    slug_to_name[source_slug] = source_name

            if opponent_slug and opponent_name and opponent_slug not in slug_to_name:
                slug_to_name[opponent_slug] = opponent_name

            if source_slug and opponent_slug and patch:
                known_edges.add((source_slug, opponent_slug, patch))

    return processed_source_patches, known_edges, slug_to_name


def ensure_csv_writer(
    output_path: Path,
) -> Tuple[csv.DictWriter, object]:
    _ensure_csv_schema(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    outfile = output_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(outfile, fieldnames=OUTPUT_FIELDS)
    if not file_exists:
        writer.writeheader()
    return writer, outfile


def infer_row_patch(row: Dict[str, str]) -> str:
    patch = (row.get("patch") or "").strip()
    if patch:
        return patch

    url = (row.get("url") or "").strip()
    if url:
        parsed_query = parse_qs(urlparse(url).query)
        url_patch = (parsed_query.get("patch") or [""])[0].strip()
        if url_patch:
            return url_patch

    return DEFAULT_PATCHES[0]


def _ensure_csv_schema(output_path: Path) -> None:
    """Backfill older CSV headers so new fields can be appended safely."""
    if not output_path.exists() or output_path.stat().st_size == 0:
        return

    with output_path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        existing_fields = reader.fieldnames or []
        existing_rows = [dict(row) for row in reader]
        has_blank_patch_rows = any(not (row.get("patch") or "").strip() for row in existing_rows)
        if existing_fields == OUTPUT_FIELDS and not has_blank_patch_rows:
            return

    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in existing_rows:
            row_payload = {field: row.get(field, "") for field in OUTPUT_FIELDS}
            row_payload["patch"] = infer_row_patch(row)
            writer.writerow(row_payload)


def load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []

    with csv_path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        return [dict(row) for row in reader]


def build_matrix_from_rows(
    rows: List[Dict[str, str]],
) -> Tuple[List[str], Dict[Tuple[str, str], str]]:
    champion_order: List[str] = []
    seen_names: Set[str] = set()

    for row in rows:
        source = (row.get("source_champion") or "").strip()
        opponent = (row.get("opponent_champion") or "").strip()
        if source and source not in seen_names:
            seen_names.add(source)
            champion_order.append(source)
        if opponent and opponent not in seen_names:
            seen_names.add(opponent)
            champion_order.append(opponent)

    winrate_accumulator: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row in rows:
        source = (row.get("source_champion") or "").strip()
        opponent = (row.get("opponent_champion") or "").strip()
        raw_win_rate = (row.get("win_rate") or "").strip()
        raw_games = (row.get("games") or "").strip()
        win_rate_value = to_float_or_none(raw_win_rate)
        if not source or not opponent or win_rate_value is None:
            continue

        games_value = parse_games(raw_games) if raw_games else None
        key = (source, opponent)
        bucket = winrate_accumulator.setdefault(
            key,
            {
                "weighted_sum": 0.0,
                "weight_total": 0.0,
                "unweighted_sum": 0.0,
                "unweighted_count": 0.0,
            },
        )

        if games_value is not None and games_value > 0:
            bucket["weighted_sum"] += win_rate_value * games_value
            bucket["weight_total"] += float(games_value)
        else:
            bucket["unweighted_sum"] += win_rate_value
            bucket["unweighted_count"] += 1.0

    winrate_lookup: Dict[Tuple[str, str], str] = {}
    for key, bucket in winrate_accumulator.items():
        if bucket["weight_total"] > 0:
            win_rate = bucket["weighted_sum"] / bucket["weight_total"]
        elif bucket["unweighted_count"] > 0:
            win_rate = bucket["unweighted_sum"] / bucket["unweighted_count"]
        else:
            continue
        winrate_lookup[key] = f"{win_rate:.2f}"

    return champion_order, winrate_lookup


def write_matrix_tsv(csv_path: Path, matrix_path: Path) -> None:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        print("No CSV data found; skipped matrix export.")
        return

    rows = load_csv_rows(csv_path)
    champion_order, winrate_lookup = build_matrix_from_rows(rows)

    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    with matrix_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile, delimiter="\t")
        writer.writerow(["Play As (left) / Vs (top)", *champion_order])

        for source in champion_order:
            row = [source]
            for opponent in champion_order:
                if source == opponent:
                    row.append("/")
                else:
                    row.append(winrate_lookup.get((source, opponent), "/"))
            writer.writerow(row)

    print(f"Matrix TSV written: {matrix_path}")


def to_float_or_none(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def excel_column_name(index: int) -> str:
    index += 1
    letters = ""
    while index > 0:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def build_sheet_xml(table_rows: List[List[object]]) -> str:
    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>',
    ]

    for row_idx, row in enumerate(table_rows, start=1):
        lines.append(f'<row r="{row_idx}">')
        for col_idx, value in enumerate(row, start=1):
            if value is None or value == "":
                continue

            cell_ref = f"{excel_column_name(col_idx - 1)}{row_idx}"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                lines.append(f'<c r="{cell_ref}"><v>{value}</v></c>')
            else:
                text = xml_escape(str(value))
                lines.append(
                    f'<c r="{cell_ref}" t="inlineStr"><is><t>{text}</t></is></c>'
                )
        lines.append("</row>")

    lines.append("</sheetData></worksheet>")
    return "".join(lines)


def write_excel_workbook(csv_path: Path, excel_path: Path) -> None:
    rows = load_csv_rows(csv_path)
    if not rows:
        print("No CSV data found; skipped Excel export.")
        return

    champion_order, winrate_lookup = build_matrix_from_rows(rows)

    raw_table: List[List[object]] = [OUTPUT_FIELDS.copy()]
    for row in rows:
        raw_table.append(
            [
                row.get("lane", ""),
                row.get("patch", ""),
                row.get("source_champion", ""),
                row.get("source_slug", ""),
                to_float_or_none(row.get("source_page_win_rate", "")) if row.get("source_page_win_rate") else "",
                row.get("opponent_champion", ""),
                row.get("opponent_slug", ""),
                to_float_or_none(row.get("win_rate", "")) if row.get("win_rate") else "",
                int(row["games"]) if str(row.get("games", "")).isdigit() else row.get("games", ""),
                row.get("url", ""),
                row.get("scraped_at_utc", ""),
            ]
        )

    matrix_table: List[List[object]] = [["Play As (left) / Vs (top)", *champion_order]]
    for source in champion_order:
        matrix_row: List[object] = [source]
        for opponent in champion_order:
            if source == opponent:
                matrix_row.append("/")
            else:
                raw_winrate = winrate_lookup.get((source, opponent), "")
                matrix_row.append(to_float_or_none(raw_winrate) if raw_winrate else "/")
        matrix_table.append(matrix_row)

    raw_sheet_xml = build_sheet_xml(raw_table)
    matrix_sheet_xml = build_sheet_xml(matrix_table)

    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/worksheets/sheet2.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )
    package_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets>"
        '<sheet name="Matchups_Raw" sheetId="1" r:id="rId1"/>'
        '<sheet name="Square_Matrix" sheetId="2" r:id="rId2"/>'
        "</sheets>"
        "</workbook>"
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet2.xml"/>'
        "</Relationships>"
    )

    excel_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(excel_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", package_rels_xml)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        archive.writestr("xl/worksheets/sheet1.xml", raw_sheet_xml)
        archive.writestr("xl/worksheets/sheet2.xml", matrix_sheet_xml)

    print(f"Excel workbook written: {excel_path}")


def update_matrix_and_excel_outputs(
    csv_path: Path,
    matrix_output_path: Optional[Path],
    excel_output_path: Optional[Path],
) -> None:
    if matrix_output_path is not None:
        write_matrix_tsv(csv_path, matrix_output_path)
    if excel_output_path is not None:
        write_excel_workbook(csv_path, excel_output_path)


def crawl_expansive(
    lane: str,
    start_champion: str,
    patches: List[str],
    output_path: Path,
    matrix_output_path: Optional[Path],
    excel_output_path: Optional[Path],
    delay_seconds: float,
    timeout_seconds: float,
    retries: int,
    max_champions: Optional[int],
) -> None:
    lane_path = LANE_CONFIG[lane]["path"]
    start_slug = normalize_slug(start_champion)
    now_label = dt.datetime.now(dt.timezone.utc).isoformat()
    print("")
    print(f"[{lane}] Started crawl at {now_label}")
    print(f"[{lane}] Seed champion: {start_champion} ({start_slug})")
    print(f"[{lane}] Patches: {', '.join(patches)}")
    print(f"[{lane}] Output CSV: {output_path}")

    if excel_output_path is not None and excel_output_path.exists():
        excel_output_path.unlink()
        print(f"[{lane}] Removed existing Excel workbook for full rebuild: {excel_output_path}")

    processed_source_patches, known_edges, slug_to_name = load_existing_state(output_path, patches)
    fully_processed_sources = {
        slug
        for slug in {source_slug for source_slug, _ in processed_source_patches}
        if all((slug, patch) in processed_source_patches for patch in patches)
    }
    if processed_source_patches:
        print(
            f"[{lane}] Resuming from existing CSV. "
            f"Already processed champions (all requested patches): {len(fully_processed_sources)}"
        )

    queue: Deque[str] = deque()
    queued_slugs: Set[str] = set()

    def is_source_fully_processed(slug: str) -> bool:
        return all((slug, patch) in processed_source_patches for patch in patches)

    def enqueue(slug: str, name: Optional[str] = None) -> None:
        if not slug:
            return
        if name and slug not in slug_to_name:
            slug_to_name[slug] = name
        if is_source_fully_processed(slug) or slug in queued_slugs:
            return
        queue.append(slug)
        queued_slugs.add(slug)

    enqueue(start_slug, start_champion)
    for known_slug in list(slug_to_name.keys()):
        enqueue(known_slug, slug_to_name.get(known_slug))

    if not queue:
        print(f"[{lane}] No champions left to process.")
        update_matrix_and_excel_outputs(output_path, matrix_output_path, excel_output_path)
        print(f"[{lane}] Rebuilt matrix/Excel outputs from existing CSV state.")
        return

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )

    csv_writer, outfile = ensure_csv_writer(output_path)
    champions_processed_this_run = 0
    rows_written_this_run = 0

    update_matrix_and_excel_outputs(output_path, matrix_output_path, excel_output_path)

    try:
        while queue:
            if max_champions is not None and champions_processed_this_run >= max_champions:
                print(f"[{lane}] Reached max champion limit: {max_champions}")
                break

            source_slug = queue.popleft()
            queued_slugs.discard(source_slug)

            if is_source_fully_processed(source_slug):
                continue

            source_name_fallback = slug_to_name.get(source_slug, source_slug.title())
            print("")
            print(f"[{lane}] Scraping: {source_name_fallback} ({source_slug})")

            discovered_names: List[str] = []
            new_rows_for_source = 0
            for patch in patches:
                source_patch_key = (source_slug, patch)
                if source_patch_key in processed_source_patches:
                    continue

                target_url = build_counters_url(slug=source_slug, lane_path=lane_path, patch=patch)
                page_html, failure_reason = fetch_champion_page(
                    session=session,
                    slug=source_slug,
                    lane_path=lane_path,
                    patch=patch,
                    timeout=timeout_seconds,
                    retries=retries,
                    retry_delay=max(delay_seconds, 0.5),
                )
                if page_html is None:
                    print(f"[{lane}] Skipped {source_slug} patch {patch}: {failure_reason}")
                    processed_source_patches.add(source_patch_key)
                    continue

                source_name = parse_source_name(page_html, source_name_fallback)
                source_page_win_rate = parse_source_page_win_rate(page_html)
                slug_to_name[source_slug] = source_name

                matchups = parse_matchups_from_html(page_html)
                if not matchups:
                    print(
                        f"[{lane}] No matchup rows found in target UL for {source_slug} patch {patch}; "
                        f"marking patch as processed."
                    )
                    processed_source_patches.add(source_patch_key)
                    continue

                scraped_at = dt.datetime.now(dt.timezone.utc).isoformat()
                for matchup in matchups:
                    opponent_name = str(matchup["opponent_name"])
                    opponent_slug = str(matchup["opponent_slug"])
                    win_rate = matchup.get("win_rate")
                    games = matchup.get("games")

                    if opponent_slug:
                        enqueue(opponent_slug, opponent_name)
                    discovered_names.append(opponent_name)

                    edge_key = (source_slug, opponent_slug, patch)
                    if edge_key in known_edges:
                        continue

                    csv_writer.writerow(
                        {
                            "lane": lane,
                            "patch": patch,
                            "source_champion": source_name,
                            "source_slug": source_slug,
                            "source_page_win_rate": (
                                f"{source_page_win_rate:.2f}" if isinstance(source_page_win_rate, float) else ""
                            ),
                            "opponent_champion": opponent_name,
                            "opponent_slug": opponent_slug,
                            "win_rate": f"{win_rate:.2f}" if isinstance(win_rate, float) else "",
                            "games": games if isinstance(games, int) else "",
                            "url": target_url,
                            "scraped_at_utc": scraped_at,
                        }
                    )
                    known_edges.add(edge_key)
                    new_rows_for_source += 1
                    rows_written_this_run += 1

                processed_source_patches.add(source_patch_key)

            outfile.flush()
            champions_processed_this_run += 1

            unique_discovered = sorted(set(discovered_names), key=str.casefold)
            print(f"[{lane}] Counter matchup champs ({len(unique_discovered)}): {', '.join(unique_discovered)}")
            fully_processed_count = len(
                {
                    slug
                    for slug in {source for source, _ in processed_source_patches}
                    if all((slug, patch) in processed_source_patches for patch in patches)
                }
            )
            print(
                f"[{lane}] Summary: "
                f"{new_rows_for_source} new rows, "
                f"{fully_processed_count} total champions processed, "
                f"{len(queue)} queued."
            )

            update_matrix_and_excel_outputs(output_path, matrix_output_path, excel_output_path)

            if delay_seconds > 0:
                time.sleep(delay_seconds)
    finally:
        outfile.close()

    update_matrix_and_excel_outputs(output_path, matrix_output_path, excel_output_path)

    print("")
    print(f"[{lane}] Crawl complete.")
    print(f"[{lane}] Champions processed this run: {champions_processed_this_run}")
    print(f"[{lane}] Rows written this run: {rows_written_this_run}")
    print(f"[{lane}] CSV saved to: {output_path}")


def write_combined_lane_csv(output_dir: Path, lanes: List[str]) -> None:
    all_rows: List[Dict[str, str]] = []

    for lane in lanes:
        lane_csv_path = output_dir / f"matchups_{lane}_expansive.csv"
        if not lane_csv_path.exists():
            continue
        all_rows.extend(load_csv_rows(lane_csv_path))

    if not all_rows:
        print("No per-lane CSV rows found; skipped combined CSV.")
        return

    all_rows.sort(
        key=lambda row: (
            str(row.get("lane", "")),
            str(row.get("patch", "")),
            str(row.get("source_champion", "")).casefold(),
            str(row.get("opponent_champion", "")).casefold(),
        )
    )

    combined_path = output_dir / "matchups_all_lanes_expansive.csv"
    with combined_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({field: row.get(field, "") for field in OUTPUT_FIELDS})

    print(f"Combined all-lane CSV written: {combined_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape OP.GG counters expansively for one or more lanes and export "
            "lane-specific matrices."
        )
    )
    parser.add_argument(
        "--lanes",
        default="all",
        help="Lanes to scrape: all, top, jungle/jg, mid, adc/bot, support/sup.",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Optional seed champion override applied to each selected lane.",
    )
    parser.add_argument(
        "--patches",
        default=None,
        help=(
            "Comma-separated patch list to scrape. "
            "If omitted, reads `patches` from config.json; "
            "if missing there, falls back to 16.03,16.02."
        ),
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Config file path used for patch fallback (default: config.json).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where per-lane output files are written.",
    )
    parser.add_argument(
        "--no-matrix",
        action="store_true",
        help="Skip per-lane matrix TSV export.",
    )
    parser.add_argument(
        "--no-excel",
        action="store_true",
        help="Skip per-lane Excel export.",
    )
    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="Skip writing matchups_all_lanes_expansive.csv.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.8,
        help="Delay in seconds between champion page requests.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry attempts per champion page.",
    )
    parser.add_argument(
        "--max-champions",
        type=int,
        default=None,
        help="Optional cap for number of champions to process per lane (useful for testing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        lanes = parse_lane_selection(args.lanes)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    config_path = Path(args.config).resolve()
    if args.patches is not None:
        try:
            patches = parse_patch_selection(args.patches)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    else:
        try:
            config_patches = load_patches_from_config(config_path)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        patches = config_patches if config_patches is not None else DEFAULT_PATCHES.copy()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for lane in lanes:
        start_champion = args.start if args.start else LANE_CONFIG[lane]["seed"]
        output_path = output_dir / f"matchups_{lane}_expansive.csv"
        matrix_output_path = None if args.no_matrix else output_dir / f"matchups_{lane}_expansive.tsv"
        excel_output_path = None if args.no_excel else output_dir / f"matchups_{lane}_expansive.xlsx"

        crawl_expansive(
            lane=lane,
            start_champion=start_champion,
            patches=patches,
            output_path=output_path,
            matrix_output_path=matrix_output_path,
            excel_output_path=excel_output_path,
            delay_seconds=max(0.0, args.delay),
            timeout_seconds=max(1.0, args.timeout),
            retries=max(1, args.retries),
            max_champions=args.max_champions,
        )

    if not args.skip_combined:
        write_combined_lane_csv(output_dir, lanes)


if __name__ == "__main__":
    main()

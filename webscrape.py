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

Rows include a `lane` field so data is categorized per role.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import re
import time
import unicodedata
import zipfile
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse
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

OUTPUT_FIELDS = [
    "lane",
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
    timeout: float,
    retries: int,
    retry_delay: float,
) -> Tuple[Optional[str], str]:
    target_url = BASE_URL_TEMPLATE.format(slug=slug, lane_path=lane_path)
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
) -> Tuple[Set[str], Set[Tuple[str, str]], Dict[str, str], Dict[str, str]]:
    processed_slugs: Set[str] = set()
    known_edges: Set[Tuple[str, str]] = set()
    slug_to_name: Dict[str, str] = {}
    source_to_url: Dict[str, str] = {}

    if not output_path.exists() or output_path.stat().st_size == 0:
        return processed_slugs, known_edges, slug_to_name, source_to_url

    with output_path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            source_slug = (row.get("source_slug") or "").strip()
            source_name = (row.get("source_champion") or "").strip()
            source_url = (row.get("url") or "").strip()
            opponent_slug = (row.get("opponent_slug") or "").strip()
            opponent_name = (row.get("opponent_champion") or "").strip()

            if source_slug:
                processed_slugs.add(source_slug)
                if source_name and source_slug not in slug_to_name:
                    slug_to_name[source_slug] = source_name
                if source_url:
                    source_to_url[source_slug] = source_url

            if opponent_slug and opponent_name and opponent_slug not in slug_to_name:
                slug_to_name[opponent_slug] = opponent_name

            if source_slug and opponent_slug:
                known_edges.add((source_slug, opponent_slug))

    return processed_slugs, known_edges, slug_to_name, source_to_url


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


def _ensure_csv_schema(output_path: Path) -> None:
    """Backfill older CSV headers so new fields can be appended safely."""
    if not output_path.exists() or output_path.stat().st_size == 0:
        return

    with output_path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        existing_fields = reader.fieldnames or []
        if existing_fields == OUTPUT_FIELDS:
            return
        existing_rows = [dict(row) for row in reader]

    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow({field: row.get(field, "") for field in OUTPUT_FIELDS})


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

    winrate_lookup: Dict[Tuple[str, str], str] = {}
    for row in rows:
        source = (row.get("source_champion") or "").strip()
        opponent = (row.get("opponent_champion") or "").strip()
        win_rate = (row.get("win_rate") or "").strip()
        if source and opponent and win_rate:
            winrate_lookup[(source, opponent)] = win_rate

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
    print(f"[{lane}] Output CSV: {output_path}")

    processed_slugs, known_edges, slug_to_name, _ = load_existing_state(output_path)
    if processed_slugs:
        print(f"[{lane}] Resuming from existing CSV. Already processed champions: {len(processed_slugs)}")

    queue: Deque[str] = deque()
    queued_slugs: Set[str] = set()

    def enqueue(slug: str, name: Optional[str] = None) -> None:
        if not slug:
            return
        if name and slug not in slug_to_name:
            slug_to_name[slug] = name
        if slug in processed_slugs or slug in queued_slugs:
            return
        queue.append(slug)
        queued_slugs.add(slug)

    enqueue(start_slug, start_champion)
    for known_slug in list(slug_to_name.keys()):
        enqueue(known_slug, slug_to_name.get(known_slug))

    if not queue:
        print(f"[{lane}] No champions left to process.")
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

    try:
        while queue:
            if max_champions is not None and champions_processed_this_run >= max_champions:
                print(f"[{lane}] Reached max champion limit: {max_champions}")
                break

            source_slug = queue.popleft()
            queued_slugs.discard(source_slug)

            if source_slug in processed_slugs:
                continue

            target_url = BASE_URL_TEMPLATE.format(slug=source_slug, lane_path=lane_path)
            source_name_fallback = slug_to_name.get(source_slug, source_slug.title())
            print("")
            print(f"[{lane}] Scraping: {source_name_fallback} ({source_slug})")

            page_html, failure_reason = fetch_champion_page(
                session=session,
                slug=source_slug,
                lane_path=lane_path,
                timeout=timeout_seconds,
                retries=retries,
                retry_delay=max(delay_seconds, 0.5),
            )
            if page_html is None:
                print(f"[{lane}] Skipped {source_slug}: {failure_reason}")
                processed_slugs.add(source_slug)
                champions_processed_this_run += 1
                continue

            source_name = parse_source_name(page_html, source_name_fallback)
            source_page_win_rate = parse_source_page_win_rate(page_html)
            slug_to_name[source_slug] = source_name

            matchups = parse_matchups_from_html(page_html)
            if not matchups:
                print(f"[{lane}] No matchup rows found in target UL; marking champion as processed.")
                processed_slugs.add(source_slug)
                champions_processed_this_run += 1
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
                continue

            discovered_names: List[str] = []
            scraped_at = dt.datetime.now(dt.timezone.utc).isoformat()
            new_rows_for_source = 0

            for matchup in matchups:
                opponent_name = str(matchup["opponent_name"])
                opponent_slug = str(matchup["opponent_slug"])
                win_rate = matchup.get("win_rate")
                games = matchup.get("games")

                if opponent_slug:
                    enqueue(opponent_slug, opponent_name)
                discovered_names.append(opponent_name)

                edge_key = (source_slug, opponent_slug)
                if edge_key in known_edges:
                    continue

                csv_writer.writerow(
                    {
                        "lane": lane,
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

            outfile.flush()
            processed_slugs.add(source_slug)
            champions_processed_this_run += 1

            unique_discovered = sorted(set(discovered_names), key=str.casefold)
            print(f"[{lane}] Counter matchup champs ({len(unique_discovered)}): {', '.join(unique_discovered)}")
            print(
                f"[{lane}] Summary: "
                f"{new_rows_for_source} new rows, "
                f"{len(processed_slugs)} total champions processed, "
                f"{len(queue)} queued."
            )

            update_matrix_and_excel_outputs(output_path, matrix_output_path, excel_output_path)

            if delay_seconds > 0:
                time.sleep(delay_seconds)
    finally:
        outfile.close()

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

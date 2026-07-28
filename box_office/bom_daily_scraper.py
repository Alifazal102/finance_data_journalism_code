"""
Box Office Mojo daily-gross scraper
------------------------------------
Pulls day-by-day domestic box office (date, daily gross, theater count,
per-theater average, cumulative gross, day-of-run) for a list of films
and writes:
  1) one CSV per film   -> ./bom_daily/<slug>.csv
  2) one combined panel -> ./bom_daily/all_films_daily.csv

WHY THIS APPROACH
Box Office Mojo does not publish a bulk/API dataset, so this script
does the same three steps a human would do on the site:
  1. Search BOM for the title -> get the title page URL
  2. From the title page, find the Domestic release link
     (BOM stores daily data under /release/<id>/, not /title/<id>/)
  3. Fetch /release/<id>/daily/ and parse the daily table

NOTES / CAVEATS (read before running)
- This network sandbox cannot reach boxofficemojo.com, so this script
  is written from BOM's known page/table structure but has NOT been
  executed against the live site. Scraper code for any site is brittle
  by nature -- if BOM changes markup, the CSS selectors below will need
  adjusting. Run it locally, inspect the first film's output CSV, and
  adjust SELECTORS if columns look wrong.
- Be polite: this script sleeps between requests and identifies itself
  with a normal browser User-Agent. Do not remove the delay or hammer
  the site with concurrent requests.
- Films still in their theatrical run (e.g. The Odyssey as of writing)
  will simply return a partial daily table up to the most recent day
  BOM has published -- rerun later to backfill.
- Some 2026 titles may not be up on BOM yet, may be listed under a
  slightly different title, or (for day-and-date streaming titles)
  may not have a meaningful theatrical daily chart at all. The script
  logs a warning and skips rather than crashing, so check bom_daily/
  scrape_log.txt after running.
- For films with very sparse theatrical release (e.g. Following, which
  played on a handful of screens), BOM may not have a daily chart at
  all -- only a lifetime total. Those will show up as no_daily_table
  in the log; pull the lifetime total manually if needed.
"""

import csv
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

BASE_URL = "https://www.boxofficemojo.com"
OUTPUT_DIR = Path("bom_daily")
REQUEST_DELAY_SECONDS = 2.5  # be polite -- do not lower this
TIMEOUT = 20

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Each entry: display title -> search query to use on BOM.
# Add a "year" hint where titles are ambiguous (common words, remakes, etc.)
# to help pick the right search result.
#
# Optionally add "release_url": "<full boxofficemojo.com/release/rl.../ url>"
# to bypass the automated search + release-link detection entirely and
# scrape that release directly. Use this for titles where the automated
# path picks the wrong release -- this commonly happens when a title has
# multiple listed releases (original theatrical run vs. a later awards-
# season or anniversary rerelease) and the "first row mentioning Domestic"
# heuristic grabs the rerelease's sparse/limited-theater data instead of
# the original wide release. To get the right URL by hand: search the
# title on boxofficemojo.com -> click into the "Original release" entry
# -> click "Domestic" -> copy that page's URL.
FILMS = {
    # Christopher Nolan filmography
    "Following":                {"query": "Following", "year": 1998},
    "Memento":                  {"query": "Memento", "year": 2000},
    "Insomnia":                 {"query": "Insomnia", "year": 2002},
    "Batman Begins":            {"query": "Batman Begins", "year": 2005},
    "The Prestige":             {"query": "The Prestige", "year": 2006},
    "The Dark Knight":          {"query": "The Dark Knight", "year": 2008,
                                  "release_url": "https://www.boxofficemojo.com/release/rl3729098241/"},
    "Inception":                {"query": "Inception", "year": 2010,
                                  "release_url": "https://www.boxofficemojo.com/release/rl2908456449/"},
    "The Dark Knight Rises":    {"query": "The Dark Knight Rises", "year": 2012},
    "Interstellar":             {"query": "Interstellar", "year": 2014},
    "Dunkirk":                  {"query": "Dunkirk", "year": 2017,
                                  "release_url": "https://www.boxofficemojo.com/release/rl4118644225/"},
    "Tenet":                    {"query": "Tenet", "year": 2020},
    "Oppenheimer":              {"query": "Oppenheimer", "year": 2023},
    "The Odyssey":              {"query": "The Odyssey", "year": 2026},
    # 2026 comp set
    "Obsession":                {"query": "Obsession", "year": 2026},
    "Backrooms":                {"query": "Backrooms", "year": 2026},
    "Toy Story 5":               {"query": "Toy Story 5", "year": 2026},
    "Michael":                  {"query": "Michael", "year": 2026},
    "The Devil Wears Prada 2":  {"query": "The Devil Wears Prada 2", "year": 2026},
    "Project Hail Mary":        {"query": "Project Hail Mary", "year": 2026},
    "Super Mario Galaxy Movie": {"query": "Super Mario Galaxy Movie", "year": 2026},
}


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class ScrapeResult:
    title: str
    release_url: Optional[str] = None
    rows: list = field(default_factory=list)
    status: str = "not_attempted"  # ok | no_search_match | no_release_link | no_daily_table | error
    note: str = ""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _get(session: requests.Session, url: str) -> Optional[BeautifulSoup]:
    """GET a URL with polite delay + basic retry, return parsed soup or None."""
    for attempt in range(3):
        try:
            resp = session.get(url, headers=HEADERS, timeout=TIMEOUT)
            time.sleep(REQUEST_DELAY_SECONDS)
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "lxml")
            logging.warning("Non-200 (%s) for %s", resp.status_code, url)
        except requests.RequestException as exc:
            logging.warning("Request error on attempt %d for %s: %s", attempt + 1, url, exc)
            time.sleep(REQUEST_DELAY_SECONDS * (attempt + 1))
    return None


def find_title_url(session: requests.Session, query: str, year_hint: Optional[int]) -> Optional[str]:
    """Search BOM for a title and return the best-matching /title/.../ URL."""
    search_url = f"{BASE_URL}/search/?q={requests.utils.quote(query)}"
    soup = _get(session, search_url)
    if soup is None:
        return None

    candidates = []
    for link in soup.select("a[href*='/title/']"):
        href = link.get("href", "")
        text = link.get_text(strip=True)
        if not href or not text:
            continue
        row_text = link.find_parent("tr")
        row_text = row_text.get_text(" ", strip=True) if row_text else text
        candidates.append((href, text, row_text))

    if not candidates:
        return None

    # Prefer exact (case-insensitive) title match; break ties with year hint.
    def score(candidate):
        href, text, row_text = candidate
        s = 0
        if text.strip().lower() == query.strip().lower():
            s += 10
        if year_hint and str(year_hint) in row_text:
            s += 5
        return s

    candidates.sort(key=score, reverse=True)
    best_href = candidates[0][0]
    return best_href if best_href.startswith("http") else BASE_URL + best_href


RERELEASE_PATTERN = re.compile(r"\(\s*\d{4}\s*re-?release\s*\)", re.IGNORECASE)


def is_rerelease_page(soup: BeautifulSoup) -> bool:
    """BOM marks re-release pages with a visible '(YYYY Re-release)' line
    directly under the title (confirmed against live pages, e.g.
    'Interstellar / (2024 Re-release)'). The original theatrical release
    page has no such marker. Checking the page itself is more reliable
    than trying to parse ambiguous link/row text on the title page.
    """
    # Restrict the search to roughly the first screenful of text so we
    # don't false-positive on some unrelated mention of re-release
    # further down the page (e.g. in a related-news blurb).
    head_text = soup.get_text(" ", strip=True)[:1000]
    return bool(RERELEASE_PATTERN.search(head_text))


def find_domestic_release_url(session: requests.Session, title_url: str) -> Optional[str]:
    """From a /title/.../ page, find the ORIGINAL domestic /release/.../ page.

    Titles with theatrical rereleases (anniversary runs, awards-season
    reissues, IMAX re-releases, etc.) list multiple releases on their
    title page, and simple 'first row mentioning Domestic' logic can grab
    a rerelease's page instead of the original wide release -- rereleases
    usually cover a handful of theaters over a few days, which silently
    produces a tiny, wrong daily table. To avoid that, we collect every
    domestic-release candidate link, then check each candidate release
    page itself for BOM's '(YYYY Re-release)' marker and skip any that
    have it, returning the first genuine original release we find.
    """
    soup = _get(session, title_url)
    if soup is None:
        return None

    domestic_candidates = []
    for row in soup.select("table tr"):
        row_text = row.get_text(" ", strip=True)
        if "domestic" not in row_text.lower():
            continue
        for link in row.find_all("a", href=re.compile(r"^/release/")):
            href = link.get("href")
            if href not in domestic_candidates:
                domestic_candidates.append(href)

    if not domestic_candidates:
        for link in soup.find_all("a", href=re.compile(r"^/release/")):
            href = link.get("href")
            if href not in domestic_candidates:
                domestic_candidates.append(href)

    if not domestic_candidates:
        return None

    # Prefer bare .../release/<id>/ links (no extra path segment) over
    # view-specific ones like .../release/<id>/weekend, since those parse
    # more reliably -- but keep all candidates in play in case the bare
    # one turns out to be a rerelease and a later one is the original.
    bare_pattern = re.compile(r"^/release/rl\d+/?(\?.*)?(#.*)?$")
    domestic_candidates.sort(key=lambda href: 0 if bare_pattern.match(href) else 1)

    fallback_url = None
    for href in domestic_candidates:
        full_url = href if href.startswith("http") else BASE_URL + href
        clean_url = normalize_release_url(full_url) or full_url

        candidate_soup = _get(session, clean_url)
        if candidate_soup is None:
            continue

        if fallback_url is None:
            fallback_url = clean_url  # in case every candidate is somehow a rerelease

        if not is_rerelease_page(candidate_soup):
            return clean_url

    # Every candidate looked like a rerelease (or none could be fetched) --
    # return whatever we found rather than nothing, but this case should
    # be rare and is worth spot-checking in the log/output.
    return fallback_url


def _looks_like_daily_table(table) -> bool:
    """Return True if this <table>'s header row matches BOM's daily-chart
    columns (Date / DOW / Rank / Daily / Theaters / Avg / To Date / Day).
    Picking by header content -- rather than assuming the first <table> on
    the page is the right one -- avoids silently scraping an unrelated
    table (e.g. a rankings/trivia table) that happens to appear earlier
    in the page markup.
    """
    header_cells = table.find_all(["th"])
    if not header_cells:
        first_row = table.find("tr")
        header_cells = first_row.find_all(["td", "th"]) if first_row else []
    header_text = " ".join(c.get_text(strip=True).lower() for c in header_cells)
    required = ["date", "daily"]
    return all(term in header_text for term in required)


def normalize_release_url(url: str) -> Optional[str]:
    """Reduce any /release/<id>/... URL (possibly carrying a /weekend,
    /daily, etc. path suffix, a query string like ?ref_=bo_tt_gr, and/or
    a #fragment) down to the canonical bare release URL:
    https://www.boxofficemojo.com/release/<id>/

    This matters because BOM's own internal links sometimes point at a
    specific view (e.g. the weekend chart) rather than the base release
    page, and naively appending a path onto a URL that still has a query
    string or fragment produces a malformed URL (e.g.
    '.../weekend?ref_=bo_tt_gr#table/') that silently fails to load the
    real daily chart.
    """
    match = re.search(r"/release/(rl\d+)", url)
    if not match:
        return None
    return f"{BASE_URL}/release/{match.group(1)}/"


def scrape_daily_table(session: requests.Session, release_url: str, title: str) -> list:
    """Fetch and parse the release page's daily chart into a list of dict rows.

    BOM serves the daily chart at the bare release URL itself
    (e.g. /release/rl3725886209/), not at a /daily/ sub-path. We try that
    first, and only fall back to a /daily/ suffix (older/alternate URL
    pattern) if the primary page doesn't contain a recognizable daily
    table. Either way we pick the specific table whose header matches
    the daily-chart columns, rather than assuming it's the first table
    in the page markup -- release pages also contain summary/nav tables
    that would otherwise be scraped by mistake.
    """
    clean_url = normalize_release_url(release_url)
    if clean_url is None:
        logging.warning("%s: could not extract a release id from %s", title, release_url)
        return []

    candidate_urls = [clean_url, clean_url + "daily/"]

    table = None
    for url in candidate_urls:
        soup = _get(session, url)
        if soup is None:
            continue
        for candidate_table in soup.find_all("table"):
            if _looks_like_daily_table(candidate_table):
                table = candidate_table
                break
        if table is not None:
            break

    if table is None:
        return []

    # pandas.read_html is more robust to minor markup changes than manual
    # cell-by-cell parsing -- use it, then normalize column names.
    try:
        from io import StringIO
        df = pd.read_html(StringIO(str(table)))[0]
    except ValueError:
        return []

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Order matters: check the most specific / collision-prone patterns
    # first (e.g. "to_date" contains "date", "day_#" contains "day", etc.)
    # so a generic later rule doesn't clobber an earlier, more specific one.
    rename_rules = [
        (lambda c: "to_date" in c or "cumulative" in c or c == "total_gross", "cumulative_gross"),
        (lambda c: "rank" in c, "daily_rank"),
        (lambda c: ("avg" in c or "average" in c) and "theater" in c, "avg_per_theater"),
        (lambda c: "avg" in c or "average" in c, "avg_per_theater"),
        (lambda c: "theater" in c or "theatre" in c, "theaters"),
        (lambda c: c == "date" or (c.endswith("date") and "to" not in c), "date"),
        (lambda c: "daily" in c or c == "gross", "daily_gross"),
        (lambda c: c.startswith("day") and "date" not in c, "day_of_run"),
    ]

    rename_map = {}
    used_targets = set()
    for col in df.columns:
        for matches, target in rename_rules:
            if target in used_targets:
                continue
            if matches(col):
                rename_map[col] = target
                used_targets.add(target)
                break
    df = df.rename(columns=rename_map)

    for money_col in ("daily_gross", "avg_per_theater", "cumulative_gross"):
        if money_col in df.columns:
            df[money_col] = (
                df[money_col]
                .astype(str)
                .str.replace(r"[^\d.]", "", regex=True)
                .replace("", "0")
                .astype(float)
            )
    if "theaters" in df.columns:
        df["theaters"] = (
            df["theaters"].astype(str).str.replace(r"[^\d]", "", regex=True).replace("", "0").astype(int)
        )

    df["title"] = title
    return df.to_dict("records")


# --------------------------------------------------------------------------
# Main scrape loop
# --------------------------------------------------------------------------

def scrape_all(films: dict) -> list:
    OUTPUT_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=OUTPUT_DIR / "scrape_log.txt",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        filemode="w",
    )

    session = requests.Session()
    results = []

    for title, meta in films.items():
        print(f"Scraping: {title} ...")
        result = ScrapeResult(title=title)

        override_url = meta.get("release_url")
        if override_url:
            release_url = normalize_release_url(override_url) or override_url
            print(f"  (using manual release_url override: {release_url})")
        else:
            title_url = find_title_url(session, meta["query"], meta.get("year"))
            if not title_url:
                result.status = "no_search_match"
                result.note = "Could not find a title page via BOM search."
                logging.warning("%s: %s", title, result.note)
                results.append(result)
                continue

            release_url = find_domestic_release_url(session, title_url)
            if not release_url:
                result.status = "no_release_link"
                result.note = f"Found title page ({title_url}) but no domestic release link."
                logging.warning("%s: %s", title, result.note)
                results.append(result)
                continue
        result.release_url = release_url

        rows = scrape_daily_table(session, release_url, title)
        if not rows:
            result.status = "no_daily_table"
            result.note = (
                f"Found release page ({release_url}) but no parseable daily table "
                "(film may have too limited a release for BOM to publish daily data)."
            )
            logging.warning("%s: %s", title, result.note)
            results.append(result)
            continue

        result.rows = rows
        result.status = "ok"
        results.append(result)

        # Write per-film CSV
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        out_path = OUTPUT_DIR / f"{slug}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  -> {len(rows)} daily rows written to {out_path}")

    return results


def build_combined_panel(results: list) -> pd.DataFrame:
    all_rows = [row for r in results for row in r.rows]
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    combined_path = OUTPUT_DIR / "all_films_daily.csv"
    df.to_csv(combined_path, index=False)
    print(f"\nCombined panel written to {combined_path} ({len(df)} rows total)")
    return df


def print_summary(results: list) -> None:
    print("\n=== Scrape summary ===")
    for r in results:
        n_rows = len(r.rows)
        print(f"{r.title:30s} status={r.status:16s} rows={n_rows}")
        if r.note:
            print(f"    note: {r.note}")
    ok = sum(1 for r in results if r.status == "ok")
    print(f"\n{ok}/{len(results)} films scraped successfully. Full log: {OUTPUT_DIR / 'scrape_log.txt'}")


if __name__ == "__main__":
    scrape_results = scrape_all(FILMS)
    build_combined_panel(scrape_results)
    print_summary(scrape_results)

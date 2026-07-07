#!/usr/bin/env python3
"""Fetch the open-access real-document fixtures for the opt-in empirical tests.

Downloads the two CC BY articles from the PubMed Central Open Access
subset used by ``tests/test_pmc_real_document_tables.rs`` into
``tests/fixtures/real/`` (gitignored — real-world PDFs stay out of the
tree). The tests skip gracefully when the files are absent.

Usage:
    python3 scripts/fetch_real_fixtures.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path


DEST = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "real"

# (PMC id, output name). Both CC BY in the PMC Open Access subset:
#   PMC8103274 — Tomography 2021;7(2):95-106
#   PMC8025823
ARTICLES = [
    ("PMC8103274", "pmc8103274.pdf"),
    ("PMC8025823", "pmc8025823.pdf"),
]


def fetch(pmcid: str, name: str) -> None:
    out = DEST / name
    if out.is_file() and out.read_bytes()[:5] == b"%PDF-":
        print(f"already present: {out}")
        return
    url = f"https://europepmc.org/articles/{pmcid}?pdf=render"
    print(f"fetching {pmcid} -> {out}")
    req = urllib.request.Request(url, headers={"User-Agent": "pdf_oxide-fixture-fetch"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    if data[:5] != b"%PDF-":
        sys.exit(f"error: {pmcid} did not download as a PDF ({len(data)} bytes)")
    out.write_bytes(data)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for pmcid, name in ARTICLES:
        fetch(pmcid, name)
    print("done. Run: cargo test --test test_pmc_real_document_tables")


if __name__ == "__main__":
    main()

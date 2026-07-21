#!/usr/bin/env python3
"""PDF extraction regression harness for pdf_oxide.

Compares the HEAD build of pdf_oxide's ``extract_text_simple``,
``extract_markdown_simple``, and ``extract_html_simple`` against the
v0.3.23 baseline and external references (pdftotext, pypdfium2,
pymupdf4llm) on a curated corpus of ~60 PDFs. Used to investigate
whether commits on ``release/v0.3.25`` have regressed output quality
versus v0.3.23 across all three extraction formats (text, markdown,
html).

Subcommands:
  collect      Select a diverse corpus and write regression_corpus.txt
  run          Run every (extractor, format) combo on every PDF
  diff         Compare HEAD vs baseline in a given format
  groundtruth  Compare an extractor against pdftotext (or another ref)
  show         Dump the selected format's output for a single PDF

The script is self-contained: stdlib + (optional) pypdfium2 /
pymupdf4llm, plus subprocess calls to pdftotext and the
extract_{text,markdown,html}_simple binaries.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


# ---------------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

CORPUS_FILE_DEFAULT = SCRIPTS_DIR / "regression_corpus.txt"
RUNS_ROOT_DEFAULT = Path("/tmp/regression_runs")

TESTS_ROOT = Path(os.path.expanduser("~/projects/pdf_oxide_tests"))
REPRO_ROOT = Path("/tmp/repro_pdfs")

V0323_EXAMPLES = Path("/tmp/pdf_oxide_v0323/target/release/examples")
HEAD_EXAMPLES = REPO_ROOT / "target" / "release" / "examples"

PDFTOTEXT_BIN = "/usr/bin/pdftotext"

MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB
EXTRACTOR_TIMEOUT = 30  # seconds

FORMATS: tuple[str, ...] = ("text", "markdown", "html")

FORMAT_EXT: dict[str, str] = {
    "text": ".txt",
    "markdown": ".md",
    "html": ".html",
}


# ---------------------------------------------------------------------------
# Extractor catalogue
# ---------------------------------------------------------------------------


@dataclass
class ExtractorSpec:
    """One (extractor, format) combination.

    ``runner`` is a callable ``(pdf: Path, pages: int) -> (ok, rc, text,
    err)`` that produces a UTF-8 string to be persisted.  ``available``
    is a zero-arg availability check that reports whether this combo
    can be executed on the current machine.  When the combo is
    unavailable ``unavail_reason`` explains why (e.g. "binary not
    found at X" or "module not installed").
    """

    name: str  # logical extractor name, e.g. "head", "v0323"
    format: str  # "text" | "markdown" | "html"
    runner: Callable[[Path, int], tuple[bool, int | None, str, str | None]]
    available: Callable[[], bool]
    unavail_reason: Callable[[], str]
    file_ext: str

    @property
    def key(self) -> str:
        return f"{self.name}.{self.format}"


# ---- helper runners for rust binaries -------------------------------------


def _run_rust_bin(binary: Path, pdf: Path) -> tuple[bool, int | None, str, str | None]:
    try:
        proc = subprocess.run(
            [str(binary), str(pdf)],
            capture_output=True,
            timeout=EXTRACTOR_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return False, None, "", "timeout"
    except Exception as e:  # pragma: no cover
        return False, None, "", f"exception: {e}"
    ok = proc.returncode == 0
    try:
        text = proc.stdout.decode("utf-8", errors="replace")
    except Exception as e:
        return False, proc.returncode, "", f"decode error: {e}"
    err = None
    if not ok:
        err = proc.stderr.decode("utf-8", errors="replace")[-4000:] or f"exit {proc.returncode}"
    return ok, proc.returncode, text, err


def _make_rust_runner(
    binary: Path,
) -> Callable[[Path, int], tuple[bool, int | None, str, str | None]]:
    def runner(pdf: Path, pages: int) -> tuple[bool, int | None, str, str | None]:
        return _run_rust_bin(binary, pdf)

    return runner


def _binary_available(binary: Path) -> Callable[[], bool]:
    return lambda: binary.exists()


def _binary_unavail(binary: Path) -> Callable[[], str]:
    return lambda: f"binary not found at {binary}"


# ---- pdftotext runners (text + html) --------------------------------------


def _run_pdftotext_text(pdf: Path, pages: int) -> tuple[bool, int | None, str, str | None]:
    cmd: list[str] = [PDFTOTEXT_BIN, "-layout"]
    if pages > 0:
        cmd += ["-f", "1", "-l", str(pages)]
    cmd += [str(pdf), "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=EXTRACTOR_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False, None, "", "timeout"
    except Exception as e:
        return False, None, "", f"exception: {e}"
    if proc.returncode != 0 and pages > 0:
        # Retry without page flags — some poppler builds reject them on
        # some encrypted PDFs.
        try:
            proc = subprocess.run(
                [PDFTOTEXT_BIN, "-layout", str(pdf), "-"],
                capture_output=True,
                timeout=EXTRACTOR_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return False, None, "", "timeout"
        except Exception as e:
            return False, None, "", f"exception: {e}"
    ok = proc.returncode == 0
    text = proc.stdout.decode("utf-8", errors="replace")
    err = None
    if not ok:
        err = proc.stderr.decode("utf-8", errors="replace")[-4000:] or f"exit {proc.returncode}"
    return ok, proc.returncode, text, err


def _run_pdftotext_html(pdf: Path, pages: int) -> tuple[bool, int | None, str, str | None]:
    # pdftotext -htmlmeta writes to a file argument, not stdout.
    with tempfile.TemporaryDirectory(prefix="pdftotext_html_") as tmp:
        out_file = Path(tmp) / "out.html"
        cmd: list[str] = [PDFTOTEXT_BIN, "-layout", "-htmlmeta"]
        if pages > 0:
            cmd += ["-f", "1", "-l", str(pages)]
        cmd += [str(pdf), str(out_file)]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=EXTRACTOR_TIMEOUT)
        except subprocess.TimeoutExpired:
            return False, None, "", "timeout"
        except Exception as e:
            return False, None, "", f"exception: {e}"
        if proc.returncode != 0 and pages > 0:
            try:
                proc = subprocess.run(
                    [PDFTOTEXT_BIN, "-layout", "-htmlmeta", str(pdf), str(out_file)],
                    capture_output=True,
                    timeout=EXTRACTOR_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                return False, None, "", "timeout"
            except Exception as e:
                return False, None, "", f"exception: {e}"
        ok = proc.returncode == 0 and out_file.exists()
        if not ok:
            err = proc.stderr.decode("utf-8", errors="replace")[-4000:] or f"exit {proc.returncode}"
            return False, proc.returncode, "", err
        try:
            text = out_file.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return False, proc.returncode, "", f"read error: {e}"
        return True, proc.returncode, text, None


def _pdftotext_available() -> bool:
    return Path(PDFTOTEXT_BIN).exists()


def _pdftotext_unavail() -> str:
    return f"pdftotext binary not found at {PDFTOTEXT_BIN}"


# ---- pypdfium2 runner (text only) -----------------------------------------


def _run_pypdfium2(pdf: Path, pages: int) -> tuple[bool, int | None, str, str | None]:
    try:
        import pypdfium2 as pdfium  # type: ignore
    except Exception as e:
        return False, None, "", f"import error: {e}"
    start = time.time()
    try:
        doc = pdfium.PdfDocument(str(pdf))
        total = len(doc)
        last = total if pages <= 0 else min(total, pages)
        chunks: list[str] = []
        for i in range(last):
            if time.time() - start > EXTRACTOR_TIMEOUT:
                return False, None, "", "timeout"
            page = doc[i]
            tp = page.get_textpage()
            try:
                chunks.append(tp.get_text_range())
            finally:
                with contextlib.suppress(Exception):
                    tp.close()
                with contextlib.suppress(Exception):
                    page.close()
        with contextlib.suppress(Exception):
            doc.close()
        return True, 0, "\n".join(chunks), None
    except Exception as e:
        return False, None, "", f"exception: {e}"


def _pypdfium2_available() -> bool:
    try:
        import pypdfium2  # noqa: F401
    except Exception:
        return False
    return True


def _pypdfium2_unavail() -> str:
    try:
        import pypdfium2  # noqa: F401
    except Exception as e:
        return f"pypdfium2 import error: {e}"
    return "pypdfium2 unavailable"


# ---- pymupdf4llm runner (markdown only) -----------------------------------


def _run_pymupdf4llm(pdf: Path, pages: int) -> tuple[bool, int | None, str, str | None]:
    try:
        import pymupdf4llm  # type: ignore
    except Exception as e:
        return False, None, "", f"import error: {e}"
    try:
        # pymupdf4llm.to_markdown may or may not accept pages kwarg.
        if pages > 0:
            try:
                text = pymupdf4llm.to_markdown(str(pdf), pages=list(range(pages)))
            except TypeError:
                text = pymupdf4llm.to_markdown(str(pdf))
        else:
            text = pymupdf4llm.to_markdown(str(pdf))
        if text is None:
            text = ""
        return True, 0, text, None
    except Exception as e:
        return False, None, "", f"exception: {e}"


def _pymupdf4llm_available() -> bool:
    try:
        import pymupdf4llm  # noqa: F401
    except Exception:
        return False
    return True


def _pymupdf4llm_unavail() -> str:
    try:
        import pymupdf4llm  # noqa: F401
    except Exception as e:
        return f"pymupdf4llm import error: {e}"
    return "pymupdf4llm unavailable"


# ---- catalogue factory ----------------------------------------------------


def build_extractor_specs() -> list[ExtractorSpec]:
    head_text = HEAD_EXAMPLES / "extract_text_simple"
    head_md = HEAD_EXAMPLES / "extract_markdown_simple"
    head_html = HEAD_EXAMPLES / "extract_html_simple"

    v0323_text = V0323_EXAMPLES / "extract_text_simple"
    v0323_md = V0323_EXAMPLES / "extract_markdown_simple"
    v0323_html = V0323_EXAMPLES / "extract_html_simple"

    specs: list[ExtractorSpec] = [
        # ---- text ---------------------------------------------------------
        ExtractorSpec(
            name="v0323",
            format="text",
            runner=_make_rust_runner(v0323_text),
            available=_binary_available(v0323_text),
            unavail_reason=_binary_unavail(v0323_text),
            file_ext=FORMAT_EXT["text"],
        ),
        ExtractorSpec(
            name="head",
            format="text",
            runner=_make_rust_runner(head_text),
            available=_binary_available(head_text),
            unavail_reason=_binary_unavail(head_text),
            file_ext=FORMAT_EXT["text"],
        ),
        ExtractorSpec(
            name="pdftotext",
            format="text",
            runner=_run_pdftotext_text,
            available=_pdftotext_available,
            unavail_reason=_pdftotext_unavail,
            file_ext=FORMAT_EXT["text"],
        ),
        ExtractorSpec(
            name="pypdfium2",
            format="text",
            runner=_run_pypdfium2,
            available=_pypdfium2_available,
            unavail_reason=_pypdfium2_unavail,
            file_ext=FORMAT_EXT["text"],
        ),
        # ---- markdown -----------------------------------------------------
        ExtractorSpec(
            name="v0323",
            format="markdown",
            runner=_make_rust_runner(v0323_md),
            available=_binary_available(v0323_md),
            unavail_reason=_binary_unavail(v0323_md),
            file_ext=FORMAT_EXT["markdown"],
        ),
        ExtractorSpec(
            name="head",
            format="markdown",
            runner=_make_rust_runner(head_md),
            available=_binary_available(head_md),
            unavail_reason=_binary_unavail(head_md),
            file_ext=FORMAT_EXT["markdown"],
        ),
        ExtractorSpec(
            name="pymupdf4llm",
            format="markdown",
            runner=_run_pymupdf4llm,
            available=_pymupdf4llm_available,
            unavail_reason=_pymupdf4llm_unavail,
            file_ext=FORMAT_EXT["markdown"],
        ),
        # ---- html ---------------------------------------------------------
        ExtractorSpec(
            name="v0323",
            format="html",
            runner=_make_rust_runner(v0323_html),
            available=_binary_available(v0323_html),
            unavail_reason=_binary_unavail(v0323_html),
            file_ext=FORMAT_EXT["html"],
        ),
        ExtractorSpec(
            name="head",
            format="html",
            runner=_make_rust_runner(head_html),
            available=_binary_available(head_html),
            unavail_reason=_binary_unavail(head_html),
            file_ext=FORMAT_EXT["html"],
        ),
        ExtractorSpec(
            name="pdftotext",
            format="html",
            runner=_run_pdftotext_html,
            available=_pdftotext_available,
            unavail_reason=_pdftotext_unavail,
            file_ext=FORMAT_EXT["html"],
        ),
    ]
    return specs


# ---------------------------------------------------------------------------
# Corpus collection
# ---------------------------------------------------------------------------


@dataclass
class Bucket:
    name: str
    target: int
    candidates: list[Path] = field(default_factory=list)
    picked: list[Path] = field(default_factory=list)


def _walk_pdfs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for dirpath, _dirs, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".pdf"):
                p = Path(dirpath) / fn
                try:
                    sz = p.stat().st_size
                except OSError:
                    continue
                if sz == 0 or sz > MAX_PDF_BYTES:
                    continue
                out.append(p)
    out.sort()
    return out


def _first_n(paths: Iterable[Path], n: int, seen: set) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if len(out) >= n:
            break
        if p in seen:
            continue
        out.append(p)
        seen.add(p)
    return out


def collect_corpus(output: Path) -> dict[str, list[Path]]:
    """Deterministically pick a diverse ~60-PDF corpus."""

    seen: set = set()
    buckets: dict[str, Bucket] = {}

    def add_bucket(name: str, target: int) -> Bucket:
        b = Bucket(name=name, target=target)
        buckets[name] = b
        return b

    single = add_bucket("single_column", 10)
    diverse = _walk_pdfs(TESTS_ROOT / "pdfs" / "diverse")
    theses = _walk_pdfs(TESTS_ROOT / "pdfs" / "theses")
    text_heavy = _walk_pdfs(TESTS_ROOT / "pdfs" / "text_heavy")
    single_seeds: list[Path] = []
    for p in diverse + theses + text_heavy:
        name = p.name.lower()
        if any(key in name for key in ("rfc", "thesis", "gdpr", "apollo", "nasa")):
            single_seeds.append(p)
    single.candidates = single_seeds + diverse + theses + text_heavy
    single.picked = _first_n(single.candidates, single.target, seen)

    multi = add_bucket("multi_column", 10)
    academic = _walk_pdfs(TESTS_ROOT / "pdfs" / "academic")
    technical = _walk_pdfs(TESTS_ROOT / "pdfs" / "technical")
    arxiv_elsewhere = [p for p in _walk_pdfs(TESTS_ROOT / "pdfs") if "arxiv" in p.name.lower()]
    multi.candidates = academic + technical + arxiv_elsewhere
    multi.picked = _first_n(multi.candidates, multi.target, seen)

    ds = add_bucket("datasheet_form", 10)
    repro_orafol = sorted(REPRO_ROOT.glob("orafol_*.pdf"))
    forms = _walk_pdfs(TESTS_ROOT / "pdfs" / "forms")
    tables = _walk_pdfs(TESTS_ROOT / "pdfs" / "tables")
    irs = _walk_pdfs(TESTS_ROOT / "irs")
    ds.candidates = repro_orafol + forms + tables + irs
    ds.picked = _first_n(ds.candidates, ds.target, seen)

    cjk = add_bucket("cjk_complex", 10)
    cn_repro = sorted(REPRO_ROOT.glob("cn_*.pdf")) + sorted(
        REPRO_ROOT.glob("cancer_lab_tests_zh.pdf")
    )
    multilingual = _walk_pdfs(TESTS_ROOT / "pdfs" / "multilingual")
    diverse_cjk = [
        p
        for p in diverse
        if any(tok in p.name.lower() for tok in ("cjk", "zh", "chinese", "japan", "kor", "cn_"))
    ]
    cjk.candidates = cn_repro + diverse_cjk + multilingual
    cjk.picked = _first_n(cjk.candidates, cjk.target, seen)

    pdfjs = add_bucket("encrypted_pdfjs", 10)
    pdfjs.candidates = _walk_pdfs(TESTS_ROOT / "pdfs_pdfjs")
    pdfjs.picked = _first_n(pdfjs.candidates, pdfjs.target, seen)

    rnd = add_bucket("random_pdfs", 10)
    all_pdfs = _walk_pdfs(TESTS_ROOT / "pdfs")
    rng = random.Random(0xC0FFEE)
    shuffled = list(all_pdfs)
    rng.shuffle(shuffled)
    rnd.candidates = shuffled
    rnd.picked = _first_n(rnd.candidates, rnd.target, seen)

    peer_map = {
        "single_column": ["random_pdfs", "multi_column", "datasheet_form"],
        "multi_column": ["random_pdfs", "single_column", "datasheet_form"],
        "datasheet_form": ["random_pdfs", "multi_column", "single_column"],
        "cjk_complex": ["random_pdfs", "single_column", "multi_column"],
        "encrypted_pdfjs": ["random_pdfs", "multi_column", "single_column"],
        "random_pdfs": ["multi_column", "single_column", "datasheet_form"],
    }
    for bname, bucket in buckets.items():
        if len(bucket.picked) >= bucket.target:
            continue
        for peer_name in peer_map[bname]:
            if len(bucket.picked) >= bucket.target:
                break
            peer = buckets[peer_name]
            need = bucket.target - len(bucket.picked)
            extras = _first_n(peer.candidates, need, seen)
            bucket.picked.extend(extras)

    output.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for bucket in buckets.values():
        for p in bucket.picked:
            lines.append(f"{bucket.name}\t{p}")
    output.write_text("\n".join(lines) + "\n")

    return {b.name: b.picked for b in buckets.values()}


def load_corpus(path: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            bucket, p = line.split("\t", 1)
        else:
            bucket, p = "unknown", line
        entries.append((bucket, Path(p)))
    return entries


# ---------------------------------------------------------------------------
# Running extractors
# ---------------------------------------------------------------------------


def _relative_key(pdf: Path) -> str:
    """Produce a deterministic, filesystem-safe path for saving output."""
    try:
        rel = pdf.resolve().relative_to(TESTS_ROOT.resolve())
        return str(Path("tests") / rel)
    except Exception:
        pass
    try:
        rel = pdf.resolve().relative_to(REPRO_ROOT.resolve())
        return str(Path("repro") / rel)
    except Exception:
        pass
    digest = hashlib.sha1(str(pdf).encode("utf-8")).hexdigest()[:12]
    return str(Path("other") / f"{digest}_{pdf.name}")


DIAGNOSTIC_STRINGS = [
    "ORALITE",
    "Commercial Grade",
    "Premium Grade",
    "Datasheet",
    "Page 1 of",
    "rowspan",
    "colspan",
    "\ufeff",  # BOM
]


def _count_lines(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _diagnostics(text: str) -> dict[str, int]:
    return {s: text.count(s) for s in DIAGNOSTIC_STRINGS if s in text}


def run_all(
    corpus: list[tuple[str, Path]],
    out_dir: Path,
    pages: int,
    force: bool,
    formats: Sequence[str],
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_specs = build_extractor_specs()

    # Group specs by format, honouring the --formats filter.
    specs_by_format: dict[str, list[ExtractorSpec]] = {f: [] for f in formats}
    for spec in all_specs:
        if spec.format in specs_by_format:
            specs_by_format[spec.format].append(spec)

    manifest: dict = {
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "out_dir": str(out_dir),
        "pages": pages,
        "v0323_examples": str(V0323_EXAMPLES),
        "head_examples": str(HEAD_EXAMPLES),
        "requested_formats": list(formats),
        "formats": {},
    }

    # Precompute extractor availability / reasons for each format.
    for fmt in formats:
        fmt_specs = specs_by_format[fmt]
        ext_meta: dict[str, dict] = {}
        for spec in fmt_specs:
            if spec.available():
                ext_meta[spec.name] = {"status": "available"}
            else:
                ext_meta[spec.name] = {
                    "status": "unavailable",
                    "reason": spec.unavail_reason(),
                }
        manifest["formats"][fmt] = {
            "extractors": ext_meta,
            "pdfs": {},
        }
        unavailable = [
            f"{n}.{fmt} ({m['reason']})"
            for n, m in ext_meta.items()
            if m["status"] == "unavailable"
        ]
        if unavailable:
            print(f"[warn] unavailable for {fmt}: {', '.join(unavailable)}", file=sys.stderr)

    total = len(corpus)
    for idx, (bucket, pdf) in enumerate(corpus, start=1):
        rel_key = _relative_key(pdf)
        print(f"[{idx:>3}/{total}] {bucket:<16} {pdf}", flush=True)

        exists = pdf.exists()
        size = pdf.stat().st_size if exists else 0

        for fmt in formats:
            fmt_specs = specs_by_format[fmt]
            pdf_results: dict[str, dict] = {
                "bucket": bucket,
                "pdf": str(pdf),
                "rel": rel_key,
                "exists": exists,
                "size": size,
                "results": {},
            }
            if not exists:
                pdf_results["error"] = "pdf missing"
                manifest["formats"][fmt]["pdfs"][str(pdf)] = pdf_results
                continue

            for spec in fmt_specs:
                ext_meta = manifest["formats"][fmt]["extractors"][spec.name]
                out_path = out_dir / f"{spec.name}.{spec.format}" / (rel_key + spec.file_ext)
                out_path.parent.mkdir(parents=True, exist_ok=True)

                if ext_meta["status"] == "unavailable":
                    pdf_results["results"][spec.name] = {
                        "status": "unavailable",
                        "reason": ext_meta["reason"],
                        "ok": False,
                        "bytes": 0,
                        "lines": 0,
                        "elapsed": 0.0,
                        "out_path": str(out_path),
                    }
                    continue

                if out_path.exists() and not force:
                    try:
                        text = out_path.read_text(encoding="utf-8", errors="replace")
                        err = None
                    except Exception as e:
                        text = ""
                        err = f"reread error: {e}"
                    status = "ok" if err is None else "error"
                    pdf_results["results"][spec.name] = {
                        "status": status,
                        "ok": err is None,
                        "returncode": 0,
                        "elapsed": 0.0,
                        "bytes": len(text.encode("utf-8")),
                        "lines": _count_lines(text),
                        "out_path": str(out_path),
                        "error": err,
                        "cached": True,
                        "diagnostics": _diagnostics(text),
                    }
                    continue

                t0 = time.time()
                ok, rc, text, err = spec.runner(pdf, pages)
                elapsed = time.time() - t0
                try:
                    out_path.write_text(text or "", encoding="utf-8")
                except Exception as e:
                    err = f"write error: {e}" if err is None else f"{err}; write error: {e}"
                status = "ok" if ok and err is None else "error"
                if err == "timeout":
                    status = "timeout"
                pdf_results["results"][spec.name] = {
                    "status": status,
                    "ok": ok and err is None,
                    "returncode": rc,
                    "elapsed": round(elapsed, 3),
                    "bytes": len((text or "").encode("utf-8")),
                    "lines": _count_lines(text or ""),
                    "out_path": str(out_path),
                    "error": err,
                    "cached": False,
                    "diagnostics": _diagnostics(text or ""),
                }

            manifest["formats"][fmt]["pdfs"][str(pdf)] = pdf_results

    # Backwards-compat: if only text was requested, expose the old
    # top-level shape so downstream tooling that parses manifest.json
    # still keeps working.
    if list(formats) == ["text"]:
        text_fmt = manifest["formats"]["text"]
        manifest["extractors"] = [
            n for n, m in text_fmt["extractors"].items() if m["status"] == "available"
        ]
        manifest["missing_extractors"] = [
            n for n, m in text_fmt["extractors"].items() if m["status"] != "available"
        ]
        manifest["pdfs"] = list(text_fmt["pdfs"].values())

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[done] manifest: {manifest_path}")
    return manifest


# ---------------------------------------------------------------------------
# Diff / groundtruth analysis
# ---------------------------------------------------------------------------


_WORD_RE = re.compile(r"[\w/\-\.@]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _distinctive(tokens: Iterable[str]) -> set:
    out: set = set()
    for tok in tokens:
        if len(tok) > 6 or any(ch.isdigit() for ch in tok):
            out.add(tok)
    return out


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _read_result_text(res: dict | None) -> str:
    if not res:
        return ""
    path = Path(res.get("out_path", ""))
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _format_section(manifest: dict, fmt: str) -> dict | None:
    """Return the per-format section of the manifest for ``fmt``.

    Handles both the new per-format layout and the legacy flat layout
    (which is implicitly the "text" format).
    """
    formats = manifest.get("formats")
    if isinstance(formats, dict) and fmt in formats:
        return formats[fmt]
    if fmt == "text" and "pdfs" in manifest:
        legacy_pdfs = manifest.get("pdfs", [])
        pdfs_map: dict[str, dict] = {}
        for entry in legacy_pdfs:
            pdfs_map[entry["pdf"]] = entry
        return {
            "extractors": {n: {"status": "available"} for n in manifest.get("extractors", [])},
            "pdfs": pdfs_map,
        }
    return None


def _iter_pdf_entries(section: dict) -> list[dict]:
    pdfs = section.get("pdfs", {})
    if isinstance(pdfs, dict):
        return list(pdfs.values())
    if isinstance(pdfs, list):
        return list(pdfs)
    return []


def cmd_diff(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    fmt = args.format
    section = _format_section(manifest, fmt)
    if section is None:
        print(f"No section for format {fmt!r} in {run_dir}/manifest.json", file=sys.stderr)
        return 2
    base = args.baseline
    head = args.head
    rows: list[tuple[float, dict]] = []
    total_base_bytes = 0
    total_head_bytes = 0
    errors = 0
    flagged: list[dict] = []

    base_avail = section.get("extractors", {}).get(base, {}).get("status") == "available"
    head_avail = section.get("extractors", {}).get(head, {}).get("status") == "available"
    if not base_avail:
        reason = section.get("extractors", {}).get(base, {}).get("reason", "?")
        print(f"[warn] baseline {base}.{fmt} unavailable: {reason}", file=sys.stderr)
    if not head_avail:
        reason = section.get("extractors", {}).get(head, {}).get("reason", "?")
        print(f"[warn] head {head}.{fmt} unavailable: {reason}", file=sys.stderr)

    for entry in _iter_pdf_entries(section):
        res = entry.get("results", {})
        if base not in res or head not in res:
            continue
        if res[base].get("status") == "unavailable" or res[head].get("status") == "unavailable":
            continue
        base_text = _read_result_text(res[base])
        head_text = _read_result_text(res[head])
        base_tokens = _tokenize(base_text)
        head_tokens = _tokenize(head_text)
        base_set = set(base_tokens)
        head_set = set(head_tokens)
        j = _jaccard(base_set, head_set)
        base_distinct = _distinctive(base_set)
        head_distinct = _distinctive(head_set)
        missing = sorted(base_distinct - head_distinct)[:20]

        row = {
            "pdf": entry["pdf"],
            "bucket": entry.get("bucket", "?"),
            "jaccard": j,
            "byte_delta": res[head]["bytes"] - res[base]["bytes"],
            "line_delta": res[head]["lines"] - res[base]["lines"],
            "base_bytes": res[base]["bytes"],
            "head_bytes": res[head]["bytes"],
            "base_err": res[base].get("error"),
            "head_err": res[head].get("error"),
            "missing_from_head": missing,
        }
        rows.append((j, row))
        total_base_bytes += res[base]["bytes"]
        total_head_bytes += res[head]["bytes"]
        if res[base].get("error") or res[head].get("error"):
            errors += 1
        if j < 0.90:
            flagged.append(row)

    rows.sort(key=lambda r: r[0])

    print(f"Regression diff [{fmt}]: {head} vs {base}")
    print(f"Run dir:   {run_dir}")
    print(f"PDFs:      {len(rows)}")
    print(f"Errors:    {errors}")
    print(f"Baseline total bytes: {total_base_bytes}")
    print(f"Head     total bytes: {total_head_bytes}")
    print(f"Delta:              {total_head_bytes - total_base_bytes:+d}")
    print()
    print(f"{'jaccard':>7} {'dB':>8} {'dL':>6}  bucket            pdf")
    print("-" * 100)
    for _, row in rows:
        flag = "!" if row["jaccard"] < 0.90 else " "
        print(
            f"{flag}{row['jaccard']:6.3f} {row['byte_delta']:+8d} {row['line_delta']:+6d}  "
            f"{row['bucket']:<16} {row['pdf']}"
        )

    if flagged:
        print()
        print(f"Flagged regressions (<0.90 jaccard): {len(flagged)}")
        for row in flagged[:30]:
            print(f"  {row['pdf']}")
            if row["missing_from_head"]:
                sample = ", ".join(row["missing_from_head"][:8])
                print(f"    missing_from_head: {sample}")
    return 0


def cmd_groundtruth(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    fmt = args.format
    section = _format_section(manifest, fmt)
    if section is None:
        print(f"No section for format {fmt!r} in {run_dir}/manifest.json", file=sys.stderr)
        return 2
    ref = args.ref
    actual = args.actual

    rows: list[tuple[float, dict]] = []
    for entry in _iter_pdf_entries(section):
        res = entry.get("results", {})
        if ref not in res or actual not in res:
            continue
        if res[ref].get("status") == "unavailable" or res[actual].get("status") == "unavailable":
            continue
        ref_text = _read_result_text(res[ref])
        act_text = _read_result_text(res[actual])
        ref_tokens = set(_tokenize(ref_text))
        act_tokens = set(_tokenize(act_text))
        j = _jaccard(ref_tokens, act_tokens)
        ref_distinct = _distinctive(ref_tokens)
        act_distinct = _distinctive(act_tokens)
        missing = sorted(ref_distinct - act_distinct)[:20]
        rows.append(
            (
                j,
                {
                    "pdf": entry["pdf"],
                    "bucket": entry.get("bucket", "?"),
                    "jaccard": j,
                    "ref_bytes": res[ref]["bytes"],
                    "actual_bytes": res[actual]["bytes"],
                    "missing_from_actual": missing,
                },
            )
        )
    rows.sort(key=lambda r: r[0])

    print(f"Groundtruth [{fmt}]: {actual} vs ref={ref}")
    print(f"Run dir:   {run_dir}")
    print(f"PDFs:      {len(rows)}")
    print()
    print(f"{'jaccard':>7}  bucket            pdf")
    print("-" * 100)
    for _, row in rows:
        flag = "!" if row["jaccard"] < 0.70 else " "
        print(f"{flag}{row['jaccard']:6.3f}  {row['bucket']:<16} {row['pdf']}")

    bad = [r for _, r in rows if r["jaccard"] < 0.70]
    if bad:
        print()
        print(f"Flagged (<0.70 jaccard vs {ref}): {len(bad)}")
        for row in bad[:30]:
            sample = ", ".join(row["missing_from_actual"][:8])
            print(f"  {row['pdf']}")
            if sample:
                print(f"    missing_from_actual: {sample}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    fmt = args.format
    section = _format_section(manifest, fmt)
    if section is None:
        print(f"No section for format {fmt!r} in {run_dir}/manifest.json", file=sys.stderr)
        return 2
    target = args.pdf
    hit: dict | None = None
    for entry in _iter_pdf_entries(section):
        if entry["pdf"] == target or Path(entry["pdf"]).name == target:
            hit = entry
            break
    if hit is None:
        print(
            f"No entry matching {target!r} in {run_dir}/manifest.json (format {fmt})",
            file=sys.stderr,
        )
        return 1

    extractors = list(section.get("extractors", {}).keys()) or [
        "v0323",
        "head",
        "pdftotext",
        "pypdfium2",
    ]
    print(f"PDF:    {hit['pdf']}")
    print(f"Bucket: {hit.get('bucket', '?')}")
    print(f"Size:   {hit.get('size', 0)} bytes")
    print(f"Format: {fmt}")
    print()
    for ext in extractors:
        res = hit.get("results", {}).get(ext)
        if res is None:
            continue
        text = _read_result_text(res)
        header = (
            f"===== {ext}.{fmt} | status={res.get('status', '?')} "
            f"ok={res.get('ok')} bytes={res.get('bytes')} lines={res.get('lines')} "
            f"elapsed={res.get('elapsed')}s ====="
        )
        print(header)
        if res.get("error"):
            print(f"[error] {res['error']}")
        print(text.rstrip("\n"))
        print()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_collect(args: argparse.Namespace) -> int:
    out = Path(args.output)
    picks = collect_corpus(out)
    total = sum(len(v) for v in picks.values())
    print(f"Corpus file: {out}")
    print(f"Total PDFs:  {total}")
    for bucket, entries in picks.items():
        print(f"  {bucket:<16} {len(entries):>3}")
    return 0


def _parse_formats(raw: str) -> list[str]:
    out: list[str] = []
    for tok in raw.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok not in FORMATS:
            raise argparse.ArgumentTypeError(f"unknown format {tok!r}; choose from {FORMATS}")
        if tok not in out:
            out.append(tok)
    if not out:
        raise argparse.ArgumentTypeError("no formats specified")
    return out


def cmd_run(args: argparse.Namespace) -> int:
    corpus_file = Path(args.corpus)
    if not corpus_file.exists():
        print(f"corpus file {corpus_file} does not exist; run `collect` first", file=sys.stderr)
        return 2
    corpus = load_corpus(corpus_file)
    if args.out is None:
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = RUNS_ROOT_DEFAULT / stamp
    else:
        out_dir = Path(args.out)
    formats = _parse_formats(args.formats)
    run_all(corpus, out_dir, pages=args.pages, force=args.force, formats=formats)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_collect = sub.add_parser("collect", help="Build the regression corpus file")
    p_collect.add_argument("--output", default=str(CORPUS_FILE_DEFAULT))
    p_collect.set_defaults(func=cmd_collect)

    p_run = sub.add_parser("run", help="Run every (extractor, format) combo on the corpus")
    p_run.add_argument("--corpus", default=str(CORPUS_FILE_DEFAULT))
    p_run.add_argument("--out", default=None)
    p_run.add_argument(
        "--pages",
        type=int,
        default=3,
        help="Pages per PDF for pdftotext/pypdfium2/pymupdf4llm (rust extractors always dump the whole doc). -1 for all.",
    )
    p_run.add_argument(
        "--force", action="store_true", help="Re-run extractors even if output files exist"
    )
    p_run.add_argument(
        "--formats",
        default=",".join(FORMATS),
        help="Comma-separated subset of {text,markdown,html}",
    )
    p_run.set_defaults(func=cmd_run)

    p_diff = sub.add_parser("diff", help="Compare head vs baseline from a run directory")
    p_diff.add_argument("--run", required=True)
    p_diff.add_argument("--baseline", default="v0323")
    p_diff.add_argument("--head", default="head")
    p_diff.add_argument("--format", default="text", choices=list(FORMATS))
    p_diff.set_defaults(func=cmd_diff)

    p_gt = sub.add_parser(
        "groundtruth", help="Compare an extractor against a reference (default pdftotext)"
    )
    p_gt.add_argument("--run", required=True)
    p_gt.add_argument("--ref", default="pdftotext")
    p_gt.add_argument("--actual", default="head")
    p_gt.add_argument("--format", default="text", choices=list(FORMATS))
    p_gt.set_defaults(func=cmd_groundtruth)

    p_show = sub.add_parser("show", help="Print extractors' output for a single PDF")
    p_show.add_argument("--run", required=True)
    p_show.add_argument("--pdf", required=True, help="Full PDF path or bare filename")
    p_show.add_argument("--format", default="text", choices=list(FORMATS))
    p_show.set_defaults(func=cmd_show)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

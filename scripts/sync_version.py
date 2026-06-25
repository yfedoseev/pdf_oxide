#!/usr/bin/env python3
"""Single-source version sync for pdf_oxide and ALL language bindings.

The canonical version lives in the workspace `Cargo.toml` ([package] version).
This script propagates it into every binding manifest + version/parity assert so
the release version is changed in exactly ONE place.

Usage:
  scripts/sync_version.py            # sync all bindings to Cargo.toml's version
  scripts/sync_version.py --check    # verify all are in sync (exit 1 on drift)
  scripts/sync_version.py --set X.Y.Z  # set Cargo.toml then sync everything

Idempotent. Run from the repo root (or anywhere — paths resolve to the repo).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def canonical_version() -> str:
    text = (ROOT / "Cargo.toml").read_text()
    # First `version = "..."` after [package].
    m = re.search(r"\[package\].*?^version\s*=\s*\"([^\"]+)\"", text, re.S | re.M)
    if not m:
        m = re.search(r"^version\s*=\s*\"([^\"]+)\"", text, re.M)
    if not m:
        sys.exit("could not read [package] version from Cargo.toml")
    return m.group(1)


def set_cargo_version(ver: str) -> None:
    path = ROOT / "Cargo.toml"
    text = path.read_text()
    new = re.sub(
        r"(\[package\].*?^version\s*=\s*\")[^\"]+(\")",
        rf"\g<1>{ver}\g<2>",
        text,
        count=1,
        flags=re.S | re.M,
    )
    path.write_text(new)


# Each rule: (relative path, regex with one capture group around the version).
# The capture group is replaced with the canonical version. Missing files are
# skipped (a binding may not exist yet).
def rules() -> list[tuple[str, str, bool]]:
    # (path, regex, multi) — multi=True replaces EVERY occurrence (a few files
    # carry the version more than once with the SAME pattern). Default first-
    # match-only, important for files like pom.xml whose later <version> tags are
    # dependency versions that must NOT be touched.
    return [
        # ── workspace sub-crates (package version + path-dep version) ───────
        ("pdf_oxide_cli/Cargo.toml", r'(?m)^version\s*=\s*"([^"]+)"', False),
        ("pdf_oxide_cli/Cargo.toml", r'pdf_oxide = \{ version = "([^"]+)"', False),
        ("pdf_oxide_mcp/Cargo.toml", r'(?m)^version\s*=\s*"([^"]+)"', False),
        ("pdf_oxide_mcp/Cargo.toml", r'pdf_oxide = \{ version = "([^"]+)"', False),
        ("pdf_oxide_jni/Cargo.toml", r'(?m)^version\s*=\s*"([^"]+)"', False),
        ("pdf_oxide_jni/Cargo.toml", r'pdf_oxide = \{ version = "([^"]+)"', False),
        # ── existing bindings ──────────────────────────────────────────────
        ("pyproject.toml", r'(?m)^version\s*=\s*"([^"]+)"', False),
        ("python/tests/test_core_parity.py", r'pdf_oxide\.VERSION == "([^"]+)"', False),
        ("tests/core_parity.rs", r'env!\("CARGO_PKG_VERSION"\),\s*"([^"]+)"', False),
        ("js/package.json", r'"version":\s*"([^"]+)"', False),
        ("wasm-pkg/package.json", r'"version":\s*"([^"]+)"', False),
        ("csharp/PdfOxide/PdfOxide.csproj", r"<Version>([^<]+)</Version>", False),
        ("java/pom.xml", r"<version>([^<]+)</version>", False),  # project version (first)
        ("java/pom.xml", r"<tag>v([^<]+)</tag>", False),
        ("java/pom.xml", r"(?m)^  version:\s+([0-9][^\s]+)", False),  # header comment
        ("go/cmd/install/main.go", r'fallbackVersion\s*=\s*"([^"]+)"', False),
        ("ruby/lib/pdf_oxide/version.rb", r"VERSION\s*=\s*'([^']+)'", False),
        ("ruby/spec/cdylib_smoke_spec.rb", r"PdfOxide::VERSION\)\.to eq\('([^']+)'\)", False),
        ("ruby/spec/core_parity_spec.rb", r"exposes version ([0-9][^'\s]+)", False),
        ("ruby/spec/core_parity_spec.rb", r"PdfOxide::VERSION\)\.to eq\('([^']+)'\)", False),
        ("php/src/Pdf.php", r"const VERSION\s*=\s*'([^']+)'", False),
        (
            "php/tests/Integration/CoreParityTest.php",
            r"assertSame\('([^']+)', Pdf::VERSION\)",
            False,
        ),
        ("php/scripts/download-native-lib.php", r"PACKAGE_VERSION_DEFAULT\s*=\s*'v([^']+)'", False),
        ("php/scripts/download-native-lib.php", r"pdf_oxide-php-installer/([0-9][^']+)'", True),
        # ── v0.3.68 new bindings ───────────────────────────────────────────
        ("dart/pubspec.yaml", r"(?m)^version:\s*([0-9][^\s]+)", False),
        ("r/DESCRIPTION", r"(?m)^Version:\s*([0-9][^\s]+)", False),
        ("julia/Project.toml", r'(?m)^version\s*=\s*"([^"]+)"', False),
        ("kotlin/build.gradle.kts", r'(?m)^version\s*=\s*"([^"]+)"', False),
        ("scala/build.sbt", r'ThisBuild / version := "([^"]+)"', False),
        # JVM facades depend on the Java binding artifact at the same version —
        # keep these dependency coordinates in lock-step.
        ("kotlin/build.gradle.kts", r'fyi\.oxide:pdf-oxide:([0-9][^"]+)"', False),
        ("scala/build.sbt", r'"fyi\.oxide" % "pdf-oxide" % "([0-9][^"]+)"', False),
        ("clojure/deps.edn", r'fyi\.oxide/pdf-oxide \{:mvn/version "([^"]+)"', False),
        ("zig/build.zig.zon", r'\.version\s*=\s*"([^"]+)"', False),
        ("cpp/CMakeLists.txt", r"project\(pdf_oxide_cpp VERSION ([0-9][^\s]+)", False),
        ("elixir/mix.exs", r'version:\s*"([^"]+)"', False),
        ("objc/include/POXPdfOxide.h", r'POX_PDF_OXIDE_VERSION "([^"]+)"', False),
        ("objc/PdfOxide.podspec", r"spec\.version\s*=\s*'([^']+)'", False),
        # uv lockfile — the editable pdf-oxide package entry (matches only the
        # pdf-oxide block, not the hundreds of other package version lines).
        ("uv.lock", r'name = "pdf-oxide"\nversion = "([^"]+)"', False),
    ]


def apply(check: bool) -> int:
    ver = canonical_version()
    drift, changed, missing = [], [], []
    for rel, pat, multi in rules():
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
            continue
        text = path.read_text()
        matches = list(re.finditer(pat, text))
        if not matches:
            drift.append(f"{rel}: version pattern not found  ({pat})")
            continue
        # First-match-only by default; `multi` handles patterns that legitimately
        # recur (e.g. the php installer user-agent string), while keeping
        # first-only for files like pom.xml whose later matches are dep versions.
        if not multi:
            matches = matches[:1]
        stale = [m for m in matches if m.group(1) != ver]
        if not stale:
            continue
        if check:
            for m in stale:
                drift.append(f"{rel}: {m.group(1)} != {ver}")
        else:
            # Rebuild right-to-left so earlier spans stay valid.
            for m in reversed(stale):
                s, e = m.span(1)
                text = text[:s] + ver + text[e:]
            path.write_text(text)
            changed.append(f"{rel}: -> {ver} ({len(stale)} occurrence(s))")

    print(f"canonical version (Cargo.toml): {ver}")
    if missing:
        print("skipped (absent): " + ", ".join(missing))
    if check:
        if drift:
            print("OUT OF SYNC:")
            for d in drift:
                print("  " + d)
            return 1
        print("all bindings in sync ✓")
        return 0
    for c in changed:
        print("  " + c)
    print(f"synced {len(changed)} file(s) to {ver}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify, exit 1 on drift")
    ap.add_argument("--set", metavar="X.Y.Z", help="set Cargo.toml version then sync")
    args = ap.parse_args()
    if args.set:
        set_cargo_version(args.set)
        print(f"set Cargo.toml version -> {args.set}")
    return apply(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Preprocess pdf_oxide_c/pdf_oxide.h for PHP FFI::cdef() compatibility.

PHP FFI's C parser is restrictive; this script strips:
- Block + line comments
- ALL #-preprocessor lines (PHP FFI cannot evaluate them)
- `extern "C" { ... }` wrappers (we keep the contained decls)
- The PDF_OXIDE_TARGET_WASM32 guarded blocks: we drop the `#if !defined(...)`
  guard and keep the body unconditionally (PHP runs on host platforms only).
- `<stdarg.h>` style preprocessor includes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


# Repo-relative defaults: php/scripts/preprocess_header.py → repo root is parent-parent.
_REPO = Path(__file__).resolve().parent.parent.parent
SRC = _REPO / "include" / "pdf_oxide_c" / "pdf_oxide.h"
DST = _REPO / "php" / "include" / "pdf_oxide.h"


def strip_block_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def strip_line_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        # naive but adequate (no // inside strings in this header)
        idx = line.find("//")
        if idx >= 0:
            line = line[:idx].rstrip()
        out.append(line)
    return "\n".join(out)


def strip_preprocessor(text: str) -> str:
    out = []
    for line in text.splitlines():
        # whitespace-leading + #
        if re.match(r"^\s*#", line):
            continue
        out.append(line)
    return "\n".join(out)


def strip_extern_c(text: str) -> str:
    # Remove `extern "C" {`  and the matching `}` near EOF.
    # We can't bracket-match in regex; do textual replace.
    text = re.sub(r'extern\s+"C"\s*\{', "", text)
    return text


def collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", text)


def main():
    raw = SRC.read_text()
    t = strip_block_comments(raw)
    t = strip_line_comments(t)
    t = strip_preprocessor(t)
    t = strip_extern_c(t)
    t = collapse_blank_lines(t)
    # Drop any orphan closing brace from extern "C"
    # extern blocks pair: { (now removed) ... } — orphan }.
    # We'll remove single } on its own line at top-level by matching balance.
    # Simpler: re-balance braces only top-level.
    lines = t.splitlines()
    depth = 0
    cleaned = []
    for line in lines:
        # count braces ignoring inside strings/comments (already stripped)
        opens = line.count("{")
        closes = line.count("}")
        # If a line is purely "}" and depth == 0 it's an orphan from extern "C"
        stripped = line.strip()
        if stripped == "}" and depth == 0:
            continue
        cleaned.append(line)
        depth += opens - closes
        if depth < 0:
            depth = 0
    out = "\n".join(cleaned)
    DST.write_text(out)
    # Count function decls in output: lines ending with `);` at brace depth 0
    fns = 0
    depth = 0
    for line in out.splitlines():
        opens = line.count("{")
        closes = line.count("}")
        if (
            depth == 0
            and line.rstrip().endswith(");")
            and "typedef" not in line
            and "(*" not in line
        ):
            fns += 1
        depth += opens - closes
    print(
        f"Wrote {DST} ({len(out)} bytes, {len(out.splitlines())} lines, ~{fns} fn decls)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

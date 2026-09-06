#!/usr/bin/env python3
"""Verify that pdf_oxide's .pyi stub and the installed module export the
same public symbols, in both directions.

Exit 1 if any stub symbol is absent from the module — stubs generated with
wider Cargo features than the installed wheel (issue #464) — or if any public
module export is absent from the stub.

Usage:
    python scripts/check_stub_parity.py <path-to-pyi>
"""

from __future__ import annotations

import ast
import importlib
import sys


def pyi_top_level_names(pyi_path: str) -> set[str]:
    """Return public names *defined* at the top level of a .pyi file.

    Deliberately excludes ImportFrom nodes: stubs routinely re-import
    stdlib helpers (``from __future__ import annotations``,
    ``from pathlib import Path``, ``from typing import ...``) purely for
    type-annotation purposes.  Those names are not module exports and
    should not be checked against ``dir(module)`` (issue #464).
    """
    with open(pyi_path, encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filename=pyi_path)
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("_"):
                    names.add(t.id)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and not node.target.id.startswith("_")
        ):
            names.add(node.target.id)
        # ImportFrom intentionally skipped: stdlib type-annotation imports
        # (pathlib.Path, __future__.annotations, typing.*) are not module
        # exports and must not be checked against dir(module).
    return names


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-pyi>", file=sys.stderr)
        return 2

    pyi_path = sys.argv[1]
    stub_names = pyi_top_level_names(pyi_path)

    mod = importlib.import_module("pdf_oxide.pdf_oxide")
    mod_names = set(dir(mod))

    missing = stub_names - mod_names
    if missing:
        print("FAIL: stub symbols missing from installed wheel:")
        for name in sorted(missing):
            print(f"  {name}")
        print(
            "\nThe stub was likely generated with broader Cargo features than the"
            " installed wheel. Fix: regenerate the stub with --features matching"
            " the release wheel (see rylai.toml)."
        )
        return 1

    # The other direction: a public symbol the module exports that the stub
    # does not declare is invisible to type checkers and editors, and this
    # check used to pass on it. Names the interpreter puts on every module are
    # not exports.
    module_boilerplate = {"__builtins__", "__doc__", "__file__", "__loader__",
                          "__name__", "__package__", "__spec__", "__all__",
                          "__path__", "__cached__"}
    extra = {n for n in mod_names - stub_names
             if not n.startswith("_") and n not in module_boilerplate}
    if extra:
        print("FAIL: module exports missing from the stub:")
        for name in sorted(extra):
            print(f"  {name}")
        print("\nRegenerate the stub (uvx rylai -o python/pdf_oxide/) and commit it.")
        return 1

    print(f"OK: all {len(stub_names)} stub symbols present in installed module,"
          " and every module export is declared in the stub.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

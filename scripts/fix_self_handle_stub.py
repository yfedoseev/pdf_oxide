#!/usr/bin/env python3
"""Strip leaked PyO3 ``Py<Self>`` receiver params from a rylai-generated stub.

rylai emits an implicit ``self`` for every ``#[pymethods]`` method, but it does
*not* recognise a receiver taken by value as the first parameter (e.g.
``fn page(slf_handle: Py<Self>, ...)`` or ``fn __iter__(slf: Py<Self>, ...)``).
Those receivers are how pyo3 hands a method an owned handle to its own
instance; pyo3 treats them as the receiver at runtime, but rylai re-emits them
as an *extra positional parameter* alongside the injected ``self``. The stub
then reads e.g. ``def page(self, slf_handle: DocumentBuilder, width, height)``,
so ``builder.page(w, h)`` (correct at runtime) trips mypy's ``call-arg`` /
``arg-type`` checks (issue: python-typing).

For ordinary methods we fix this in Rust with ``#[pyo3(signature = (...))]``,
which rylai honours. pyo3 *forbids* ``signature`` on magic methods
(``__iter__``, ``__getitem__``, ...), so those receivers still leak. This
post-processor — wired into rylai's ``format`` hook in ``rylai.toml`` so it runs
on every generation, CI included — removes them.

It deletes any parameter literally named ``slf`` or ``slf_handle``. Those are
Rust receiver-binding names and never a legitimate Python parameter, so the
transform is safe and also auto-covers any future ``Py<Self>`` method whose
signature attribute is missing or disallowed.

Usage (rylai appends the generated path):
    fix_self_handle_stub.py <path-to-.pyi>
"""

from __future__ import annotations

import ast
import sys


# Parameter names pyo3 code uses to bind an owned `Py<Self>` receiver. These are
# never valid Python parameter names, so removing them on sight is safe.
RECEIVER_NAMES = frozenset({"slf", "slf_handle"})


def _line_starts(src: str) -> list[int]:
    """Absolute char offset at which each (1-based) line begins."""
    offsets = [0]
    for line in src.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def fix(src: str) -> str:
    tree = ast.parse(src)
    starts = _line_starts(src)

    def off(lineno: int, col: int) -> int:
        return starts[lineno - 1] + col

    # Collect (start, end) char spans to delete, then apply right-to-left so
    # earlier offsets stay valid.
    cuts: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = node.args.args  # positional-or-keyword params, in order
        for i, arg in enumerate(params):
            if arg.arg not in RECEIVER_NAMES:
                continue
            ann = arg.annotation
            arg_start = off(arg.lineno, arg.col_offset)
            if i + 1 < len(params):
                # Not last: swallow this arg and the following ", ".
                nxt = params[i + 1]
                cuts.append((arg_start, off(nxt.lineno, nxt.col_offset)))
            elif i > 0:
                # Last arg: swallow the preceding ", " plus this arg, ending at
                # the annotation (or the name if somehow unannotated).
                prev = params[i - 1]
                prev_end = (
                    off(prev.annotation.end_lineno, prev.annotation.end_col_offset)
                    if prev.annotation is not None
                    else off(prev.lineno, prev.col_offset + len(prev.arg))
                )
                end = (
                    off(ann.end_lineno, ann.end_col_offset)
                    if ann is not None
                    else off(arg.lineno, arg.col_offset + len(arg.arg))
                )
                cuts.append((prev_end, end))
            else:
                # Sole parameter: just remove the arg itself.
                end = (
                    off(ann.end_lineno, ann.end_col_offset)
                    if ann is not None
                    else off(arg.lineno, arg.col_offset + len(arg.arg))
                )
                cuts.append((arg_start, end))

    for start, end in sorted(cuts, reverse=True):
        src = src[:start] + src[end:]
    return src


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-.pyi>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        src = f.read()
    fixed = fix(src)
    if fixed != src:
        with open(path, "w", encoding="utf-8") as f:
            f.write(fixed)
    return 0


if __name__ == "__main__":
    sys.exit(main())

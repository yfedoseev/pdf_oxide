"""
OfficeConverter must not hold the GIL while it waits for the document parser.

The Office parser runs each document on a helper thread and joins it, and that
thread logs through the Python `logging` bridge, which needs the GIL. A caller
that keeps the GIL while it waits for the join deadlocks against its own helper
thread: the process sits at ~0% CPU forever.

The conversions run in a child process with a timeout, so a deadlock fails this
test in seconds instead of hanging the whole test run.
"""

import subprocess
import sys
import textwrap

import pytest

from pdf_oxide import OfficeConverter, Pdf, PdfDocument


_TIMEOUT_SECONDS = 60

# The Office bytes are produced by the library itself from a PDF it built, so
# the fixture is synthetic and carries no third-party document.
_BYTES_CHILD = textwrap.dedent(
    """
    import sys
    from pdf_oxide import OfficeConverter, Pdf, PdfDocument

    fmt = sys.argv[1]
    pdf = Pdf.from_markdown("# Title\\n\\nBody text.").to_bytes()
    office = getattr(PdfDocument.from_bytes(pdf), f"to_{fmt}_bytes")()
    out = getattr(OfficeConverter, f"from_{fmt}_bytes")(office).to_bytes()
    assert out[:5] == b"%PDF-", out[:16]
    print("converted", fmt, len(out))
    """
)

_PATH_CHILD = textwrap.dedent(
    """
    import sys
    from pdf_oxide import OfficeConverter

    out = OfficeConverter.from_docx(sys.argv[1]).to_bytes()
    assert out[:5] == b"%PDF-", out[:16]
    print("converted path", len(out))
    """
)


def _run_child(code, arg, what):
    """Run `code` in a child interpreter; a timeout means the GIL deadlock."""
    try:
        return subprocess.run(
            [sys.executable, "-c", code, arg],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"{what} did not return within {_TIMEOUT_SECONDS}s: "
            "the GIL is held while the parser thread is joined"
        )


@pytest.mark.parametrize("fmt", ["docx", "pptx", "xlsx"])
def test_office_bytes_to_pdf_does_not_deadlock_against_the_parse_thread(fmt):
    result = _run_child(_BYTES_CHILD, fmt, f"OfficeConverter.from_{fmt}_bytes")
    assert result.returncode == 0, result.stderr
    assert f"converted {fmt}" in result.stdout


def test_office_path_to_pdf_does_not_deadlock_against_the_parse_thread(tmp_path):
    # The path entry points share the helper-thread parse with the bytes ones.
    pdf = Pdf.from_markdown("# Title\n\nBody text.").to_bytes()
    path = tmp_path / "in.docx"
    path.write_bytes(PdfDocument.from_bytes(pdf).to_docx_bytes())

    result = _run_child(_PATH_CHILD, str(path), "OfficeConverter.from_docx")
    assert result.returncode == 0, result.stderr
    assert "converted path" in result.stdout


def test_converter_returns_the_same_result_after_releasing_the_gil():
    # Releasing the GIL must not change what comes back.
    pdf = Pdf.from_markdown("# Title\n\nBody text.").to_bytes()
    docx = PdfDocument.from_bytes(pdf).to_docx_bytes()
    out = OfficeConverter.from_docx_bytes(docx).to_bytes()
    assert out[:5] == b"%PDF-"
    assert PdfDocument.from_bytes(out).page_count() >= 1

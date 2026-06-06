"""#632: onnxruntime shared-library name detection must cover the macOS
versioned ``.dylib`` form (``libonnxruntime.1.16.0.dylib``), where the version
precedes the extension — previously only the unversioned ``libonnxruntime.dylib``
matched, so installed onnxruntime was never found on macOS and OCR was skipped.

The matcher is pure string logic, so this runs on any OS."""

import pytest

from pdf_oxide import _is_ort_lib


@pytest.mark.parametrize(
    "name",
    [
        "libonnxruntime.so",  # Linux unversioned
        "libonnxruntime.so.1.20.1",  # Linux versioned (version after ext)
        "libonnxruntime.dylib",  # macOS unversioned
        "libonnxruntime.1.16.0.dylib",  # macOS versioned (#632 regression)
        "libonnxruntime.1.20.1.dylib",  # macOS versioned
        "onnxruntime.dll",  # Windows
    ],
)
def test_recognizes_onnxruntime_libraries(name):
    assert _is_ort_lib(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "libonnxruntime_providers_shared.dylib",  # provider lib, not the runtime
        "libonnxruntime_providers_shared.so",
        "libonnxruntime_providers_cuda.dylib",
        "libfoo.dylib",
        "onnxruntime.txt",
        "README",
        "",
    ],
)
def test_rejects_non_runtime_files(name):
    assert _is_ort_lib(name) is False

"""
Cross-binding API-parity tests.

These assert the Python package surfaces the same idiomatic API the
other bindings expose:

  - crypto-governance: set/get policy, inventory, CBOM
  - destructive redaction + standalone sanitize_document
  - PAdES signing + DSS read side + has_document_timestamp
  - split-by-bookmarks

The gap these guard against: the PAdES surface and crypto_cbom were
reachable only via the private `pdf_oxide.pdf_oxide` extension under
non-idiomatic ``py_``-prefixed names; they must now import straight
from the public ``pdf_oxide`` package.
"""

import json

import pytest

import pdf_oxide


# ── #230 crypto-governance ───────────────────────────────────────────────────


def test_crypto_governance_symbols_are_public():
    for name in (
        "crypto_set_policy",
        "crypto_policy",
        "crypto_inventory",
        "crypto_cbom",
    ):
        assert hasattr(pdf_oxide, name), f"{name} missing from pdf_oxide package"
        assert name in pdf_oxide.__all__, f"{name} missing from __all__"


def test_crypto_policy_default_and_inventory():
    # Default (never set) is "compat"; getter returns the grammar string.
    assert isinstance(pdf_oxide.crypto_policy(), str)
    assert isinstance(pdf_oxide.crypto_inventory(), list)


def test_crypto_cbom_is_cyclonedx_json():
    cbom = pdf_oxide.crypto_cbom()
    assert isinstance(cbom, str) and cbom
    doc = json.loads(cbom)  # must be valid JSON
    assert doc.get("bomFormat") == "CycloneDX"


# ── #235 PAdES signing / DSS / B-LTA ─────────────────────────────────────────


def test_pades_surface_is_public_and_idiomatic():
    # Idiomatic names (NOT the old py_ prefixed extension symbols).
    for name in (
        "sign_pdf_bytes",
        "sign_pdf_bytes_pades",
        "has_document_timestamp",
        "Certificate",
        "Signature",
        "PadesLevel",
        "RevocationMaterial",
        "Dss",
    ):
        assert hasattr(pdf_oxide, name), f"{name} not importable from pdf_oxide"
        assert name in pdf_oxide.__all__, f"{name} missing from __all__"
    # The non-idiomatic legacy names must NOT be the public surface.
    assert not hasattr(pdf_oxide, "py_sign_pdf_bytes_pades")


def test_pades_level_enum_mapping_is_frozen():
    # eq_int pyclass: members compare equal to their frozen integer code.
    lv = pdf_oxide.PadesLevel
    assert lv.B_B == 0
    assert lv.B_T == 1
    assert lv.B_LT == 2
    assert lv.B_LTA == 3


def test_has_document_timestamp_returns_bool_for_plain_pdf():
    pdf = pdf_oxide.Pdf.from_markdown("# parity\n\nplain document").to_bytes()
    result = pdf_oxide.has_document_timestamp(pdf)
    assert result is False  # a freshly-built PDF has no /DocTimeStamp


# ── #231 redaction + standalone sanitize ─────────────────────────────────────


def test_redaction_and_sanitize_methods_present():
    doc = pdf_oxide.PdfDocument.from_bytes(
        pdf_oxide.Pdf.from_markdown("# secret\n\nclassified body").to_bytes()
    )
    for m in (
        "add_redaction",
        "redaction_count",
        "apply_redactions_destructive",
        "sanitize_document",
    ):
        assert hasattr(doc, m), f"PdfDocument.{m} missing (#231)"


def test_sanitize_document_runs_and_reports():
    doc = pdf_oxide.PdfDocument.from_bytes(
        pdf_oxide.Pdf.from_markdown("# title\n\nbody").to_bytes()
    )
    report = doc.sanitize_document()
    # Report is a dict with the documented redaction-report keys.
    assert isinstance(report, dict)
    assert "annotations_removed" in report


# ── #482 split-by-bookmarks ──────────────────────────────────────────────────


def test_split_by_bookmarks_symbols_are_public():
    for name in ("plan_split_by_bookmarks", "split_by_bookmarks"):
        assert hasattr(pdf_oxide, name)
        assert name in pdf_oxide.__all__


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

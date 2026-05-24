// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.

import { describe, expect, it } from 'vitest';
import pdfOxide from '../index.js';

const { PdfDocument } = pdfOxide;

/**
 * Locks in the PDF/A wire-format integer mapping inside the JS
 * binding's validatePdfA / convertToPdfA helpers. The string-keyed
 * `levelMap` lives at js/src/index.ts:811 and js/src/index.ts:833 —
 * any future re-ordering of those tables silently flips the
 * cdylib-side meaning of a level.
 *
 * Cross-binding companion tests:
 *   - csharp/PdfOxide.Tests/PdfLevelWireFormatTests.cs
 *   - java/src/test/.../compliance/PdfLevelWireFormatTest.java
 *   - php/tests/Unit/PdfValidatorLevelMappingTest.php
 *   - ruby/spec/ffi_signature_regression_spec.rb (PDF/A + PDF/UA section)
 *
 * The JS binding does NOT expose the level integers directly — they
 * are internal to validatePdfA / convertToPdfA. We test the mapping
 * by introspecting the prototype's source text: an indirect probe,
 * but the alternative (a real `PdfDocument.open()` + native call)
 * needs a cdylib + fixture and only catches `1b` (validatePdfA's
 * default), missing the other seven levels.
 *
 * If the levelMap is ever extracted to an exported constant (cleaner
 * design), this test should switch to importing it directly.
 */
describe('PDF/A level wire-format mapping (parity with cdylib C ABI)', () => {
  const validatePdfASrc = PdfDocument.prototype.validatePdfA?.toString() ?? '';
  const convertToPdfASrc = PdfDocument.prototype.convertToPdfA?.toString() ?? '';

  function extractLevelMap(src) {
    // levelMap: Record<string, number> = { '1b': 0, ... }
    const m = src.match(/levelMap\s*[:=][^{]*\{([^}]*)\}/);
    if (!m) return null;
    const entries = {};
    for (const e of m[1].split(',')) {
      const kv = e.match(/['"](\w+)['"]\s*:\s*(\d+)/);
      if (kv) entries[kv[1]] = Number(kv[2]);
    }
    return entries;
  }

  // src/ffi.rs:1225 — `0=A1b 1=A1a 2=A2b 3=A2a 4=A2u 5=A3b 6=A3a 7=A3u`.
  const CANONICAL = {
    '1b': 0,
    '1a': 1,
    '2b': 2,
    '2a': 3,
    '2u': 4,
    '3b': 5,
    '3a': 6,
    '3u': 7,
  };

  it('validatePdfA uses the cdylib wire format', () => {
    const map = extractLevelMap(validatePdfASrc);
    expect(map).toEqual(CANONICAL);
  });

  it('convertToPdfA uses the cdylib wire format', () => {
    const map = extractLevelMap(convertToPdfASrc);
    expect(map).toEqual(CANONICAL);
  });

  it('B comes before A within each level (not alphabetical)', () => {
    const map = extractLevelMap(validatePdfASrc);
    expect(map['1b']).toBeLessThan(map['1a']);
    expect(map['2b']).toBeLessThan(map['2a']);
    expect(map['3b']).toBeLessThan(map['3a']);
  });
});

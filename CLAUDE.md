- PDF Spec compliant RUST parser

## PDF Specification Reference

The authoritative PDF specification is located at `docs/spec/pdf.md` (ISO 32000-1:2008 PDF 1.7).

**Key sections for text extraction:**
- Section 9: Text (fonts, text state, text objects)
- Section 9.10: Extraction of Text Content (ToUnicode CMaps, character mapping)
- Section 9.4: Text Objects (BT/ET, positioning operators Td, TD, Tm, T*)
- Section 9.3: Text State Parameters (Tc, Tw, Tz, TL, Tf, Tr, Ts)
- Section 14.7: Logical Structure (Tagged PDF structure trees)
- Section 14.8: Tagged PDF (semantic structure)

**Compliance Rules:**
1. All text extraction MUST follow Section 9.10 character-to-Unicode mapping priority
2. Word boundary detection should use TJ offset values (Section 9.4.4) and geometric positioning
3. Do NOT use linguistic heuristics (CamelCase, pattern matching) for word segmentation
4. Prefer Tagged PDF structure (Section 14.7) when available for reading order
5. Font metrics from PDF spec (Section 9.6-9.8) are acceptable for spacing calculations
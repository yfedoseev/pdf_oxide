#!/usr/bin/env python3
"""Generate a minimal, self-contained PDF that reproduces pdf_oxide issue #2130.

The bug: for a Type0 (CID) font whose ToUnicode CMap is PRESENT but does NOT
cover the code of a line-break hyphen, pdf_oxide emits U+FFFD (the replacement
character) instead of falling back to U+002D. This is exactly the behaviour
seen on the Elsevier *Waste Management* 2025 article (S0956053X25001680), where
inline hyphens extract fine but line-break hyphens become U+FFFD.

This fixture is intentionally tiny: a Type0 font whose ToUnicode CMap covers
every printable ASCII code *except* the single code we use for the line-break
hyphen (0x1E). The inline hyphen (0x2D) IS covered, so the asymmetry is
reproduced. No external/copyrighted font or document is involved.

Outputs: issue_2130_linebreak_hyphen.pdf
"""
import os

OUT = os.path.join(os.path.dirname(__file__),
                   "internal", "deepdoc", "parser", "pdf", "pdfoxide",
                   "testdata", "issue_2130_linebreak_hyphen.pdf")

# Code we use for the *line-break* hyphen: intentionally omitted from ToUnicode.
LB_HYPHEN = 0x1E
# Code we use for the *inline* hyphen: covered by ToUnicode -> U+002D.
INLINE_HYPHEN = 0x2D

# ---- Encoding CMap (custom stream -> encoding is NOT Identity-H/V) -----------
# 1:1 mapping code == CID for all of <00>..<FF>.
cid_entries = []
cid_entries.append("1 begincidrange")
cid_entries.append("<20> <7E> 32")  # code 0x20 -> CID 32 (== 0x20) ... 0x7E -> 0x7E
cid_entries.append("endcidrange")
cid_entries.append("1 begincidchar")
cid_entries.append("<1E> 30")        # LB hyphen code 0x1E -> CID 0x1E
cid_entries.append("endcidchar")

cmap_stream = (
    "/CMapName /MyRepro-H\n"
    "/CMapType 1\n"
    "/WMode 0\n"
    "1 begincodespacerange\n"
    "<00> <FF>\n"
    "endcodespacerange\n"
    + "\n".join(cid_entries) + "\n"
)

# ---- ToUnicode CMap: cover everything EXCEPT LB_HYPHEN -----------------------
bf = []
for code in range(0x20, 0x7F):
    if code == LB_HYPHEN:
        continue  # <- the deliberate omission that triggers U+FFFD
    bf.append("<%02X> <%04X>" % (code, code))
# Make sure the inline hyphen is explicitly covered (it's in 0x20..0x7F already).
to_unicode_stream = (
    "/CMapName /Adobe-Identity-UCS\n"
    "/CMapType 2\n"
    "/WMode 0\n"
    "1 begincodespacerange\n"
    "<00> <FF>\n"
    "endcodespacerange\n"
    "%d beginbfchar\n" % len(bf)
    + "\n".join(bf) + "\n"
    "endbfchar\n"
)

# ---- Content stream ----------------------------------------------------------
# "pre" + LB_HYPHEN on one line, "cipitation" on the next (line break).
# Then an inline-hyphen word "lithium-ion" whose hyphen extracts correctly.
content = (
    "BT\n"
    "/F1 12 Tf\n"
    "1 0 0 1 72 720 Tm\n"
    "(pre\\036) Tj\n"          # \036 octal == 0x1E == LB_HYPHEN
    "0 -18 Td\n"
    "(cipitation) Tj\n"
    "0 -18 Td\n"
    "(lithium\\055ion) Tj\n"   # \055 octal == 0x2D == INLINE_HYPHEN
    "ET\n"
)

objects = []
objects.append("<< /Type /Catalog /Pages 2 0 R >>")
objects.append("<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
objects.append(
    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
    "/Resources << /Font << /F1 4 0 R >> >> /Contents 6 0 R >>"
)
objects.append(
    "<< /Type /Font /Subtype /Type0 /BaseFont /MyRepro "
    "/Encoding 5 0 R /ToUnicode 9 0 R /DescendantFonts [7 0 R] >>"
)
objects.append("<< /Length %d >>\nstream\n%s\nendstream" % (len(cmap_stream), cmap_stream))
objects.append("<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content))
objects.append(
    "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /MyRepro "
    "/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
    "/CIDToGIDMap /Identity /FontDescriptor 8 0 R >>"
)
objects.append(
    "<< /Type /FontDescriptor /FontName /MyRepro /Flags 4 "
    "/FontBBox [0 0 0 0] /Ascent 1000 /Descent -200 /CapHeight 1000 "
    "/ItalicAngle 0 /StemV 80 >>"
)
objects.append("<< /Length %d >>\nstream\n%s\nendstream" % (len(to_unicode_stream), to_unicode_stream))

pdf = "%PDF-1.7\n"
offsets = [0]
for i, body in enumerate(objects, start=1):
    offsets.append(len(pdf.encode("latin-1")))
    pdf += "%d 0 obj\n%s\nendobj\n" % (i, body)

xref_pos = len(pdf.encode("latin-1"))
n = len(objects) + 1
xref = "xref\n0 %d\n" % n
xref += "0000000000 65535 f \n"
for off in offsets[1:]:
    xref += "%010d 00000 n \n" % off
pdf += xref
pdf += "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (n, xref_pos)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "wb") as f:
    f.write(pdf.encode("latin-1"))

print("wrote", OUT, "bytes=", len(pdf.encode("latin-1")))
print("LB_HYPHEN 0x%02X omitted from ToUnicode; INLINE_HYPHEN 0x%02X covered" %
      (LB_HYPHEN, INLINE_HYPHEN))

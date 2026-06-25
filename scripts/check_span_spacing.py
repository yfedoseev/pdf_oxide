#!/usr/bin/env python3
"""Check span extraction spacing: Tj operator buffering must not split words."""

import pdf_oxide


# Test problematic PDF (5PFVA6CO2FP66IJYJJ4YMWOLK5EHRCCD.pdf)
print("=" * 80)
print("TESTING PROBLEMATIC PDF")
print("=" * 80)

doc = pdf_oxide.PdfDocument("test_datasets/pdfs/mixed/5PFVA6CO2FP66IJYJJ4YMWOLK5EHRCCD.pdf")

# Extract spans (NEW - with buffering)
spans = doc.extract_spans(0)

print(f"\n Number of spans: {len(spans)}")

# Get first 300 chars of text
text = "".join([s.text for s in spans])
print(f"Total text length: {len(text)} characters")
print("\nFirst 300 chars:")
print(text[:300])
print()

# Check for problematic spacing pattern "F i s c a l"
if " F i s c a l " in text[:300]:
    print("❌ STILL BROKEN - text has spacing between characters")
    print("   Pattern found: ' F i s c a l '")
elif "Fiscal Year" in text[:300]:
    print("✅ FIXED - text is properly formatted")
    print("   Pattern found: 'Fiscal Year'")
else:
    print("⚠️  UNEXPECTED - neither pattern found")
    print("   Text sample:", text[:100])

print("\n" + "=" * 80)

#!/usr/bin/env python3
"""
OCR text extraction example using pdf_oxide.

This script demonstrates how to extract text from scanned PDFs using
PaddleOCR PP-OCRv5 models via ONNX Runtime.

Prerequisites:
    1. Build pdf_oxide with OCR feature:
       maturin develop --features python,ocr

    2. Download PaddleOCR models:
       - en_PP-OCRv5_det_infer.onnx  (detection model)
       - en_PP-OCRv5_rec_infer.onnx  (recognition model)
       - en_dict.txt                  (character dictionary)

Usage:
    python ocr_example.py <pdf_file> --det <det_model> --rec <rec_model> --dict <dict_file>

Example:
    python ocr_example.py scanned.pdf \\
        --det models/en_PP-OCRv5_det_infer.onnx \\
        --rec models/en_PP-OCRv5_rec_infer.onnx \\
        --dict models/en_dict.txt
"""

import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from scanned PDFs using OCR"
    )
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--det", required=True, help="Path to detection model (ONNX)")
    parser.add_argument("--rec", required=True, help="Path to recognition model (ONNX)")
    parser.add_argument("--dict", required=True, help="Path to character dictionary")
    parser.add_argument("--dpi", type=float, default=300.0, help="DPI for rendering (default: 300)")
    parser.add_argument("--page", type=int, help="Process only this page (0-indexed)")
    args = parser.parse_args()

    # Import pdf_oxide
    try:
        from pdf_oxide import PdfDocument, has_ocr
    except ImportError as e:
        print(f"Error: Failed to import pdf_oxide: {e}")
        print("Make sure to build with: maturin develop --features python,ocr")
        sys.exit(1)

    # Check if OCR feature is available
    if not has_ocr():
        print("Error: pdf_oxide was not built with OCR support")
        print("Rebuild with: maturin develop --features python,ocr")
        sys.exit(1)

    # Import OCR classes (only available when OCR feature is enabled)
    from pdf_oxide import OcrEngine, OcrConfig

    # Validate paths
    if not Path(args.pdf).exists():
        print(f"Error: PDF file not found: {args.pdf}")
        sys.exit(1)
    for path, name in [(args.det, "detection model"),
                       (args.rec, "recognition model"),
                       (args.dict, "dictionary")]:
        if not Path(path).exists():
            print(f"Error: {name} not found: {path}")
            sys.exit(1)

    print("=" * 70)
    print("PDF OCR Example")
    print("=" * 70)
    print()

    # Create OCR configuration
    print("Configuring OCR engine...")
    config = OcrConfig(
        det_threshold=0.3,
        box_threshold=0.5,
        rec_threshold=0.5,
        num_threads=4
    )
    print(f"Config: {config}")

    # Load OCR engine
    print("\nLoading OCR models...")
    try:
        engine = OcrEngine(
            det_model_path=args.det,
            rec_model_path=args.rec,
            dict_path=args.dict,
            config=config
        )
        print("OCR engine loaded successfully!")
    except Exception as e:
        print(f"Error loading OCR engine: {e}")
        sys.exit(1)

    # Open PDF
    print(f"\nOpening PDF: {args.pdf}")
    try:
        doc = PdfDocument(args.pdf)
        page_count = doc.page_count()
        print(f"PDF has {page_count} pages")
    except Exception as e:
        print(f"Error opening PDF: {e}")
        sys.exit(1)

    # Determine pages to process
    if args.page is not None:
        if args.page < 0 or args.page >= page_count:
            print(f"Error: Page {args.page} out of range (0-{page_count-1})")
            sys.exit(1)
        pages = [args.page]
    else:
        pages = range(page_count)

    # Process each page
    for page_idx in pages:
        print()
        print("-" * 70)
        print(f"Page {page_idx + 1} of {page_count}")
        print("-" * 70)

        # Check if page needs OCR
        try:
            needs_ocr = doc.needs_ocr(page_idx)
        except Exception as e:
            print(f"Warning: Could not check if OCR needed: {e}")
            needs_ocr = True  # Assume it needs OCR

        if needs_ocr:
            print("Page appears to be scanned, running OCR...")
            try:
                # Use OCR extraction
                text = doc.extract_text_with_ocr(
                    page=page_idx,
                    engine=engine,
                    dpi=args.dpi
                )
                if text.strip():
                    print("\nExtracted text:")
                    print(text)
                else:
                    print("(No text detected)")
            except Exception as e:
                print(f"OCR failed: {e}")
        else:
            print("Page has native text, using standard extraction...")
            try:
                text = doc.extract_text(page_idx)
                print("\nExtracted text:")
                print(text if text.strip() else "(No text found)")
            except Exception as e:
                print(f"Text extraction failed: {e}")

    print()
    print("=" * 70)
    print("Done!")
    print("=" * 70)


def demo_ocr_image(engine, image_path: str):
    """
    Demonstrate OCR on a standalone image file.

    Args:
        engine: OcrEngine instance
        image_path: Path to image file
    """
    print(f"\nRunning OCR on image: {image_path}")

    try:
        result = engine.ocr_image(image_path)

        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Detected {len(result['spans'])} text regions")
        print(f"\nFull text:\n{result['text']}")

        # Show individual spans
        if result['spans']:
            print("\nIndividual spans:")
            for i, span in enumerate(result['spans']):
                print(f"  {i+1}. [{span['confidence']:.2%}] {span['text']}")

    except Exception as e:
        print(f"OCR failed: {e}")


if __name__ == "__main__":
    main()

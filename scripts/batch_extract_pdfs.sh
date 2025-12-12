#!/bin/bash
#
# Batch Extract PDFs to Markdown
#
# This script extracts all PDFs from the test corpus to markdown format.
# It follows the correct approach: running export_to_markdown ONCE with
# directory parameters, not 356 times in a loop.
#
# Usage:
#   ./scripts/batch_extract_pdfs.sh
#
# Output:
#   - Timestamped directory in /tmp with all extracted .md files
#   - Log file at /tmp/extraction_output.log
#

set -e

echo "=== PDF Batch Extraction Script ==="
echo ""

# Create output directory with timestamp
OUTPUT_DIR="/tmp/pdf_extraction_$(date +%s)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Change to project directory
cd /home/yfedoseev/projects/pdf_oxide

# Run export_to_markdown ONCE with --input-dir and --output-dir
# This processes all PDFs in the directory in a single execution
echo "Starting batch extraction..."
echo "PDFs will be extracted from: /home/yfedoseev/projects/pdf_oxide_tests/pdfs/"
echo "Output format: Markdown (.md files)"
echo ""

cargo run --release --bin export_to_markdown -- \
    --input-dir /home/yfedoseev/projects/pdf_oxide_tests/pdfs \
    --output-dir "$OUTPUT_DIR" \
    --verbose 2>&1 | tee /tmp/extraction_output.log

echo ""
echo "=== EXTRACTION COMPLETE ==="
echo "Output dir: $OUTPUT_DIR"
echo "Total size: $(du -sh "$OUTPUT_DIR" | cut -f1)"
echo "File count: $(find "$OUTPUT_DIR" -type f | wc -l)"
echo "Log file: /tmp/extraction_output.log"
echo ""
echo "To analyze results:"
echo "  find \"$OUTPUT_DIR\" -type f -name \"*.md\" | head -5  # View first 5 files"
echo "  head -50 \"$OUTPUT_DIR\"/*.md                          # View first file content"

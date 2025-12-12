#!/bin/bash
#
# Compare PDF Extraction Methods
#
# This script compares extraction results between:
# 1. export_to_markdown binary (built-in converters)
# 2. extract_with_pipeline binary (TextPipeline API with features enabled)
#
# This helps validate that the TextPipeline produces equivalent or better results.
#
# Usage:
#   ./scripts/compare_extraction_methods.sh
#
# Output:
#   - Two timestamped directories in /tmp with extraction results
#   - Comparison report in /tmp/extraction_comparison_<timestamp>.txt
#

set -e

echo "=== PDF Extraction Comparison ==="
echo ""

# Create timestamped directories
TIMESTAMP=$(date +%s)
OUTPUT_BINARY="/tmp/pdf_extraction_binary_${TIMESTAMP}"
OUTPUT_PIPELINE="/tmp/pdf_extraction_pipeline_${TIMESTAMP}"
REPORT="/tmp/extraction_comparison_${TIMESTAMP}.txt"

mkdir -p "$OUTPUT_BINARY"
mkdir -p "$OUTPUT_PIPELINE"

echo "Comparison timestamp: $TIMESTAMP"
echo ""

# Change to project directory
cd /home/yfedoseev/projects/pdf_oxide

# ============================================================================
# STEP 1: Extract using export_to_markdown binary
# ============================================================================
echo "STEP 1: Extracting with export_to_markdown binary..."
echo "Output directory: $OUTPUT_BINARY"
echo ""

cargo run --release --bin export_to_markdown -- \
    --input-dir /home/yfedoseev/projects/pdf_oxide_tests/pdfs \
    --output-dir "$OUTPUT_BINARY" \
    --verbose 2>&1 | tee /tmp/extraction_binary_${TIMESTAMP}.log

echo ""
echo "✓ Binary extraction complete"
echo ""

# ============================================================================
# STEP 2: Extract using extract_with_pipeline binary
# ============================================================================
echo "STEP 2: Extracting with extract_with_pipeline (TextPipeline API)..."
echo "Output directory: $OUTPUT_PIPELINE"
echo ""

cargo run --release --bin extract_with_pipeline -- \
    --input-dir /home/yfedoseev/projects/pdf_oxide_tests/pdfs \
    --output-dir "$OUTPUT_PIPELINE" \
    --verbose 2>&1 | tee /tmp/extraction_pipeline_${TIMESTAMP}.log

echo ""
echo "✓ Pipeline extraction complete"
echo ""

# ============================================================================
# STEP 3: Generate comparison report
# ============================================================================
echo "STEP 3: Generating comparison report..."
echo ""

{
    echo "═════════════════════════════════════════════════════════════════════════"
    echo "PDF EXTRACTION COMPARISON REPORT"
    echo "═════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "Timestamp: $TIMESTAMP"
    echo "Date: $(date)"
    echo ""

    echo "─────────────────────────────────────────────────────────────────────────"
    echo "DIRECTORY STATISTICS"
    echo "─────────────────────────────────────────────────────────────────────────"
    echo ""

    BINARY_SIZE=$(du -sh "$OUTPUT_BINARY" | cut -f1)
    PIPELINE_SIZE=$(du -sh "$OUTPUT_PIPELINE" | cut -f1)
    BINARY_COUNT=$(find "$OUTPUT_BINARY" -type f | wc -l)
    PIPELINE_COUNT=$(find "$OUTPUT_PIPELINE" -type f | wc -l)

    echo "export_to_markdown (Binary):"
    echo "  Directory: $OUTPUT_BINARY"
    echo "  Total size: $BINARY_SIZE"
    echo "  File count: $BINARY_COUNT"
    echo ""

    echo "extract_with_pipeline (TextPipeline API):"
    echo "  Directory: $OUTPUT_PIPELINE"
    echo "  Total size: $PIPELINE_SIZE"
    echo "  File count: $PIPELINE_COUNT"
    echo ""

    echo "─────────────────────────────────────────────────────────────────────────"
    echo "FILE SIZE COMPARISON"
    echo "─────────────────────────────────────────────────────────────────────────"
    echo ""

    BINARY_TOTAL_BYTES=$(find "$OUTPUT_BINARY" -type f -exec du -b {} + | awk '{sum+=$1} END {print sum}')
    PIPELINE_TOTAL_BYTES=$(find "$OUTPUT_PIPELINE" -type f -exec du -b {} + | awk '{sum+=$1} END {print sum}')

    if [ -n "$BINARY_TOTAL_BYTES" ] && [ "$BINARY_TOTAL_BYTES" -gt 0 ]; then
        BINARY_AVG=$((BINARY_TOTAL_BYTES / BINARY_COUNT))
    else
        BINARY_AVG=0
    fi

    if [ -n "$PIPELINE_TOTAL_BYTES" ] && [ "$PIPELINE_TOTAL_BYTES" -gt 0 ]; then
        PIPELINE_AVG=$((PIPELINE_TOTAL_BYTES / PIPELINE_COUNT))
    else
        PIPELINE_AVG=0
    fi

    echo "Binary method:"
    echo "  Total bytes: $BINARY_TOTAL_BYTES"
    echo "  Average bytes/file: $BINARY_AVG"
    echo ""

    echo "Pipeline method:"
    echo "  Total bytes: $PIPELINE_TOTAL_BYTES"
    echo "  Average bytes/file: $PIPELINE_AVG"
    echo ""

    if [ "$PIPELINE_AVG" -gt "$BINARY_AVG" ]; then
        DIFF=$((PIPELINE_AVG - BINARY_AVG))
        PERCENT=$((DIFF * 100 / BINARY_AVG))
        echo "Pipeline produces ${DIFF} more bytes/file (+${PERCENT}%)"
    elif [ "$PIPELINE_AVG" -lt "$BINARY_AVG" ]; then
        DIFF=$((BINARY_AVG - PIPELINE_AVG))
        PERCENT=$((DIFF * 100 / BINARY_AVG))
        echo "Pipeline produces ${DIFF} fewer bytes/file (-${PERCENT}%)"
    else
        echo "Both methods produce identical average file sizes"
    fi
    echo ""

    echo "─────────────────────────────────────────────────────────────────────────"
    echo "SAMPLE FILE COMPARISON (First 3 files)"
    echo "─────────────────────────────────────────────────────────────────────────"
    echo ""

    # Get first 3 files from binary extraction
    SAMPLE_FILES=$(find "$OUTPUT_BINARY" -type f -name "*.md" | head -3)

    for BFILE in $SAMPLE_FILES; do
        # Get corresponding pipeline file
        RELATIVE=$(echo "$BFILE" | sed "s|$OUTPUT_BINARY||")
        PFILE="$OUTPUT_PIPELINE$RELATIVE"

        if [ -f "$PFILE" ]; then
            BSIZE=$(wc -c < "$BFILE")
            PSIZE=$(wc -c < "$PFILE")
            FILENAME=$(basename "$BFILE")

            echo "File: $FILENAME"
            echo "  Binary size: $BSIZE bytes"
            echo "  Pipeline size: $PSIZE bytes"

            if [ "$PSIZE" -gt "$BSIZE" ]; then
                DIFF=$((PSIZE - BSIZE))
                PERCENT=$((DIFF * 100 / BSIZE))
                echo "  Difference: +${DIFF} bytes (+${PERCENT}%)"
            elif [ "$PSIZE" -lt "$BSIZE" ]; then
                DIFF=$((BSIZE - PSIZE))
                PERCENT=$((DIFF * 100 / BSIZE))
                echo "  Difference: -${DIFF} bytes (-${PERCENT}%)"
            else
                echo "  Difference: None (identical)"
            fi
            echo ""
        fi
    done

    echo "─────────────────────────────────────────────────────────────────────────"
    echo "HOW TO COMPARE CONTENT"
    echo "─────────────────────────────────────────────────────────────────────────"
    echo ""
    echo "View binary extraction sample:"
    echo "  head -50 $OUTPUT_BINARY/*.md"
    echo ""
    echo "View pipeline extraction sample:"
    echo "  head -50 $OUTPUT_PIPELINE/*.md"
    echo ""
    echo "Find differences in a specific file:"
    echo "  diff <(head -100 $OUTPUT_BINARY/academic/arxiv_2510.21165v1.md) \\"
    echo "       <(head -100 $OUTPUT_PIPELINE/academic/arxiv_2510.21165v1.md)"
    echo ""
    echo "Count unique lines (shows content differences):"
    echo "  comm -3 <(sort $OUTPUT_BINARY/file.md | uniq) \\"
    echo "          <(sort $OUTPUT_PIPELINE/file.md | uniq) | wc -l"
    echo ""

    echo "═════════════════════════════════════════════════════════════════════════"
} | tee "$REPORT"

echo ""
echo "✓ Comparison complete!"
echo ""
echo "Report saved to: $REPORT"
echo ""
echo "Summary:"
echo "  Binary extraction:   $OUTPUT_BINARY (${BINARY_SIZE})"
echo "  Pipeline extraction: $OUTPUT_PIPELINE (${PIPELINE_SIZE})"
echo "  Comparison report:   $REPORT"

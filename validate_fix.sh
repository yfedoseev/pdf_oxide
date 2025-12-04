#!/bin/bash
set -e

echo "=== Adaptive Threshold Bug Fix Validation ==="
echo "Started: $(date)"
echo ""

# Step 1: Build
echo "Step 1: Building fixed binary..."
echo "----------------------------------------"
cargo build --release --bin batch_convert 2>&1 | grep -E "Finished|error" | head -5
cargo build --release --bin analyze_quality 2>&1 | grep -E "Finished|error" | head -5
echo ""

# Step 2: Clean previous output
echo "Step 2: Cleaning previous output..."
echo "----------------------------------------"
rm -rf batch_output quality_reports
echo "Cleaned: batch_output/ and quality_reports/"
echo ""

# Step 3: Re-run batch conversion
echo "Step 3: Running batch conversion with fixed adaptive threshold..."
echo "----------------------------------------"
echo "Input directories:"
echo "  - /home/yfedoseev/projects/pdf_oxide_new_docs"
echo "  - /home/yfedoseev/projects/pdf_oxide_tests/pdfs"
echo ""
time timeout 900 ./target/release/batch_convert \
  --input /home/yfedoseev/projects/pdf_oxide_new_docs \
  --input /home/yfedoseev/projects/pdf_oxide_tests/pdfs \
  --output ./batch_output \
  --save-metadata 2>&1 | tail -20
echo ""

# Step 4: Check output
echo "Step 4: Verifying batch conversion output..."
echo "----------------------------------------"
MARKDOWN_COUNT=$(find batch_output -name "*.md" -type f 2>/dev/null | wc -l)
METADATA_COUNT=$(find batch_output -name "*.json" -type f 2>/dev/null | wc -l)
TOTAL_SIZE=$(du -sh batch_output 2>/dev/null | cut -f1)
echo "Markdown files created: $MARKDOWN_COUNT"
echo "Metadata files created: $METADATA_COUNT"
echo "Total output size: $TOTAL_SIZE"
echo ""

# Step 5: Run quality analysis
echo "Step 5: Running quality analysis..."
echo "----------------------------------------"
time ./target/release/analyze_quality \
  --input ./batch_output \
  --output ./quality_reports \
  --formats html,json,csv 2>&1 | tail -10
echo ""

# Step 6: Extract key metrics
echo ""
echo "========================================"
echo "=== VALIDATION RESULTS ==="
echo "========================================"
echo "Completed: $(date)"
echo ""

if [ -f "quality_reports/quality_report.json" ]; then
    echo "✓ Quality Report Generated Successfully"
    echo ""
    echo "Key Metrics:"
    echo "----------------------------------------"

    # Extract metrics using grep and basic parsing
    AVG_SCORE=$(grep -o '"average_quality_score":[^,]*' quality_reports/quality_report.json | head -1 | cut -d':' -f2)
    PASS_RATE=$(grep -o '"pass_rate":[^,]*' quality_reports/quality_report.json | head -1 | cut -d':' -f2)
    TOTAL_DOCS=$(grep -o '"total_documents":[^,]*' quality_reports/quality_report.json | head -1 | cut -d':' -f2)

    echo "Average Quality Score: $AVG_SCORE"
    echo "Pass Rate: $PASS_RATE"
    echo "Total Documents: $TOTAL_DOCS"
    echo ""

    # Success criteria evaluation
    echo "Success Criteria Evaluation:"
    echo "----------------------------------------"

    # Convert to comparable format (remove decimals for comparison)
    SCORE_INT=$(echo "$AVG_SCORE" | awk '{print int($1 * 10)}')
    PASS_INT=$(echo "$PASS_RATE" | awk '{print int($1 * 100)}')

    if [ "$SCORE_INT" -ge 80 ] && [ "$PASS_INT" -ge 70 ]; then
        echo "✅ READY FOR RELEASE"
        echo "   - Quality score >= 8.0: YES ($AVG_SCORE)"
        echo "   - Pass rate >= 70%: YES ($PASS_RATE)"
    elif [ "$SCORE_INT" -ge 70 ] && [ "$PASS_INT" -ge 50 ]; then
        echo "⚠️  PARTIAL SUCCESS"
        echo "   - Quality score >= 7.0: $([ $SCORE_INT -ge 70 ] && echo YES || echo NO) ($AVG_SCORE)"
        echo "   - Pass rate >= 50%: $([ $PASS_INT -ge 50 ] && echo YES || echo NO) ($PASS_RATE)"
    else
        echo "❌ NEEDS IMPROVEMENT"
        echo "   - Quality score: $AVG_SCORE (target: 8.0+)"
        echo "   - Pass rate: $PASS_RATE (target: 70%+)"
    fi
    echo ""

    echo "Detailed reports available at:"
    echo "  - HTML: quality_reports/quality_report.html"
    echo "  - JSON: quality_reports/quality_report.json"
    echo "  - CSV: quality_reports/quality_report.csv"

else
    echo "❌ ERROR: Quality report not found at quality_reports/quality_report.json"
    echo ""
    echo "Output directory contents:"
    ls -la quality_reports/ 2>/dev/null || echo "Directory does not exist"
    exit 1
fi

echo ""
echo "========================================"
echo "Validation complete!"
echo "========================================"

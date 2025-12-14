#!/bin/bash
#
# Master script to collect 1000+ diverse PDFs for testing
#
# This script automates the collection of PDFs from various sources.
# Run with: bash scripts/collect_all.sh
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/test_datasets/pdfs_1000"

echo "=== PDF Collection Script ==="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p "$OUTPUT_DIR"/{academic,financial,government,business,technical,books,forms,news_media}/{arxiv,journals,10k,10q,policy,reports,press_releases,presentations,manuals,standards,newspapers,magazines}

echo "Done!"
echo ""

# 1. Academic Papers (200 target)
echo "=== Collecting Academic Papers ==="
echo "Target: 200 papers"
echo ""

# ArXiv papers - various categories
declare -a ARXIV_CATEGORIES=("cs.AI" "cs.LG" "cs.CV" "cs.CL" "math.CO" "stat.ML" "physics.data-an")

for category in "${ARXIV_CATEGORIES[@]}"; do
    echo "Downloading from $category..."
    python3 "$SCRIPT_DIR/download_arxiv.py" \
        --category "$category" \
        --max 15 \
        --output "$OUTPUT_DIR/academic/arxiv"

    sleep 5  # Be nice to servers
done

echo "ArXiv papers collected"
echo ""

# 2. Financial Documents (200 target)
echo "=== Collecting Financial Documents ==="
echo "Target: 200 documents"
echo ""

# 10-K filings
echo "Downloading 10-K filings..."
python3 "$SCRIPT_DIR/download_sec_filings.py" \
    --type "10-K" \
    --max 50 \
    --output "$OUTPUT_DIR/financial/10k"

sleep 10

# 10-Q filings
echo "Downloading 10-Q filings..."
python3 "$SCRIPT_DIR/download_sec_filings.py" \
    --type "10-Q" \
    --max 50 \
    --output "$OUTPUT_DIR/financial/10q"

echo "SEC filings collected"
echo ""

# 3. Government Documents
echo "=== Collecting Government Documents ==="
echo "Note: These often require manual collection"
echo "Suggested sources:"
echo "  - GAO reports: https://www.gao.gov/reports-testimonies"
echo "  - Congressional reports: https://www.congress.gov/reports"
echo "  - Federal Register: https://www.federalregister.gov/"
echo ""

# 4. Count collected PDFs
echo "=== Collection Summary ==="
echo ""

total_pdfs=$(find "$OUTPUT_DIR" -name "*.pdf" -type f | wc -l)
academic=$(find "$OUTPUT_DIR/academic" -name "*.pdf" -type f 2>/dev/null | wc -l)
financial=$(find "$OUTPUT_DIR/financial" -name "*.pdf" -type f 2>/dev/null | wc -l)

echo "Total PDFs collected: $total_pdfs"
echo "  Academic: $academic"
echo "  Financial: $financial"
echo ""

if [ $total_pdfs -lt 100 ]; then
    echo "⚠️  Less than 100 PDFs collected."
    echo "Continue collecting manually or run additional scripts."
elif [ $total_pdfs -lt 500 ]; then
    echo "✓ Milestone 1: 100+ PDFs collected"
    echo "Continue collecting 500+ PDFs"
elif [ $total_pdfs -lt 1000 ]; then
    echo "✓ Milestone 2: 500+ PDFs collected"
    echo "Continue collecting 1000+ PDFs"
else
    echo "✅ Target reached! 1000+ PDFs collected"
    echo "Ready for comprehensive testing"
fi

echo ""
echo "Next steps:"
echo "  1. Review collected PDFs: ls -R $OUTPUT_DIR"
echo "  2. Run comprehensive test: cargo run --release --bin comprehensive_test $OUTPUT_DIR"
echo "  3. Analyze results: cat $OUTPUT_DIR/test_results.json"
echo ""

echo "=== Collection Complete ==="

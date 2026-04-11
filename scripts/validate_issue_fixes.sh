#!/bin/bash
# Validate that the fixes for #313-316 hold on real PDFs from the
# pdf_oxide_tests collection and /tmp/repro_pdfs downloads.
#
# Usage: scripts/validate_issue_fixes.sh
# Requires ./target/release/examples/debug_issues to be built.

set -u

HARNESS="./target/release/examples/debug_issues"
TESTS="$HOME/projects/pdf_oxide_tests"
REPRO="/tmp/repro_pdfs"

fail=0
pass=0

run() {
    local name="$1"; shift
    local expect_pattern="$1"; shift
    local out
    out=$("$HARNESS" "$@" 2>&1)
    if echo "$out" | /usr/bin/grep -qE "$expect_pattern"; then
        echo "  PASS  $name"
        pass=$((pass + 1))
    else
        echo "  FAIL  $name  (expected pattern: $expect_pattern)"
        echo "$out" | /usr/bin/tail -10 | /usr/bin/sed 's/^/    | /'
        fail=$((fail + 1))
    fi
}

echo "=== Issue #313: AES-256 copy-protected widget text ==="
if [ -f "$TESTS/pdfs_pdfjs/secHandler.pdf" ]; then
    run "secHandler.pdf contains 'Security Handler'" \
        "Security Handler" \
        313 "$TESTS/pdfs_pdfjs/secHandler.pdf" 0
fi

echo
echo "=== Issue #314: ColumnAware fragmentation on single-column pages ==="
if [ -f "$TESTS/pdfs/diverse/RFC_2616_HTTP_1_1.pdf" ]; then
    run "RFC 2616 p10 ColumnAware = 1 group" \
        "ColumnAware inferred groups    : 1" \
        314 "$TESTS/pdfs/diverse/RFC_2616_HTTP_1_1.pdf" 10
    run "RFC 2616 p10 TopToBottom ≡ ColumnAware" \
        "TopToBottom and ColumnAware text identical: true" \
        314 "$TESTS/pdfs/diverse/RFC_2616_HTTP_1_1.pdf" 10
fi
if [ -f "$TESTS/pdfs/theses/Berkeley_Thesis_Systems_1.pdf" ]; then
    run "Berkeley Systems thesis p20 ColumnAware = 1 group" \
        "ColumnAware inferred groups    : 1" \
        314 "$TESTS/pdfs/theses/Berkeley_Thesis_Systems_1.pdf" 20
    run "Berkeley Systems thesis p50 ColumnAware = 1 group" \
        "ColumnAware inferred groups    : 1" \
        314 "$TESTS/pdfs/theses/Berkeley_Thesis_Systems_1.pdf" 50
fi
if [ -f "$TESTS/pdfs/diverse/EU_GDPR_Regulation.pdf" ]; then
    run "EU GDPR p5 ColumnAware = 1 group" \
        "ColumnAware inferred groups    : 1" \
        314 "$TESTS/pdfs/diverse/EU_GDPR_Regulation.pdf" 5
fi

echo
echo "=== Issue #315: table-detection label/value drop ==="
if [ -f "$REPRO/orafol_5900.pdf" ]; then
    run "ORAFOL 5900 p3 preserves 'Resistance to cleaning agents'" \
        "Resistance to cleaning agents" \
        315 "$REPRO/orafol_5900.pdf" 3
    run "ORAFOL 5900 p3 preserves 'Service life by specialist'" \
        "Service life by specialist" \
        315 "$REPRO/orafol_5900.pdf" 3
else
    echo "  SKIP  ORAFOL 5900 (download to /tmp/repro_pdfs/orafol_5900.pdf)"
fi

echo
echo "=== Issue #316: CJK tabular reorder ==="
if [ -f "$REPRO/cn_nhc_ws779.pdf" ]; then
    run "WS/T 779 p4 extract succeeds (smoke)" \
        "extract_text:" \
        316 "$REPRO/cn_nhc_ws779.pdf" 4
else
    echo "  SKIP  WS/T 779 (download to /tmp/repro_pdfs/cn_nhc_ws779.pdf)"
fi

echo
echo "=== Multi-column academic paper (regression guard) ==="
# arxiv papers in text_heavy / academic / pdfs_slow collections typically
# use a real 2-column layout; make sure we still detect them as multi-col
# and don't collapse the columns into a single row.
for arxiv in "$TESTS/pdfs/technical/arxiv_2312.00001.pdf" \
             "$TESTS/pdfs/technical/arxiv_2401.00001.pdf"; do
    if [ -f "$arxiv" ]; then
        name=$(basename "$arxiv")
        run "$name p0 does not crash" \
            "extract_text .merged." \
            314 "$arxiv" 0
    fi
done

# Academic paper body page with real figure/chart content must still
# produce multiple spatial groups (not get collapsed to one).
if [ -f "$TESTS/pdfs/academic/arxiv_2510.22239v1.pdf" ]; then
    run "arxiv_2510.22239v1 p5 shows >1 group (figure page)" \
        "ColumnAware inferred groups    : ([2-9]|[0-9][0-9])" \
        314 "$TESTS/pdfs/academic/arxiv_2510.22239v1.pdf" 5
fi

echo
echo "=========================================="
echo "Results: $pass pass, $fail fail"
echo "=========================================="
[ $fail -eq 0 ]

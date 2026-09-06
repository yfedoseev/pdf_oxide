#!/usr/bin/env bash
# Every PDF committed under tests/ must be listed, with its provenance, in
# tests/fixtures/TRACKED_PDFS.txt. The policy (CONTRIBUTING.md "Test fixture
# policy", AGENTS.md rule 4) is that fixtures are minimal synthetic PDFs built
# in code; a real specimen is the exception and has to say where it came from
# and why it cannot be synthesised. A PDF that is not listed fails the job.
#
# Usage: ./tools/check_tracked_fixtures.sh      (from the repository root)
set -euo pipefail
LIST="tests/fixtures/TRACKED_PDFS.txt"
[ -f "$LIST" ] || { echo "missing $LIST" >&2; exit 1; }
status=0
while IFS= read -r pdf; do
    # A listed path is the first whitespace-separated field of a non-comment line.
    if ! grep -vE '^\s*(#|$)' "$LIST" | awk '{print $1}' | grep -qxF "$pdf"; then
        echo "UNLISTED: $pdf — add it to $LIST with its provenance, or build the fixture in code." >&2
        status=1
    fi
done < <(git ls-files -- 'tests/**/*.pdf' 'tests/*.pdf')
if [ "$status" -eq 0 ]; then
    echo "every tracked PDF under tests/ is listed with provenance in $LIST"
fi
exit $status

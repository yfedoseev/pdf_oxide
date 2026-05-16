#!/bin/bash
# Regression tests for extract-release-notes.sh (issue #506).
#
# Covers the bug classes that shipped three consecutive wrong release titles:
#   * missing version section            → must fail loudly (not silent)
#   * missing subtitle                   → must fail loudly (NOT scrape an
#                                          older version's blockquote)
#   * cross-version false-scrape         → target version w/o '>' but an
#                                          older version has one → must fail
#   * multi-line subtitle                → concatenated, not truncated at L1
#   * empty stub section (no '### ')     → must fail loudly
#   * valid single-line subtitle         → correct "vX | subtitle" title
#   * --check mode                       → same verdicts, writes nothing
#
# Self-contained: no network, no cargo. Exits 0 iff every case passes.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/extract-release-notes.sh"

PASS=0
FAIL=0

# run_case <name> <version> <expect: ok|fail> <changelog-content-file> [expected-title] [expected-body-substr]
run_case() {
  local name="$1" version="$2" expect="$3" changelog="$4" expected_title="${5:-}" expected_body="${6:-}"
  local workdir rc title

  workdir="$(mktemp -d)"
  cp "$changelog" "$workdir/CHANGELOG.md"

  ( cd "$workdir" && "$SCRIPT" "$version" ) >/dev/null 2>&1
  rc=$?

  if [ "$expect" = "ok" ]; then
    if [ "$rc" -ne 0 ]; then
      echo "FAIL [$name]: expected success, script exited $rc"
      FAIL=$((FAIL + 1)); rm -rf "$workdir"; return
    fi
    title="$(cat "$workdir/release-title.txt" 2>/dev/null || echo '<no title>')"
    if [ -n "$expected_title" ] && [ "$title" != "$expected_title" ]; then
      echo "FAIL [$name]: title mismatch"
      echo "  expected: $expected_title"
      echo "  actual:   $title"
      FAIL=$((FAIL + 1)); rm -rf "$workdir"; return
    fi
    if [ -n "$expected_body" ] && ! grep -qF -- "$expected_body" "$workdir/release-notes.md" 2>/dev/null; then
      echo "FAIL [$name]: release-notes.md missing expected body content"
      echo "  expected substring: $expected_body"
      FAIL=$((FAIL + 1)); rm -rf "$workdir"; return
    fi
    # --check must agree and write nothing.
    ( cd "$workdir" && rm -f release-title.txt release-notes.md && "$SCRIPT" --check "$version" ) >/dev/null 2>&1
    if [ $? -ne 0 ]; then
      echo "FAIL [$name]: --check disagreed (expected ok)"
      FAIL=$((FAIL + 1)); rm -rf "$workdir"; return
    fi
    if [ -f "$workdir/release-title.txt" ] || [ -f "$workdir/release-notes.md" ]; then
      echo "FAIL [$name]: --check wrote output (release-title.txt / release-notes.md) — should be no-op"
      FAIL=$((FAIL + 1)); rm -rf "$workdir"; return
    fi
    echo "PASS [$name]"
    PASS=$((PASS + 1))
  else
    if [ "$rc" -eq 0 ]; then
      echo "FAIL [$name]: expected failure, but script succeeded (title: $(cat "$workdir/release-title.txt" 2>/dev/null))"
      FAIL=$((FAIL + 1)); rm -rf "$workdir"; return
    fi
    ( cd "$workdir" && "$SCRIPT" --check "$version" ) >/dev/null 2>&1
    if [ $? -eq 0 ]; then
      echo "FAIL [$name]: --check should also fail but succeeded"
      FAIL=$((FAIL + 1)); rm -rf "$workdir"; return
    fi
    echo "PASS [$name]"
    PASS=$((PASS + 1))
  fi
  rm -rf "$workdir"
}

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# --- Fixture: a CHANGELOG with multiple versions; 0.3.49 has NO subtitle but
#     an older version (0.3.44) does. The buggy script scraped 0.3.44's line.
cat > "$TMP/no_subtitle.md" <<'EOF'
# Changelog

## [0.3.49] - 2026-05-15

### Fixed
- Something without a subtitle blockquote.

## [0.3.44] - 2026-05-05

> Pluggable cryptographic provider — FIPS 140-3 compliance for
> government / regulated deployments.

### Added
- Older release that DOES have a subtitle.
EOF

cat > "$TMP/missing_section.md" <<'EOF'
# Changelog

## [0.3.48] - 2026-05-14

> Some other version.

### Fixed
- Not the version we ask for.
EOF

cat > "$TMP/multiline.md" <<'EOF'
# Changelog

## [0.3.49] - 2026-05-15

> Text-extraction reading-order rewire — fixes [#211]
> and closes the [#457] refactor.

### Fixed
- Multi-line subtitle must be concatenated, not truncated.
EOF

cat > "$TMP/stub.md" <<'EOF'
# Changelog

## [0.3.49] - 2026-05-15

> A subtitle but no notes at all.
EOF

cat > "$TMP/valid.md" <<'EOF'
# Changelog

## [0.3.49] - 2026-05-15

> Off-byte-0 PDF header recovery and release-automation hardening.

### Fixed
- A real fix.

## [0.3.48] - 2026-05-14

> An older release subtitle that must never leak upward.

### Added
- Older content.
EOF

# A later, separate blockquote in the section (e.g. a "> Scope note.") must
# NOT be merged into the subtitle — only the first contiguous '>' block is —
# and it must be PRESERVED in the release-notes body (not stripped).
cat > "$TMP/second_blockquote.md" <<'EOF'
# Changelog

## [0.3.49] - 2026-05-15

> The real subtitle line.

### Added
- Thing.

> **Scope note.** This must not be appended to the subtitle.
EOF

# A section that forgot its top subtitle but has a body blockquote LATER must
# fail loudly — the body note must never be anchored as the release title.
cat > "$TMP/late_blockquote_no_subtitle.md" <<'EOF'
# Changelog

## [0.3.49] - 2026-05-15

### Fixed
- A real fix but no subtitle directly under the header.

> **Scope note.** A body blockquote that is NOT the subtitle.
EOF

run_case "valid-single-line"      "0.3.49" ok   "$TMP/valid.md" \
  "v0.3.49 | Off-byte-0 PDF header recovery and release-automation hardening."
run_case "multi-line-concat"      "0.3.49" ok   "$TMP/multiline.md" \
  "v0.3.49 | Text-extraction reading-order rewire — fixes [#211] and closes the [#457] refactor."
run_case "first-block-only"       "0.3.49" ok   "$TMP/second_blockquote.md" \
  "v0.3.49 | The real subtitle line." \
  "> **Scope note.** This must not be appended to the subtitle."
run_case "missing-subtitle-no-scrape" "0.3.49" fail "$TMP/no_subtitle.md"
run_case "missing-version-section"    "0.3.49" fail "$TMP/missing_section.md"
run_case "empty-stub-no-heading"      "0.3.49" fail "$TMP/stub.md"
run_case "late-blockquote-not-subtitle" "0.3.49" fail "$TMP/late_blockquote_no_subtitle.md"

echo
echo "extract-release-notes regression: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]

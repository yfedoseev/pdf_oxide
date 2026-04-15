#!/usr/bin/env bash
# Fetch an external fixture corpus for the benchmark harness.
#
# Kreuzberg's corpus is the reference we track (see PLAN.md §scoring),
# but individual PDFs inside it carry varied licenses, so we don't
# vendor them — the script clones the upstream and symlinks the
# markdown-ground-truth subset into ./fixtures/kreuzberg.
#
# Re-run any time; idempotent.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
DEST="${SCRIPT_DIR}/../fixtures/kreuzberg"
UPSTREAM_DIR="${SCRIPT_DIR}/../.fixture-src/kreuzberg"
UPSTREAM_URL="https://github.com/Goldziher/kreuzberg.git"
# Pin so scoring numbers don't drift with upstream fixture churn.
UPSTREAM_REF="${KREUZBERG_REF:-main}"

mkdir -p "$(dirname "${DEST}")" "$(dirname "${UPSTREAM_DIR}")"

if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
  echo "cloning ${UPSTREAM_URL} → ${UPSTREAM_DIR}"
  git clone --depth 1 --branch "${UPSTREAM_REF}" "${UPSTREAM_URL}" "${UPSTREAM_DIR}"
else
  echo "updating ${UPSTREAM_DIR} to ${UPSTREAM_REF}"
  git -C "${UPSTREAM_DIR}" fetch --depth 1 origin "${UPSTREAM_REF}"
  git -C "${UPSTREAM_DIR}" checkout "${UPSTREAM_REF}"
fi

# Kreuzberg keeps PDFs under test_documents/pdf and ground-truth
# markdown under test_documents/ground_truth/pdf. We flatten this into
# one directory of symlinks so the harness's stem-matching loader
# (foo.pdf ↔ foo.md) just works.
PDF_SRC="${UPSTREAM_DIR}/test_documents/pdf"
GT_SRC="${UPSTREAM_DIR}/test_documents/ground_truth/pdf"
if [[ ! -d "${PDF_SRC}" || ! -d "${GT_SRC}" ]]; then
  echo "error: expected ${PDF_SRC} and ${GT_SRC} — upstream layout changed?" >&2
  exit 1
fi

rm -rf "${DEST}"
mkdir -p "${DEST}/pdfs" "${DEST}/gt"

# Use absolute targets so the symlinks resolve regardless of cwd.
PDF_SRC_ABS=$(cd "${PDF_SRC}" && pwd)
GT_SRC_ABS=$(cd "${GT_SRC}" && pwd)

for f in "${PDF_SRC_ABS}"/*.pdf; do
  [[ -f "$f" ]] || continue
  ln -sf "$f" "${DEST}/pdfs/$(basename "$f")"
done
for f in "${GT_SRC_ABS}"/*.md; do
  [[ -f "$f" ]] || continue
  ln -sf "$f" "${DEST}/gt/$(basename "$f")"
done

printf 'pdfs: %d\n'  "$(find -L "${DEST}/pdfs" -type f -name '*.pdf' | wc -l)"
printf 'gt:   %d\n' "$(find -L "${DEST}/gt"   -type f -name '*.md'  | wc -l)"
printf 'corpus at: %s\n' "${DEST}/pdfs"
printf 'gt dir at: %s\n' "${DEST}/gt"

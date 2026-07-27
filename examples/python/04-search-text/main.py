# Search for a term across all pages of a PDF and print matches.
# Run: python main.py document.pdf "query"

import sys

from pdf_oxide import PdfDocument


def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <file.pdf> <query>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    query = sys.argv[2]
    doc = PdfDocument(path)

    pages = doc.page_count()
    print(f'Searching for "{query}" in {path} ({pages} pages)...\n')

    # Build the per-page search index for every page up front, instead of
    # paying for it lazily on the first search_page()/search() call. Worth
    # it here since we're about to search every page anyway.
    doc.prepare_search()

    total = 0
    pages_with_hits = 0

    for i in range(pages):
        results = doc.search_page(i, query)
        if not results:
            continue
        pages_with_hits += 1
        print(f"Page {i + 1}: {len(results)} match(es)")
        for r in results:
            print(f'  - "{r["text"]}" (x={r["x"]:.1f} y={r["y"]:.1f})')
            total += 1
        print()

    print(f"Found {total} total matches across {pages_with_hits} pages.")

    # Free the cached search index now that we're done searching — useful
    # before heavy extraction work on the same document object.
    doc.clear_search_index()


if __name__ == "__main__":
    main()

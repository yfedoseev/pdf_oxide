# Changelog

## 0.3.77

- `PdfOxide.prepare_search/1`/`clear_search_index/1` — build the per-page
  search-index cache up front (instead of paying for it lazily on the
  first `search/4`/`search_all/3` call) or free it before heavy
  extraction on the same document.

## 0.3.69

- Initial release of the Elixir bindings for pdf_oxide over the C ABI as a
  dirty-scheduler NIF (CPU-bound work never blocks the BEAM): PDF text, Markdown
  and HTML extraction, page rendering, element and table extraction, document
  building, and more. Errors surface as `{:error, code}` tuples.

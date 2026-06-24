# Changelog

## 0.3.69

- Initial release of the Elixir bindings for pdf_oxide over the C ABI as a
  dirty-scheduler NIF (CPU-bound work never blocks the BEAM): PDF text, Markdown
  and HTML extraction, page rendering, element and table extraction, document
  building, and more. Errors surface as `{:error, code}` tuples.

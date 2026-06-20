# pdf_oxide binding API contract (CORE)

The canonical CORE surface every language binding must expose, consistently.
Capability **stems** and **signatures** are uniform; only **casing** follows each
language's convention (camelCase, snake_case, kebab-case, PascalCase types).

## Conventions (uniform across all bindings)

- **Error payload**: every failure carries `{ code: int, op: string }` — the C-ABI
  error code and the operation name. (Exceptions, error unions, `{:error, code}`
  tuples, or `NSError` userInfo — but always both fields.)
- **Page index**: 0-based and **required** (no default-argument shortcuts).
- **Resource freeing**: an explicit, idempotent `close` is exposed in every
  binding (in addition to any RAII/finalizer/GC). Using a closed handle raises a
  uniform "handle is closed" error.
- **`version`** returns a named `{ major, minor }` value (struct/record/map/named
  list) — never a bare tuple or out-params.

## Capability stems

`Document` (opened PDF):

| stem | signature | notes |
|---|---|---|
| `open` | `(path) -> Document` | |
| `open_from_bytes` | `(bytes) -> Document` | NOT `open_bytes` / `openData` |
| `open_with_password` | `(path, password) -> Document` | dedicated entry point, NOT a password option on `open` |
| `page_count` | `() -> int` | |
| `version` | `() -> {major, minor}` | named value |
| `is_encrypted` | `() -> bool` | method form (`?`-suffix where that is the language's boolean idiom) |
| `has_structure_tree` | `() -> bool` | method form |
| `extract_text` | `(page) -> string` | page required |
| `to_plain_text` | `(page) -> string` | page required |
| `to_markdown` | `(page) -> string` | page required |
| `to_html` | `(page) -> string` | page required |
| `to_markdown_all` | `() -> string` | |
| `extract_structured_json` | `(page) -> string` | page required |
| `close` | `() -> void` | idempotent |

`Pdf` (builder):

| stem | signature |
|---|---|
| `from_markdown` | `(markdown) -> Pdf` |
| `from_html` | `(html) -> Pdf` |
| `from_text` | `(text) -> Pdf` |
| `save` | `(path) -> void` |
| `save_to_bytes` | `() -> bytes` |
| `close` | `() -> void` (idempotent) |

## Casing per language

| stem example | C++/R/Julia/Elixir | Kotlin/Swift/Dart/Scala | Clojure | Obj-C |
|---|---|---|---|---|
| `open_from_bytes` | `open_from_bytes` | `openFromBytes` | `open-from-bytes` | `openFromBytes:error:` |
| `is_encrypted` | `is_encrypted` / `encrypted?` (Elixir) | `isEncrypted` | `encrypted?` | `isEncrypted` |

Memory string returns free via `free_string`; byte buffers via `free_bytes`
(never `free_string` on a byte buffer).

## Extended surface (added v0.3.68, all bindings)

Document, additional canonical methods (wrap existing C ABI):

| stem | signature | C ABI |
|---|---|---|
| `to_html_all` | `() -> string` | `pdf_document_to_html_all` |
| `to_plain_text_all` | `() -> string` | `pdf_document_to_plain_text_all` |
| `authenticate` | `(password) -> bool` | `pdf_document_authenticate` (wrong password = `false`, not an error) |
| `page` | `(index) -> Page` | 0-based; Page keeps its Document alive |

`Page` (lightweight, holds its Document + index; flat per-page C calls):

| stem | signature |
|---|---|
| `text` | `() -> string` (extract_text) |
| `markdown` | `() -> string` (to_markdown) |
| `html` | `() -> string` (to_html) |
| `plain_text` | `() -> string` (to_plain_text) |

Cased per language (e.g. `toHtmlAll`/`to_html_all`/`to-html-all`/`pdf_to_html_all`;
`plainText`/`plain_text`/`plain-text`; Obj-C `toHtmlAllWithError:`, `authenticate:error:`,
`pageAtIndex:`).

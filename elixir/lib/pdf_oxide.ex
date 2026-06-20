defmodule PdfOxide do
  @moduledoc """
  Idiomatic Elixir bindings for pdf_oxide — fast PDF text, Markdown and HTML
  extraction, plus building PDFs from Markdown/HTML/text.

  Backed by a NIF over the pdf_oxide C ABI; CPU-bound extraction runs on dirty
  CPU schedulers so it never blocks the BEAM. Handles are NIF resources freed by
  the GC. Functions return `{:ok, value}` / `{:error, code}`; the `!` variants
  raise `PdfOxide.Error`. Page indices are 0-based.
  """

  alias PdfOxide.Native

  defmodule Document do
    @moduledoc "An opened PDF document handle (NIF resource)."
    defstruct [:ref]
  end

  defmodule Pdf do
    @moduledoc "A built PDF handle (NIF resource)."
    defstruct [:ref]
  end

  defmodule Error do
    defexception [:code, :op]
    @impl true
    def message(%{code: code, op: op}),
      do: "pdf_oxide: #{op} failed (error code #{code})"
  end

  # ── Pdf builder ────────────────────────────────────────────────────────────
  @doc "Build a PDF from Markdown."
  def from_markdown(md), do: wrap_pdf(Native.from_markdown(md))
  @doc "Build a PDF from HTML."
  def from_html(html), do: wrap_pdf(Native.from_html(html))
  @doc "Build a PDF from plain text."
  def from_text(text), do: wrap_pdf(Native.from_text(text))

  @doc "Write a built PDF to `path`."
  def save(%Pdf{ref: ref}, path), do: Native.pdf_save(ref, path)
  @doc "Serialize a built PDF to a binary."
  def save_to_bytes(%Pdf{ref: ref}), do: Native.pdf_save_to_bytes(ref)

  # ── Document ─────────────────────────────────────────────────────────────────
  @doc "Open a PDF from a path. Pass `password:` for encrypted files."
  def open(path, opts \\ []) do
    case Keyword.get(opts, :password) do
      nil -> wrap_doc(Native.doc_open(path))
      pw -> wrap_doc(Native.doc_open_pw(path, pw))
    end
  end

  @doc "Open a PDF from a binary."
  def open_bytes(bytes), do: wrap_doc(Native.doc_open_bytes(bytes))

  @doc "Number of pages."
  def page_count(%Document{ref: ref}), do: Native.doc_page_count(ref)
  @doc "PDF version as `{major, minor}`."
  def version(%Document{ref: ref}), do: Native.doc_version(ref)
  @doc "Whether the document is encrypted."
  def encrypted?(%Document{ref: ref}), do: Native.doc_is_encrypted(ref)
  @doc "Whether the document has a logical structure tree."
  def structure_tree?(%Document{ref: ref}), do: Native.doc_has_structure_tree(ref)

  @doc "Reading-order text for a (0-based) page."
  def extract_text(%Document{ref: ref}, page \\ 0), do: Native.doc_extract_text(ref, page)
  @doc "Plain text for a page."
  def to_plain_text(%Document{ref: ref}, page \\ 0), do: Native.doc_to_plain_text(ref, page)
  @doc "Markdown for a page."
  def to_markdown(%Document{ref: ref}, page \\ 0), do: Native.doc_to_markdown(ref, page)
  @doc "HTML for a page."
  def to_html(%Document{ref: ref}, page \\ 0), do: Native.doc_to_html(ref, page)
  @doc "Markdown for the whole document."
  def to_markdown_all(%Document{ref: ref}), do: Native.doc_to_markdown_all(ref)
  @doc "Structured content for a page as a JSON string."
  def extract_structured_json(%Document{ref: ref}, page \\ 0),
    do: Native.doc_extract_structured_json(ref, page)

  # ── helpers ──────────────────────────────────────────────────────────────────
  defp wrap_doc({:ok, ref}), do: {:ok, %Document{ref: ref}}
  defp wrap_doc(other), do: other
  defp wrap_pdf({:ok, ref}), do: {:ok, %Pdf{ref: ref}}
  defp wrap_pdf(other), do: other
end

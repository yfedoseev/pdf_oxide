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

  defmodule Page do
    @moduledoc """
    A lightweight view of a single (0-based) page. Holds its `Document` so the
    underlying native handle stays alive as long as the page is referenced.
    """
    defstruct [:doc, :index]
  end

  defmodule Error do
    defexception [:code, :op]
    @impl true
    def message(%{code: code, op: op}),
      do: "pdf_oxide: #{op} failed (error code #{code})"
  end

  defmodule Bbox do
    @moduledoc "An axis-aligned bounding box (PDF user-space units)."
    defstruct [:x, :y, :width, :height]
  end

  defmodule Char do
    @moduledoc "A single extracted character. `character` is a Unicode codepoint (integer)."
    defstruct [:character, :bbox, :font_name, :font_size]
  end

  defmodule Word do
    @moduledoc "An extracted word with its layout/style metadata."
    defstruct [:text, :bbox, :font_name, :font_size, :bold]
  end

  defmodule TextLine do
    @moduledoc "An extracted line of text."
    defstruct [:text, :bbox, :word_count]
  end

  defmodule Table do
    @moduledoc """
    An extracted table. Read a cell's text with `cell/3` (0-based `row`/`col`).
    `cells` holds the cell text as a row-major list of lists.
    """
    defstruct [:row_count, :col_count, :has_header, :cells]
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
  def to_bytes(%Pdf{ref: ref}), do: Native.pdf_save_to_bytes(ref)

  @doc "Free a document or built PDF's native handle now (idempotent)."
  def close(%Document{ref: ref}), do: Native.doc_close(ref)
  def close(%Pdf{ref: ref}), do: Native.pdf_close(ref)

  # ── Document ─────────────────────────────────────────────────────────────────
  @doc "Open a PDF from a path."
  def open(path), do: wrap_doc(Native.doc_open(path))

  @doc "Open a password-protected PDF."
  def open_with_password(path, password), do: wrap_doc(Native.doc_open_pw(path, password))

  @doc "Open a PDF from a binary."
  def open_from_bytes(bytes), do: wrap_doc(Native.doc_open_bytes(bytes))

  @doc "Number of pages."
  def page_count(%Document{ref: ref}), do: Native.doc_page_count(ref)
  @doc "PDF version as `%{major: _, minor: _}`."
  def version(%Document{ref: ref}) do
    {major, minor} = Native.doc_version(ref)
    %{major: major, minor: minor}
  end

  @doc "Whether the document is encrypted."
  def encrypted?(%Document{ref: ref}), do: Native.doc_is_encrypted(ref)
  @doc "Whether the document has a logical structure tree."
  def structure_tree?(%Document{ref: ref}), do: Native.doc_has_structure_tree(ref)

  @doc "Reading-order text for a (0-based) page."
  def extract_text(%Document{ref: ref}, page), do: Native.doc_extract_text(ref, page)
  @doc "Plain text for a page."
  def to_plain_text(%Document{ref: ref}, page), do: Native.doc_to_plain_text(ref, page)
  @doc "Markdown for a page."
  def to_markdown(%Document{ref: ref}, page), do: Native.doc_to_markdown(ref, page)
  @doc "HTML for a page."
  def to_html(%Document{ref: ref}, page), do: Native.doc_to_html(ref, page)
  @doc "Markdown for the whole document."
  def to_markdown_all(%Document{ref: ref}), do: Native.doc_to_markdown_all(ref)
  @doc "HTML for the whole document."
  def to_html_all(%Document{ref: ref}), do: Native.doc_to_html_all(ref)
  @doc "Plain text for the whole document."
  def to_plain_text_all(%Document{ref: ref}), do: Native.doc_to_plain_text_all(ref)

  @doc """
  Authenticate an encrypted document with `password`. Returns `{:ok, true}` on
  success and `{:ok, false}` for a wrong password (not an error).
  """
  def authenticate(%Document{ref: ref}, password), do: Native.doc_authenticate(ref, password)

  @doc "Structured content for a page as a JSON string."
  def extract_structured_json(%Document{ref: ref}, page),
    do: Native.doc_extract_structured_json(ref, page)

  @doc """
  Extract the individual characters of a (0-based) page as a list of `Char`.
  """
  def extract_chars(%Document{ref: ref}, page) do
    with {:ok, list} <- Native.doc_extract_chars(ref, page) do
      {:ok,
       Enum.map(list, fn {cp, x, y, w, h, font, size} ->
         %Char{
           character: cp,
           bbox: %Bbox{x: x, y: y, width: w, height: h},
           font_name: font,
           font_size: size
         }
       end)}
    end
  end

  @doc """
  Extract the words of a (0-based) page as a list of `Word`.
  """
  def extract_words(%Document{ref: ref}, page) do
    with {:ok, list} <- Native.doc_extract_words(ref, page) do
      {:ok,
       Enum.map(list, fn {text, x, y, w, h, font, size, bold} ->
         %Word{
           text: text,
           bbox: %Bbox{x: x, y: y, width: w, height: h},
           font_name: font,
           font_size: size,
           bold: bold
         }
       end)}
    end
  end

  @doc """
  Extract the text lines of a (0-based) page as a list of `TextLine`.
  """
  def extract_text_lines(%Document{ref: ref}, page) do
    with {:ok, list} <- Native.doc_extract_text_lines(ref, page) do
      {:ok,
       Enum.map(list, fn {text, x, y, w, h, word_count} ->
         %TextLine{
           text: text,
           bbox: %Bbox{x: x, y: y, width: w, height: h},
           word_count: word_count
         }
       end)}
    end
  end

  @doc """
  Extract the tables of a (0-based) page as a list of `Table`. Use `cell/3` to
  read a table's (0-based) cell text.
  """
  def extract_tables(%Document{ref: ref}, page) do
    with {:ok, list} <- Native.doc_extract_tables(ref, page) do
      {:ok,
       Enum.map(list, fn {row_count, col_count, has_header, cells} ->
         %Table{
           row_count: row_count,
           col_count: col_count,
           has_header: has_header,
           cells: cells
         }
       end)}
    end
  end

  @doc "Text of a table's (0-based) `row`/`col` cell."
  def cell(%Table{cells: cells}, row, col),
    do: cells |> Enum.at(row, []) |> Enum.at(col)

  # ── Page ─────────────────────────────────────────────────────────────────────
  @doc """
  A `Page` view for the (0-based) `index`. The page keeps its document alive, so
  it must not outlive a `close/1` on the document.
  """
  def page(%Document{} = doc, index) when is_integer(index),
    do: %Page{doc: doc, index: index}

  @doc "Reading-order text for the page."
  def text(%Page{doc: doc, index: index}), do: extract_text(doc, index)
  @doc "Markdown for the page."
  def markdown(%Page{doc: doc, index: index}), do: to_markdown(doc, index)
  @doc "HTML for the page."
  def html(%Page{doc: doc, index: index}), do: to_html(doc, index)
  @doc "Plain text for the page."
  def plain_text(%Page{doc: doc, index: index}), do: to_plain_text(doc, index)

  # ── helpers ──────────────────────────────────────────────────────────────────
  defp wrap_doc({:ok, ref}), do: {:ok, %Document{ref: ref}}
  defp wrap_doc(other), do: other
  defp wrap_pdf({:ok, ref}), do: {:ok, %Pdf{ref: ref}}
  defp wrap_pdf(other), do: other
end

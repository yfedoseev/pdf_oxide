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

  defmodule DocumentEditor do
    @moduledoc """
    A mutable PDF editing handle (NIF resource). Open one with
    `PdfOxide.open_editor/1` or `PdfOxide.open_editor_from_bytes/1`, mutate it
    in place (rotate/crop/redact/flatten/merge/…) and serialise with
    `PdfOxide.editor_save/2` or `PdfOxide.editor_save_to_bytes/1`. The native
    handle is freed by the GC or eagerly via `PdfOxide.editor_close/1`. Page
    indices are 0-based.
    """
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

  defmodule Font do
    @moduledoc "An embedded/referenced font on a page."
    defstruct [:name, :type, :encoding, :embedded, :subset]
  end

  defmodule Image do
    @moduledoc "An embedded image. `data` holds its raw bytes."
    defstruct [:width, :height, :bits_per_component, :format, :colorspace, :data]
  end

  defmodule Annotation do
    @moduledoc "A page annotation with its placement and style metadata."
    defstruct [:type, :subtype, :content, :author, :rect, :border_width]
  end

  defmodule Path do
    @moduledoc "An extracted vector path (its bbox and stroke/fill style)."
    defstruct [:bbox, :stroke_width, :has_stroke, :has_fill, :operation_count]
  end

  defmodule SearchResult do
    @moduledoc "A single search hit: its `text`, 0-based `page` and `bbox`."
    defstruct [:text, :page, :bbox]
  end

  defmodule RenderedImage do
    @moduledoc """
    A rendered page raster. `width`/`height` are in pixels and `data` holds the
    encoded image bytes (PNG by default). `ref` is the live native handle kept so
    `PdfOxide.save/2` can write the image with the renderer's own encoder; it is
    freed by the GC.
    """
    defstruct [:ref, :width, :height, :data]
  end

  # ── Pdf builder ────────────────────────────────────────────────────────────
  @doc "Build a PDF from Markdown."
  def from_markdown(md), do: wrap_pdf(Native.from_markdown(md))
  @doc "Build a PDF from HTML."
  def from_html(html), do: wrap_pdf(Native.from_html(html))
  @doc "Build a PDF from plain text."
  def from_text(text), do: wrap_pdf(Native.from_text(text))

  @doc """
  Write to `path` — a built `Pdf`, or a `RenderedImage` page raster.
  """
  def save(%Pdf{ref: ref}, path), do: Native.pdf_save(ref, path)
  def save(%RenderedImage{ref: ref}, path), do: Native.img_save(ref, path)
  @doc "Serialize a built PDF to a binary."
  def to_bytes(%Pdf{ref: ref}), do: Native.pdf_save_to_bytes(ref)

  @doc "Free a document, built PDF or editor's native handle now (idempotent)."
  def close(%Document{ref: ref}), do: Native.doc_close(ref)
  def close(%Pdf{ref: ref}), do: Native.pdf_close(ref)
  def close(%DocumentEditor{ref: ref}), do: Native.editor_close(ref)

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

  @doc """
  Extract the embedded/referenced fonts of a (0-based) page as a list of `Font`.
  """
  def embedded_fonts(%Document{ref: ref}, page) do
    with {:ok, list} <- Native.doc_embedded_fonts(ref, page) do
      {:ok,
       Enum.map(list, fn {name, type, encoding, embedded, subset} ->
         %Font{
           name: name,
           type: type,
           encoding: encoding,
           embedded: embedded,
           subset: subset
         }
       end)}
    end
  end

  @doc """
  Extract the embedded images of a (0-based) page as a list of `Image`.
  """
  def embedded_images(%Document{ref: ref}, page) do
    with {:ok, list} <- Native.doc_embedded_images(ref, page) do
      {:ok,
       Enum.map(list, fn {width, height, bpc, format, colorspace, data} ->
         %Image{
           width: width,
           height: height,
           bits_per_component: bpc,
           format: format,
           colorspace: colorspace,
           data: data
         }
       end)}
    end
  end

  @doc """
  Extract the annotations of a (0-based) page as a list of `Annotation`.
  """
  def page_annotations(%Document{ref: ref}, page) do
    with {:ok, list} <- Native.doc_page_annotations(ref, page) do
      {:ok,
       Enum.map(list, fn {type, subtype, content, author, x, y, w, h, border_width} ->
         %Annotation{
           type: type,
           subtype: subtype,
           content: content,
           author: author,
           rect: %Bbox{x: x, y: y, width: w, height: h},
           border_width: border_width
         }
       end)}
    end
  end

  @doc """
  Extract the vector paths of a (0-based) page as a list of `Path`.
  """
  def extract_paths(%Document{ref: ref}, page) do
    with {:ok, list} <- Native.doc_extract_paths(ref, page) do
      {:ok,
       Enum.map(list, fn {x, y, w, h, stroke_width, has_stroke, has_fill, operation_count} ->
         %Path{
           bbox: %Bbox{x: x, y: y, width: w, height: h},
           stroke_width: stroke_width,
           has_stroke: has_stroke,
           has_fill: has_fill,
           operation_count: operation_count
         }
       end)}
    end
  end

  @doc """
  Search a (0-based) page for `term`, returning a list of `SearchResult`.
  """
  def search(%Document{ref: ref}, page, term, case_sensitive) do
    with {:ok, list} <- Native.doc_search_page(ref, page, term, case_sensitive) do
      {:ok, Enum.map(list, &to_search_result/1)}
    end
  end

  @doc """
  Search the whole document for `term`, returning a list of `SearchResult`.
  """
  def search_all(%Document{ref: ref}, term, case_sensitive) do
    with {:ok, list} <- Native.doc_search_all(ref, term, case_sensitive) do
      {:ok, Enum.map(list, &to_search_result/1)}
    end
  end

  defp to_search_result({text, page, x, y, w, h}),
    do: %SearchResult{text: text, page: page, bbox: %Bbox{x: x, y: y, width: w, height: h}}

  # ── page rendering (phase 3) ─────────────────────────────────────────────────
  @doc """
  Render a (0-based) `page_index` to a `RenderedImage`. `format` is an image
  format code (0 = PNG, the default).
  """
  def render_page(%Document{ref: ref}, page_index, format \\ 0),
    do: wrap_image(Native.doc_render_page(ref, page_index, format))

  @doc """
  Render a (0-based) `page_index` at `zoom` (1.0 = 100%) to a `RenderedImage`.
  `format` is an image format code (0 = PNG, the default).
  """
  def render_page_zoom(%Document{ref: ref}, page_index, zoom, format \\ 0),
    do: wrap_image(Native.doc_render_page_zoom(ref, page_index, zoom * 1.0, format))

  @doc """
  Render a (0-based) `page_index` as a thumbnail fitting `size` pixels on the
  longest side, to a `RenderedImage`. `format` is an image format code
  (0 = PNG, the default).
  """
  def render_page_thumbnail(%Document{ref: ref}, page_index, size, format \\ 0),
    do: wrap_image(Native.doc_render_page_thumbnail(ref, page_index, size, format))

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

  # ── DocumentEditor ───────────────────────────────────────────────────────────
  @doc "Open a PDF for editing from a path."
  def open_editor(path), do: wrap_editor(Native.editor_open(path))
  @doc "Open a PDF for editing from a binary."
  def open_editor_from_bytes(bytes), do: wrap_editor(Native.editor_open_bytes(bytes))

  @doc "Number of pages in the editor."
  def editor_page_count(%DocumentEditor{ref: ref}), do: Native.editor_page_count(ref)
  @doc "PDF version as `%{major: _, minor: _}`."
  def editor_version(%DocumentEditor{ref: ref}) do
    {major, minor} = Native.editor_version(ref)
    %{major: major, minor: minor}
  end

  @doc "Whether the editor has unsaved modifications."
  def editor_modified?(%DocumentEditor{ref: ref}), do: Native.editor_is_modified(ref)
  @doc "The editor's source path (empty for a bytes-opened editor)."
  def editor_source_path(%DocumentEditor{ref: ref}), do: Native.editor_source_path(ref)

  @doc "Read `/Info.Producer`."
  def get_producer(%DocumentEditor{ref: ref}), do: Native.editor_get_producer(ref)
  @doc "Set `/Info.Producer`."
  def set_producer(%DocumentEditor{ref: ref}, value), do: Native.editor_set_producer(ref, value)
  @doc "Read `/Info.CreationDate` as a raw PDF date string."
  def get_creation_date(%DocumentEditor{ref: ref}), do: Native.editor_get_creation_date(ref)
  @doc "Set `/Info.CreationDate` (raw PDF date string)."
  def set_creation_date(%DocumentEditor{ref: ref}, date),
    do: Native.editor_set_creation_date(ref, date)

  @doc "Delete a (0-based) page."
  def delete_page(%DocumentEditor{ref: ref}, page_index),
    do: Native.editor_delete_page(ref, page_index)

  @doc "Move a (0-based) page `from` → `to`."
  def move_page(%DocumentEditor{ref: ref}, from, to), do: Native.editor_move_page(ref, from, to)

  @doc "Rotate a single (0-based) page by `degrees` (additive)."
  def rotate_page_by(%DocumentEditor{ref: ref}, page, degrees),
    do: Native.editor_rotate_page_by(ref, page, degrees)

  @doc "Rotate all pages by `degrees` (relative)."
  def rotate_all_pages(%DocumentEditor{ref: ref}, degrees),
    do: Native.editor_rotate_all_pages(ref, degrees)

  @doc "Set the absolute rotation of a (0-based) page."
  def set_page_rotation(%DocumentEditor{ref: ref}, page, degrees),
    do: Native.editor_set_page_rotation(ref, page, degrees)

  @doc "Rotation (degrees) of a (0-based) page."
  def get_page_rotation(%DocumentEditor{ref: ref}, page),
    do: Native.editor_get_page_rotation(ref, page)

  @doc "Crop `left`/`right`/`top`/`bottom` margins off every page."
  def crop_margins(%DocumentEditor{ref: ref}, left, right, top, bottom),
    do: Native.editor_crop_margins(ref, left * 1.0, right * 1.0, top * 1.0, bottom * 1.0)

  @doc "CropBox of a (0-based) page as a `Bbox`."
  def get_page_crop_box(%DocumentEditor{ref: ref}, page),
    do: wrap_box(Native.editor_get_crop_box(ref, page))

  @doc "Set the CropBox of a (0-based) page."
  def set_page_crop_box(%DocumentEditor{ref: ref}, page, x, y, w, h),
    do: Native.editor_set_crop_box(ref, page, x * 1.0, y * 1.0, w * 1.0, h * 1.0)

  @doc "MediaBox of a (0-based) page as a `Bbox`."
  def get_page_media_box(%DocumentEditor{ref: ref}, page),
    do: wrap_box(Native.editor_get_media_box(ref, page))

  @doc "Set the MediaBox of a (0-based) page."
  def set_page_media_box(%DocumentEditor{ref: ref}, page, x, y, w, h),
    do: Native.editor_set_media_box(ref, page, x * 1.0, y * 1.0, w * 1.0, h * 1.0)

  @doc "Apply (burn in) redactions on a single (0-based) page."
  def apply_page_redactions(%DocumentEditor{ref: ref}, page),
    do: Native.editor_apply_page_redactions(ref, page)

  @doc "Apply all pending redactions across the document."
  def apply_all_redactions(%DocumentEditor{ref: ref}), do: Native.editor_apply_all_redactions(ref)

  @doc "Whether a (0-based) page is marked for redaction."
  def page_marked_for_redaction?(%DocumentEditor{ref: ref}, page),
    do: Native.editor_is_marked_for_redaction(ref, page)

  @doc "Remove the redaction mark from a (0-based) page."
  def unmark_page_for_redaction(%DocumentEditor{ref: ref}, page),
    do: Native.editor_unmark_for_redaction(ref, page)

  @doc "Erase a single rectangular region on a (0-based) page."
  def erase_region(%DocumentEditor{ref: ref}, page, x, y, w, h),
    do: Native.editor_erase_region(ref, page, x * 1.0, y * 1.0, w * 1.0, h * 1.0)

  @doc """
  Erase multiple rectangular regions on a (0-based) page. `rects` is a list of
  `{x, y, w, h}` tuples.
  """
  def erase_regions(%DocumentEditor{ref: ref}, page, rects) when is_list(rects) do
    quads = Enum.map(rects, fn {x, y, w, h} -> {x * 1.0, y * 1.0, w * 1.0, h * 1.0} end)
    Native.editor_erase_regions(ref, page, quads)
  end

  @doc "Clear all pending erase-region entries for a (0-based) page."
  def clear_erase_regions(%DocumentEditor{ref: ref}, page),
    do: Native.editor_clear_erase_regions(ref, page)

  @doc "Flatten annotations on a (0-based) page."
  def flatten_annotations(%DocumentEditor{ref: ref}, page),
    do: Native.editor_flatten_annotations(ref, page)

  @doc "Flatten annotations across the whole document."
  def flatten_all_annotations(%DocumentEditor{ref: ref}),
    do: Native.editor_flatten_all_annotations(ref)

  @doc "Whether a (0-based) page is marked for annotation-flatten."
  def page_marked_for_flatten?(%DocumentEditor{ref: ref}, page),
    do: Native.editor_is_marked_for_flatten(ref, page)

  @doc "Remove the flatten mark from a (0-based) page."
  def unmark_page_for_flatten(%DocumentEditor{ref: ref}, page),
    do: Native.editor_unmark_for_flatten(ref, page)

  @doc "Set a form field value (UTF-8)."
  def set_form_field_value(%DocumentEditor{ref: ref}, name, value),
    do: Native.editor_set_form_field_value(ref, name, value)

  @doc "Flatten all forms (bake field values into page content)."
  def flatten_forms(%DocumentEditor{ref: ref}), do: Native.editor_flatten_forms(ref)

  @doc "Flatten forms on a specific (0-based) page."
  def flatten_forms_on_page(%DocumentEditor{ref: ref}, page_index),
    do: Native.editor_flatten_forms_on_page(ref, page_index)

  @doc "Number of warnings from the last form-flatten."
  def flatten_warnings_count(%DocumentEditor{ref: ref}),
    do: Native.editor_flatten_warnings_count(ref)

  @doc "The `index`-th flatten warning string."
  def flatten_warning(%DocumentEditor{ref: ref}, index),
    do: Native.editor_flatten_warning(ref, index)

  @doc "Merge pages from a source PDF on disk into this document."
  def merge_from(%DocumentEditor{ref: ref}, source_path),
    do: Native.editor_merge_from(ref, source_path)

  @doc "Merge pages from an in-memory PDF binary into this document."
  def merge_from_bytes(%DocumentEditor{ref: ref}, bytes),
    do: Native.editor_merge_from_bytes(ref, bytes)

  @doc "Convert the document to PDF/A in place (`level` 0..7)."
  def convert_to_pdf_a(%DocumentEditor{ref: ref}, level),
    do: Native.editor_convert_to_pdf_a(ref, level)

  @doc "Embed a file attachment `name` with `bytes` into the document."
  def embed_file(%DocumentEditor{ref: ref}, name, bytes),
    do: Native.editor_embed_file(ref, name, bytes)

  @doc "Extract a subset of (0-based) `pages` to a new in-memory PDF binary."
  def extract_pages_to_bytes(%DocumentEditor{ref: ref}, pages) when is_list(pages),
    do: Native.editor_extract_pages_to_bytes(ref, pages)

  @doc "Save the edited document to `path`."
  def editor_save(%DocumentEditor{ref: ref}, path), do: Native.editor_save(ref, path)
  @doc "Serialize the edited document to a binary."
  def editor_save_to_bytes(%DocumentEditor{ref: ref}), do: Native.editor_save_to_bytes(ref)

  @doc "Serialize the edited document to bytes with compress/GC/linearize options."
  def editor_save_to_bytes_with_options(
        %DocumentEditor{ref: ref},
        compress,
        garbage_collect,
        linearize
      ),
      do: Native.editor_save_to_bytes_with_options(ref, compress, garbage_collect, linearize)

  @doc "Save the edited document AES-256 encrypted to `path`."
  def editor_save_encrypted(%DocumentEditor{ref: ref}, path, user_password, owner_password),
    do: Native.editor_save_encrypted(ref, path, user_password, owner_password)

  @doc "Serialize the edited document AES-256 encrypted to a binary."
  def editor_save_encrypted_to_bytes(%DocumentEditor{ref: ref}, user_password, owner_password),
    do: Native.editor_save_encrypted_to_bytes(ref, user_password, owner_password)

  @doc "Free the editor's native handle now (idempotent)."
  def editor_close(%DocumentEditor{ref: ref}), do: Native.editor_close(ref)

  # ── helpers ──────────────────────────────────────────────────────────────────
  defp wrap_doc({:ok, ref}), do: {:ok, %Document{ref: ref}}
  defp wrap_doc(other), do: other
  defp wrap_pdf({:ok, ref}), do: {:ok, %Pdf{ref: ref}}
  defp wrap_pdf(other), do: other
  defp wrap_editor({:ok, ref}), do: {:ok, %DocumentEditor{ref: ref}}
  defp wrap_editor(other), do: other

  defp wrap_box({:ok, {x, y, w, h}}), do: {:ok, %Bbox{x: x, y: y, width: w, height: h}}
  defp wrap_box(other), do: other

  defp wrap_image({:ok, {ref, width, height, data}}),
    do: {:ok, %RenderedImage{ref: ref, width: width, height: height, data: data}}

  defp wrap_image(other), do: other
end

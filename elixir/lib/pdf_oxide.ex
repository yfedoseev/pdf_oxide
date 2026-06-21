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

  defmodule DocumentBuilder do
    @moduledoc """
    A PDF *creation* builder handle (NIF resource). Create one with
    `PdfOxide.builder/0`, set metadata, start pages with `builder_page/3`,
    `builder_letter_page/1` or `builder_a4_page/1`, then `builder_build/1` /
    `builder_save/2`. The native handle is freed by the GC or eagerly via
    `PdfOxide.builder_close/1`.
    """
    defstruct [:ref]
  end

  defmodule PageBuilder do
    @moduledoc """
    A page-creation builder handle (NIF resource) started off a
    `DocumentBuilder`. Emit content with the fluent `page_*` ops, then commit it
    to its parent with `PdfOxide.page_done/1` (which consumes the handle) — or
    drop it with `PdfOxide.page_close/1`.
    """
    defstruct [:ref]
  end

  defmodule EmbeddedFont do
    @moduledoc """
    A loaded TTF/OTF font handle (NIF resource) for embedding via
    `PdfOxide.builder_register_embedded_font/3`. A successful register *consumes*
    the font; the wrapper is left holding a freed handle and must not be used
    again. Otherwise free it with `PdfOxide.font_close/1` (also GC-freed).
    """
    defstruct [:ref]
  end

  defmodule Certificate do
    @moduledoc """
    Signing credentials (X.509 certificate + private key) as a native handle.
    Load with `PdfOxide.certificate_from_bytes/2` (PKCS#12) or
    `PdfOxide.certificate_from_pem/2`, read its accessors, and free with
    `PdfOxide.certificate_close/1` (also GC-freed).
    """
    defstruct [:ref]
  end

  defmodule SignatureInfo do
    @moduledoc """
    A parsed PDF signature (native handle). Read its signer/time/reason metadata
    and verify it with `PdfOxide.signature_verify/1` /
    `PdfOxide.signature_verify_detached/2`. Free with
    `PdfOxide.signature_close/1` (also GC-freed).
    """
    defstruct [:ref]
  end

  defmodule Timestamp do
    @moduledoc """
    A parsed RFC 3161 timestamp token (native handle). Free with
    `PdfOxide.timestamp_close/1` (also GC-freed).
    """
    defstruct [:ref]
  end

  defmodule TsaClient do
    @moduledoc """
    An RFC 3161 Time-Stamping Authority client (native handle). Request tokens
    with `PdfOxide.tsa_request_timestamp/2` /
    `PdfOxide.tsa_request_timestamp_hash/3`. Free with
    `PdfOxide.tsa_close/1` (also GC-freed).
    """
    defstruct [:ref]
  end

  defmodule Dss do
    @moduledoc """
    A Document Security Store (native handle): the document-level certs/CRLs/OCSPs
    that back long-term-validation signatures. Free with `PdfOxide.dss_close/1`
    (also GC-freed).
    """
    defstruct [:ref]
  end

  defmodule PdfAResult do
    @moduledoc "Result of a PDF/A validation pass (native handle)."
    defstruct [:ref]
  end

  defmodule PdfUaResult do
    @moduledoc "Result of a PDF/UA accessibility validation pass (native handle)."
    defstruct [:ref]
  end

  defmodule PdfXResult do
    @moduledoc "Result of a PDF/X validation pass (native handle)."
    defstruct [:ref]
  end

  defmodule UaStats do
    @moduledoc "Accessibility element counts from a PDF/UA validation pass."
    defstruct [:struct, :images, :tables, :forms, :annotations, :pages]
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

  # ── EmbeddedFont ─────────────────────────────────────────────────────────────
  @doc "Load a TTF/OTF font from a file path into an `EmbeddedFont`."
  def font_from_file(path), do: wrap_font(Native.font_from_file(path))

  @doc """
  Load a TTF/OTF font from a binary into an `EmbeddedFont`. `name` may be an
  empty string to use the font's own PostScript name.
  """
  def font_from_bytes(bytes, name \\ ""), do: wrap_font(Native.font_from_bytes(bytes, name))

  @doc """
  Free an `EmbeddedFont`'s native handle now (idempotent). A no-op after a
  successful `builder_register_embedded_font/3` consumed the font.
  """
  def font_close(%EmbeddedFont{ref: ref}), do: Native.font_close(ref)

  # ── DocumentBuilder ──────────────────────────────────────────────────────────
  @doc "Create a new PDF-creation `DocumentBuilder`."
  def builder, do: wrap_doc_builder(Native.dbld_create())

  @doc "Set the document title."
  def builder_set_title(%DocumentBuilder{ref: ref}, title), do: Native.dbld_set_title(ref, title)
  @doc "Set the document author."
  def builder_set_author(%DocumentBuilder{ref: ref}, author),
    do: Native.dbld_set_author(ref, author)

  @doc "Set the document subject."
  def builder_set_subject(%DocumentBuilder{ref: ref}, subject),
    do: Native.dbld_set_subject(ref, subject)

  @doc "Set the document keywords (comma-separated)."
  def builder_set_keywords(%DocumentBuilder{ref: ref}, keywords),
    do: Native.dbld_set_keywords(ref, keywords)

  @doc "Set the creator application name."
  def builder_set_creator(%DocumentBuilder{ref: ref}, creator),
    do: Native.dbld_set_creator(ref, creator)

  @doc "Run JavaScript when the document is opened (`/OpenAction`)."
  def builder_on_open(%DocumentBuilder{ref: ref}, script), do: Native.dbld_on_open(ref, script)
  @doc "Set the document's natural language tag (e.g. \"en-US\")."
  def builder_language(%DocumentBuilder{ref: ref}, lang), do: Native.dbld_language(ref, lang)
  @doc "Enable PDF/UA-1 tagged-PDF mode."
  def builder_tagged_pdf_ua1(%DocumentBuilder{ref: ref}), do: Native.dbld_tagged_pdf_ua1(ref)

  @doc "Add a role-map entry: custom structure type → standard PDF structure type."
  def builder_role_map(%DocumentBuilder{ref: ref}, custom, standard),
    do: Native.dbld_role_map(ref, custom, standard)

  @doc """
  Register a TTF/OTF `EmbeddedFont` under `name`. On success the builder
  *consumes* the font handle — do not use or `font_close/1` it afterwards.
  """
  def builder_register_embedded_font(%DocumentBuilder{ref: ref}, name, %EmbeddedFont{ref: fref}),
    do: Native.dbld_register_embedded_font(ref, name, fref)

  @doc "Start a US Letter page, returning a `PageBuilder`."
  def builder_letter_page(%DocumentBuilder{ref: ref}),
    do: wrap_page_builder(Native.dbld_letter_page(ref))

  @doc "Start an A4 page, returning a `PageBuilder`."
  def builder_a4_page(%DocumentBuilder{ref: ref}), do: wrap_page_builder(Native.dbld_a4_page(ref))

  @doc "Start a custom `width`×`height` (PDF points) page, returning a `PageBuilder`."
  def builder_page(%DocumentBuilder{ref: ref}, width, height),
    do: wrap_page_builder(Native.dbld_page(ref, width * 1.0, height * 1.0))

  @doc "Build the PDF and return its bytes."
  def builder_build(%DocumentBuilder{ref: ref}), do: Native.dbld_build(ref)
  @doc "Build and save the PDF to `path`."
  def builder_save(%DocumentBuilder{ref: ref}, path), do: Native.dbld_save(ref, path)

  @doc "Build and save the PDF AES-256 encrypted to `path`."
  def builder_save_encrypted(%DocumentBuilder{ref: ref}, path, user_password, owner_password),
    do: Native.dbld_save_encrypted(ref, path, user_password, owner_password)

  @doc "Build the PDF AES-256 encrypted and return its bytes."
  def builder_to_bytes_encrypted(%DocumentBuilder{ref: ref}, user_password, owner_password),
    do: Native.dbld_to_bytes_encrypted(ref, user_password, owner_password)

  @doc "Free the builder's native handle now (idempotent)."
  def builder_close(%DocumentBuilder{ref: ref}), do: Native.dbld_close(ref)

  # ── PageBuilder ──────────────────────────────────────────────────────────────
  @doc "Set the font + size for subsequent text on this page."
  def page_font(%PageBuilder{ref: ref}, name, size), do: Native.pbld_font(ref, name, size * 1.0)
  @doc "Move the cursor to absolute `(x, y)` (PDF points, from lower-left)."
  def page_at(%PageBuilder{ref: ref}, x, y), do: Native.pbld_at(ref, x * 1.0, y * 1.0)
  @doc "Emit a line of text at the cursor, then advance one line-height."
  def page_text(%PageBuilder{ref: ref}, text), do: Native.pbld_text(ref, text)
  @doc "Emit a heading with `level` (1–6) and `text`."
  def page_heading(%PageBuilder{ref: ref}, level, text), do: Native.pbld_heading(ref, level, text)
  @doc "Emit a paragraph with automatic line wrapping."
  def page_paragraph(%PageBuilder{ref: ref}, text), do: Native.pbld_paragraph(ref, text)
  @doc "Advance the cursor down by `points`."
  def page_space(%PageBuilder{ref: ref}, points), do: Native.pbld_space(ref, points * 1.0)
  @doc "Draw a horizontal rule across the page."
  def page_horizontal_rule(%PageBuilder{ref: ref}), do: Native.pbld_horizontal_rule(ref)
  @doc "Attach a URL link to the previously-emitted text element."
  def page_link_url(%PageBuilder{ref: ref}, url), do: Native.pbld_link_url(ref, url)
  @doc "Link the previous text to an internal (0-based) `page_index`."
  def page_link_page(%PageBuilder{ref: ref}, page_index),
    do: Native.pbld_link_page(ref, page_index)

  @doc "Link the previous text to a named destination."
  def page_link_named(%PageBuilder{ref: ref}, destination),
    do: Native.pbld_link_named(ref, destination)

  @doc "Link the previous text to a JavaScript action."
  def page_link_javascript(%PageBuilder{ref: ref}, script),
    do: Native.pbld_link_javascript(ref, script)

  @doc "Run JavaScript when this page is opened (`/AA /O`)."
  def page_on_open(%PageBuilder{ref: ref}, script), do: Native.pbld_on_open(ref, script)
  @doc "Run JavaScript when this page is closed (`/AA /C`)."
  def page_on_close(%PageBuilder{ref: ref}, script), do: Native.pbld_on_close(ref, script)
  @doc "Set a keystroke JS action on the most-recently-added form field."
  def page_field_keystroke(%PageBuilder{ref: ref}, script),
    do: Native.pbld_field_keystroke(ref, script)

  @doc "Set a format JS action on the most-recently-added form field."
  def page_field_format(%PageBuilder{ref: ref}, script), do: Native.pbld_field_format(ref, script)
  @doc "Set a validate JS action on the most-recently-added form field."
  def page_field_validate(%PageBuilder{ref: ref}, script),
    do: Native.pbld_field_validate(ref, script)

  @doc "Set a calculate JS action on the most-recently-added form field."
  def page_field_calculate(%PageBuilder{ref: ref}, script),
    do: Native.pbld_field_calculate(ref, script)

  @doc "Highlight the previous text with an RGB colour (channels 0.0–1.0)."
  def page_highlight(%PageBuilder{ref: ref}, r, g, b),
    do: Native.pbld_highlight(ref, r * 1.0, g * 1.0, b * 1.0)

  @doc "Underline the previous text (RGB 0.0–1.0)."
  def page_underline(%PageBuilder{ref: ref}, r, g, b),
    do: Native.pbld_underline(ref, r * 1.0, g * 1.0, b * 1.0)

  @doc "Strikeout the previous text (RGB 0.0–1.0)."
  def page_strikeout(%PageBuilder{ref: ref}, r, g, b),
    do: Native.pbld_strikeout(ref, r * 1.0, g * 1.0, b * 1.0)

  @doc "Squiggly-underline the previous text (RGB 0.0–1.0)."
  def page_squiggly(%PageBuilder{ref: ref}, r, g, b),
    do: Native.pbld_squiggly(ref, r * 1.0, g * 1.0, b * 1.0)

  @doc "Attach a sticky-note annotation to the previous text."
  def page_sticky_note(%PageBuilder{ref: ref}, text), do: Native.pbld_sticky_note(ref, text)
  @doc "Place a free-standing sticky note at absolute `(x, y)`."
  def page_sticky_note_at(%PageBuilder{ref: ref}, x, y, text),
    do: Native.pbld_sticky_note_at(ref, x * 1.0, y * 1.0, text)

  @doc "Apply a text watermark to the entire page."
  def page_watermark(%PageBuilder{ref: ref}, text), do: Native.pbld_watermark(ref, text)
  @doc "Apply the standard \"CONFIDENTIAL\" diagonal watermark."
  def page_watermark_confidential(%PageBuilder{ref: ref}),
    do: Native.pbld_watermark_confidential(ref)

  @doc "Apply the standard \"DRAFT\" diagonal watermark."
  def page_watermark_draft(%PageBuilder{ref: ref}), do: Native.pbld_watermark_draft(ref)
  @doc "Attach a standard stamp annotation by `type_name`."
  def page_stamp(%PageBuilder{ref: ref}, type_name), do: Native.pbld_stamp(ref, type_name)

  @doc "Place a free-flowing text annotation inside the given rectangle."
  def page_freetext(%PageBuilder{ref: ref}, x, y, w, h, text),
    do: Native.pbld_freetext(ref, x * 1.0, y * 1.0, w * 1.0, h * 1.0, text)

  @doc """
  Add a single-line text form field. `default_value` may be an empty string for
  a blank field.
  """
  def page_text_field(%PageBuilder{ref: ref}, name, x, y, w, h, default_value \\ ""),
    do: Native.pbld_text_field(ref, name, x * 1.0, y * 1.0, w * 1.0, h * 1.0, default_value)

  @doc "Add a checkbox form field. `checked` is non-zero for initially-ticked."
  def page_checkbox(%PageBuilder{ref: ref}, name, x, y, w, h, checked),
    do: Native.pbld_checkbox(ref, name, x * 1.0, y * 1.0, w * 1.0, h * 1.0, checked)

  @doc """
  Add a dropdown combo-box. `options` is a list of strings; `selected` may be an
  empty string for no initial selection.
  """
  def page_combo_box(%PageBuilder{ref: ref}, name, x, y, w, h, options, selected \\ "")
      when is_list(options),
      do: Native.pbld_combo_box(ref, name, x * 1.0, y * 1.0, w * 1.0, h * 1.0, options, selected)

  @doc """
  Add a radio-button group. `values`/`xs`/`ys`/`ws`/`hs` are parallel lists
  describing each button; `selected` may be an empty string.
  """
  def page_radio_group(%PageBuilder{ref: ref}, name, values, xs, ys, ws, hs, selected \\ "")
      when is_list(values) do
    f = fn list -> Enum.map(list, &(&1 * 1.0)) end
    Native.pbld_radio_group(ref, name, values, f.(xs), f.(ys), f.(ws), f.(hs), selected)
  end

  @doc "Add a clickable push button with a visible caption."
  def page_push_button(%PageBuilder{ref: ref}, name, x, y, w, h, caption),
    do: Native.pbld_push_button(ref, name, x * 1.0, y * 1.0, w * 1.0, h * 1.0, caption)

  @doc "Add an unsigned signature placeholder field."
  def page_signature_field(%PageBuilder{ref: ref}, name, x, y, w, h),
    do: Native.pbld_signature_field(ref, name, x * 1.0, y * 1.0, w * 1.0, h * 1.0)

  @doc "Add a footnote: inline `ref_mark` + page-end `note_text`."
  def page_footnote(%PageBuilder{ref: ref}, ref_mark, note_text),
    do: Native.pbld_footnote(ref, ref_mark, note_text)

  @doc "Lay out `text` across `column_count` balanced columns with `gap_pt` between."
  def page_columns(%PageBuilder{ref: ref}, column_count, gap_pt, text),
    do: Native.pbld_columns(ref, column_count, gap_pt * 1.0, text)

  @doc "Emit `text` inline at the cursor (advances x, not y)."
  def page_inline(%PageBuilder{ref: ref}, text), do: Native.pbld_inline(ref, text)
  @doc "Emit an inline bold run."
  def page_inline_bold(%PageBuilder{ref: ref}, text), do: Native.pbld_inline_bold(ref, text)
  @doc "Emit an inline italic run."
  def page_inline_italic(%PageBuilder{ref: ref}, text), do: Native.pbld_inline_italic(ref, text)
  @doc "Emit an inline coloured run (RGB 0.0–1.0)."
  def page_inline_color(%PageBuilder{ref: ref}, r, g, b, text),
    do: Native.pbld_inline_color(ref, r * 1.0, g * 1.0, b * 1.0, text)

  @doc "Advance the cursor one line-height and reset x."
  def page_newline(%PageBuilder{ref: ref}), do: Native.pbld_newline(ref)

  @doc """
  Place a 1-D barcode. `barcode_type`: 0=Code128 1=Code39 2=EAN13 3=EAN8
  4=UPCA 5=ITF 6=Code93 7=Codabar.
  """
  def page_barcode_1d(%PageBuilder{ref: ref}, barcode_type, data, x, y, w, h),
    do: Native.pbld_barcode_1d(ref, barcode_type, data, x * 1.0, y * 1.0, w * 1.0, h * 1.0)

  @doc "Place a QR-code image (square `size`×`size` points)."
  def page_barcode_qr(%PageBuilder{ref: ref}, data, x, y, size),
    do: Native.pbld_barcode_qr(ref, data, x * 1.0, y * 1.0, size * 1.0)

  @doc "Embed an image (raw JPEG/PNG bytes) at `(x, y, w, h)`."
  def page_image(%PageBuilder{ref: ref}, bytes, x, y, w, h),
    do: Native.pbld_image(ref, bytes, x * 1.0, y * 1.0, w * 1.0, h * 1.0)

  @doc "Embed an image with accessibility `alt_text` at `(x, y, w, h)`."
  def page_image_with_alt(%PageBuilder{ref: ref}, bytes, x, y, w, h, alt_text),
    do: Native.pbld_image_with_alt(ref, bytes, x * 1.0, y * 1.0, w * 1.0, h * 1.0, alt_text)

  @doc "Embed a decorative image as an `/Artifact` (no alt text)."
  def page_image_artifact(%PageBuilder{ref: ref}, bytes, x, y, w, h),
    do: Native.pbld_image_artifact(ref, bytes, x * 1.0, y * 1.0, w * 1.0, h * 1.0)

  @doc "Draw a stroked rectangle outline (1pt black)."
  def page_rect(%PageBuilder{ref: ref}, x, y, w, h),
    do: Native.pbld_rect(ref, x * 1.0, y * 1.0, w * 1.0, h * 1.0)

  @doc "Draw a filled rectangle in RGB colour (channels 0–1)."
  def page_filled_rect(%PageBuilder{ref: ref}, x, y, w, h, r, g, b),
    do:
      Native.pbld_filled_rect(ref, x * 1.0, y * 1.0, w * 1.0, h * 1.0, r * 1.0, g * 1.0, b * 1.0)

  @doc "Draw a line from `(x1, y1)` to `(x2, y2)` (1pt black)."
  def page_line(%PageBuilder{ref: ref}, x1, y1, x2, y2),
    do: Native.pbld_line(ref, x1 * 1.0, y1 * 1.0, x2 * 1.0, y2 * 1.0)

  @doc "Buffer a stroked rectangle with `width` + RGB colour."
  def page_stroke_rect(%PageBuilder{ref: ref}, x, y, w, h, width, r, g, b),
    do:
      Native.pbld_stroke_rect(
        ref,
        x * 1.0,
        y * 1.0,
        w * 1.0,
        h * 1.0,
        width * 1.0,
        r * 1.0,
        g * 1.0,
        b * 1.0
      )

  @doc "Buffer a stroked line with `width` + RGB colour."
  def page_stroke_line(%PageBuilder{ref: ref}, x1, y1, x2, y2, width, r, g, b),
    do:
      Native.pbld_stroke_line(
        ref,
        x1 * 1.0,
        y1 * 1.0,
        x2 * 1.0,
        y2 * 1.0,
        width * 1.0,
        r * 1.0,
        g * 1.0,
        b * 1.0
      )

  @doc """
  Buffer a dashed stroked rectangle. `dash` is a list of alternating on/off
  lengths (empty = solid); `phase` is the starting offset.
  """
  def page_stroke_rect_dashed(%PageBuilder{ref: ref}, x, y, w, h, width, r, g, b, dash, phase)
      when is_list(dash),
      do:
        Native.pbld_stroke_rect_dashed(
          ref,
          x * 1.0,
          y * 1.0,
          w * 1.0,
          h * 1.0,
          width * 1.0,
          r * 1.0,
          g * 1.0,
          b * 1.0,
          Enum.map(dash, &(&1 * 1.0)),
          phase * 1.0
        )

  @doc """
  Buffer a dashed stroked line. `dash` is a list of alternating on/off lengths
  (empty = solid); `phase` is the starting offset.
  """
  def page_stroke_line_dashed(%PageBuilder{ref: ref}, x1, y1, x2, y2, width, r, g, b, dash, phase)
      when is_list(dash),
      do:
        Native.pbld_stroke_line_dashed(
          ref,
          x1 * 1.0,
          y1 * 1.0,
          x2 * 1.0,
          y2 * 1.0,
          width * 1.0,
          r * 1.0,
          g * 1.0,
          b * 1.0,
          Enum.map(dash, &(&1 * 1.0)),
          phase * 1.0
        )

  @doc "Buffer text inside a rect. `align`: 0=Left, 1=Center, 2=Right."
  def page_text_in_rect(%PageBuilder{ref: ref}, x, y, w, h, text, align),
    do: Native.pbld_text_in_rect(ref, x * 1.0, y * 1.0, w * 1.0, h * 1.0, text, align)

  @doc "Buffer a same-size page transition; later ops land on the new page."
  def page_new_page_same_size(%PageBuilder{ref: ref}), do: Native.pbld_new_page_same_size(ref)

  @doc """
  Buffer a buffered table. `widths`/`aligns` are length-`n_columns` lists
  (`aligns`: 0/1/2); `cells` is a row-major list of `n_rows * n_columns`
  strings. `has_header` promotes the first row.
  """
  def page_table(%PageBuilder{ref: ref}, n_columns, widths, aligns, n_rows, cells, has_header)
      when is_list(widths) and is_list(aligns) and is_list(cells),
      do:
        Native.pbld_table(
          ref,
          n_columns,
          Enum.map(widths, &(&1 * 1.0)),
          aligns,
          n_rows,
          cells,
          has_header
        )

  @doc """
  Open a streaming table. `headers` is a list of column header strings;
  `widths`/`aligns` are parallel lists. `repeat_header` is non-zero to repeat
  the header on each page.
  """
  def page_streaming_table_begin(
        %PageBuilder{ref: ref},
        n_columns,
        headers,
        widths,
        aligns,
        repeat_header
      )
      when is_list(headers) and is_list(widths) and is_list(aligns),
      do:
        Native.pbld_streaming_table_begin(
          ref,
          n_columns,
          headers,
          Enum.map(widths, &(&1 * 1.0)),
          aligns,
          repeat_header
        )

  @doc """
  Open a streaming table with a column-width `mode` (0=Fixed, 1=Sample,
  2=AutoAll) plus sampling/rowspan parameters.
  """
  def page_streaming_table_begin_v2(
        %PageBuilder{ref: ref},
        n_columns,
        headers,
        widths,
        aligns,
        repeat_header,
        mode,
        sample_rows,
        min_w,
        max_w,
        max_rowspan
      )
      when is_list(headers) and is_list(widths) and is_list(aligns),
      do:
        Native.pbld_streaming_table_begin_v2(
          ref,
          n_columns,
          headers,
          Enum.map(widths, &(&1 * 1.0)),
          aligns,
          repeat_header,
          mode,
          sample_rows,
          min_w * 1.0,
          max_w * 1.0,
          max_rowspan
        )

  @doc "Set the auto-flush batch size for the open streaming table (0 = default 256)."
  def page_streaming_table_set_batch_size(%PageBuilder{ref: ref}, batch_size),
    do: Native.pbld_streaming_table_set_batch_size(ref, batch_size)

  @doc "Rows pushed since the last batch boundary."
  def page_streaming_table_pending_row_count(%PageBuilder{ref: ref}),
    do: Native.pbld_streaming_table_pending_row_count(ref)

  @doc "Complete batches recorded so far."
  def page_streaming_table_batch_count(%PageBuilder{ref: ref}),
    do: Native.pbld_streaming_table_batch_count(ref)

  @doc "Push one row (list of cell strings) into the open streaming table."
  def page_streaming_table_push_row(%PageBuilder{ref: ref}, cells) when is_list(cells),
    do: Native.pbld_streaming_table_push_row(ref, cells)

  @doc """
  Push one row with per-cell `rowspans` (list of ints, empty = all rowspan=1).
  """
  def page_streaming_table_push_row_v2(%PageBuilder{ref: ref}, cells, rowspans)
      when is_list(cells) and is_list(rowspans),
      do: Native.pbld_streaming_table_push_row_v2(ref, cells, rowspans)

  @doc "Mark a batch boundary in the open streaming table."
  def page_streaming_table_flush(%PageBuilder{ref: ref}),
    do: Native.pbld_streaming_table_flush(ref)

  @doc "Close the open streaming table."
  def page_streaming_table_finish(%PageBuilder{ref: ref}),
    do: Native.pbld_streaming_table_finish(ref)

  @doc """
  Commit this page's buffered ops to its parent builder. *Consumes* the handle —
  the wrapper must not be used (or `page_close/1`-d) afterwards.
  """
  def page_done(%PageBuilder{ref: ref}), do: Native.pbld_done(ref)

  @doc "Drop an uncommitted page builder's native handle now (idempotent)."
  def page_close(%PageBuilder{ref: ref}), do: Native.pbld_close(ref)

  # ── Certificate (phase 6) ────────────────────────────────────────────────────
  @doc """
  Load signing credentials from a PKCS#12 (.p12/.pfx) binary, decrypted with
  `password` (empty string for none).
  """
  def certificate_from_bytes(bytes, password \\ "") when is_binary(bytes),
    do: wrap_certificate(Native.cert_load_from_bytes(bytes, password))

  @doc "Load signing credentials from PEM-encoded certificate + private-key strings."
  def certificate_from_pem(cert_pem, key_pem),
    do: wrap_certificate(Native.cert_load_from_pem(cert_pem, key_pem))

  @doc "The certificate's subject distinguished name."
  def certificate_subject(%Certificate{ref: ref}), do: Native.cert_get_subject(ref)
  @doc "The certificate's issuer distinguished name."
  def certificate_issuer(%Certificate{ref: ref}), do: Native.cert_get_issuer(ref)
  @doc "The certificate's serial number (string)."
  def certificate_serial(%Certificate{ref: ref}), do: Native.cert_get_serial(ref)

  @doc """
  The certificate's validity window as `{:ok, {not_before, not_after}}` (Unix
  epoch seconds).
  """
  def certificate_validity(%Certificate{ref: ref}), do: Native.cert_get_validity(ref)

  @doc "Whether the certificate is currently valid (1 = valid; see C ABI codes)."
  def certificate_valid?(%Certificate{ref: ref}), do: Native.cert_is_valid(ref)

  @doc "Free a certificate's native handle now (idempotent)."
  def certificate_close(%Certificate{ref: ref}), do: Native.cert_close(ref)

  # ── Signing (phase 6) ─────────────────────────────────────────────────────────
  @doc "Sign raw PDF `bytes` with `certificate`, returning the signed PDF binary."
  def sign_bytes(bytes, %Certificate{ref: cref}, reason \\ "", location \\ "")
      when is_binary(bytes),
      do: Native.sign_bytes(bytes, cref, reason, location)

  @doc """
  PAdES-sign raw PDF `bytes`. `level`: 0=B-B 1=B-T 2=B-LT. `tsa_url` may be an
  empty string for B-B. `certs`/`crls`/`ocsps` are lists of DER binaries
  carrying B-LT revocation material (empty lists for B-B/B-T).
  """
  def sign_bytes_pades(bytes, %Certificate{ref: cref}, level, tsa_url \\ "", opts \\ [])
      when is_binary(bytes) and is_integer(level) and is_list(opts) do
    Native.sign_bytes_pades(
      bytes,
      cref,
      level,
      tsa_url,
      Keyword.get(opts, :reason, ""),
      Keyword.get(opts, :location, ""),
      Keyword.get(opts, :certs, []),
      Keyword.get(opts, :crls, []),
      Keyword.get(opts, :ocsps, [])
    )
  end

  @doc """
  Struct-options variant of `sign_bytes_pades/5` — marshals the same parameters
  into the C `PadesSignOptionsC` struct. Returns the signed PDF binary.
  """
  def sign_bytes_pades_opts(bytes, %Certificate{ref: cref}, level, tsa_url \\ "", opts \\ [])
      when is_binary(bytes) and is_integer(level) and is_list(opts) do
    Native.sign_bytes_pades_opts(
      bytes,
      cref,
      level,
      tsa_url,
      Keyword.get(opts, :reason, ""),
      Keyword.get(opts, :location, ""),
      Keyword.get(opts, :certs, []),
      Keyword.get(opts, :crls, []),
      Keyword.get(opts, :ocsps, [])
    )
  end

  # ── SignatureInfo (phase 6) ────────────────────────────────────────────────────
  @doc "The signer's name."
  def signature_signer_name(%SignatureInfo{ref: ref}), do: Native.sig_get_signer_name(ref)
  @doc "The stated signing reason."
  def signature_reason(%SignatureInfo{ref: ref}), do: Native.sig_get_signing_reason(ref)
  @doc "The stated signing location."
  def signature_location(%SignatureInfo{ref: ref}), do: Native.sig_get_signing_location(ref)
  @doc "The signing time as `{:ok, epoch_seconds}`."
  def signature_time(%SignatureInfo{ref: ref}), do: Native.sig_get_signing_time(ref)

  @doc "The signer's `Certificate` handle."
  def signature_certificate(%SignatureInfo{ref: ref}),
    do: wrap_certificate(Native.sig_get_certificate(ref))

  @doc "The signature's PAdES level as `{:ok, level}` (-1 if unknown)."
  def signature_pades_level(%SignatureInfo{ref: ref}), do: Native.sig_get_pades_level(ref)
  @doc "Whether the signature carries an embedded timestamp."
  def signature_has_timestamp?(%SignatureInfo{ref: ref}), do: Native.sig_has_timestamp(ref)

  @doc "The signature's embedded `Timestamp` handle."
  def signature_timestamp(%SignatureInfo{ref: ref}),
    do: wrap_timestamp(Native.sig_get_timestamp(ref))

  @doc "Attach `timestamp` to the signature; returns `{:ok, bool}`."
  def signature_add_timestamp(%SignatureInfo{ref: ref}, %Timestamp{ref: tref}),
    do: Native.sig_add_timestamp(ref, tref)

  @doc "Run the signer-attributes crypto check; returns `{:ok, code}` (1/0/-1)."
  def signature_verify(%SignatureInfo{ref: ref}), do: Native.sig_verify(ref)

  @doc """
  Verify the signature end-to-end against the full PDF `bytes`; returns
  `{:ok, code}` (1/0/-1).
  """
  def signature_verify_detached(%SignatureInfo{ref: ref}, bytes) when is_binary(bytes),
    do: Native.sig_verify_detached(ref, bytes)

  @doc "Free a signature's native handle now (idempotent)."
  def signature_close(%SignatureInfo{ref: ref}), do: Native.sig_close(ref)

  # ── Timestamp (phase 6) ────────────────────────────────────────────────────────
  @doc "Parse a DER-encoded RFC 3161 timestamp token into a `Timestamp`."
  def timestamp_parse(bytes) when is_binary(bytes),
    do: wrap_timestamp(Native.ts_parse(bytes))

  @doc "The raw DER token bytes."
  def timestamp_token(%Timestamp{ref: ref}), do: Native.ts_get_token(ref)
  @doc "The message-imprint hash bytes."
  def timestamp_message_imprint(%Timestamp{ref: ref}), do: Native.ts_get_message_imprint(ref)
  @doc "The timestamp time as `{:ok, epoch_seconds}`."
  def timestamp_time(%Timestamp{ref: ref}), do: Native.ts_get_time(ref)
  @doc "The timestamp serial number (string)."
  def timestamp_serial(%Timestamp{ref: ref}), do: Native.ts_get_serial(ref)
  @doc "The issuing TSA name."
  def timestamp_tsa_name(%Timestamp{ref: ref}), do: Native.ts_get_tsa_name(ref)
  @doc "The timestamp policy OID (string)."
  def timestamp_policy_oid(%Timestamp{ref: ref}), do: Native.ts_get_policy_oid(ref)
  @doc "The hash-algorithm code as `{:ok, code}`."
  def timestamp_hash_algorithm(%Timestamp{ref: ref}), do: Native.ts_get_hash_algorithm(ref)
  @doc "Verify the timestamp token; returns `{:ok, bool}`."
  def timestamp_verify(%Timestamp{ref: ref}), do: Native.ts_verify(ref)

  @doc "Free a timestamp's native handle now (idempotent)."
  def timestamp_close(%Timestamp{ref: ref}), do: Native.ts_close(ref)

  # ── TsaClient (phase 6) ────────────────────────────────────────────────────────
  @doc """
  Create an RFC 3161 TSA client for `url`. `opts`: `:username`, `:password`
  (empty for none), `:timeout` (seconds), `:hash_algo`, `:use_nonce`,
  `:cert_req`.
  """
  def tsa_client(url, opts \\ []) when is_list(opts) do
    wrap_tsa(
      Native.tsa_create(
        url,
        Keyword.get(opts, :username, ""),
        Keyword.get(opts, :password, ""),
        Keyword.get(opts, :timeout, 30),
        Keyword.get(opts, :hash_algo, 0),
        Keyword.get(opts, :use_nonce, true),
        Keyword.get(opts, :cert_req, true)
      )
    )
  end

  @doc "Request a timestamp over `data`, returning a `Timestamp`."
  def tsa_request_timestamp(%TsaClient{ref: ref}, data) when is_binary(data),
    do: wrap_timestamp(Native.tsa_request_timestamp(ref, data))

  @doc "Request a timestamp over a precomputed `hash` (with `hash_algo`)."
  def tsa_request_timestamp_hash(%TsaClient{ref: ref}, hash, hash_algo)
      when is_binary(hash) and is_integer(hash_algo),
      do: wrap_timestamp(Native.tsa_request_timestamp_hash(ref, hash, hash_algo))

  @doc "Free a TSA client's native handle now (idempotent)."
  def tsa_close(%TsaClient{ref: ref}), do: Native.tsa_close(ref)

  # ── Dss (phase 6) ──────────────────────────────────────────────────────────────
  @doc "Number of certs in the DSS."
  def dss_cert_count(%Dss{ref: ref}), do: Native.dss_cert_count(ref)
  @doc "Number of CRLs in the DSS."
  def dss_crl_count(%Dss{ref: ref}), do: Native.dss_crl_count(ref)
  @doc "Number of OCSP responses in the DSS."
  def dss_ocsp_count(%Dss{ref: ref}), do: Native.dss_ocsp_count(ref)
  @doc "Number of VRI (validation-related-info) entries in the DSS."
  def dss_vri_count(%Dss{ref: ref}), do: Native.dss_vri_count(ref)
  @doc "The `index`-th DSS cert as `{:ok, der_bytes}`."
  def dss_cert(%Dss{ref: ref}, index), do: Native.dss_get_cert(ref, index)
  @doc "The `index`-th DSS CRL as `{:ok, der_bytes}`."
  def dss_crl(%Dss{ref: ref}, index), do: Native.dss_get_crl(ref, index)
  @doc "The `index`-th DSS OCSP response as `{:ok, der_bytes}`."
  def dss_ocsp(%Dss{ref: ref}, index), do: Native.dss_get_ocsp(ref, index)

  @doc "Free a DSS native handle now (idempotent)."
  def dss_close(%Dss{ref: ref}), do: Native.dss_close(ref)

  # ── Validation (phase 6) ───────────────────────────────────────────────────────
  @doc "Validate a `Document` against PDF/A `level`, returning a `PdfAResult`."
  def validate_pdf_a(%Document{ref: ref}, level),
    do: wrap_pdf_a(Native.validate_pdf_a(ref, level))

  @doc "Validate a `Document` against PDF/UA `level`, returning a `PdfUaResult`."
  def validate_pdf_ua(%Document{ref: ref}, level),
    do: wrap_pdf_ua(Native.validate_pdf_ua(ref, level))

  @doc "Validate a `Document` against PDF/X `level`, returning a `PdfXResult`."
  def validate_pdf_x(%Document{ref: ref}, level),
    do: wrap_pdf_x(Native.validate_pdf_x(ref, level))

  @doc "Whether the PDF/A result is compliant; returns `{:ok, bool}`."
  def pdf_a_compliant?(%PdfAResult{ref: ref}), do: Native.pdf_a_is_compliant(ref)
  @doc "PDF/A validation errors as a list of strings."
  def pdf_a_errors(%PdfAResult{ref: ref}),
    do: result_strings(ref, &Native.pdf_a_error_count/1, &Native.pdf_a_get_error/2)

  @doc "PDF/A validation warnings count (no warning accessor in the C ABI)."
  def pdf_a_warning_count(%PdfAResult{ref: ref}), do: Native.pdf_a_warning_count(ref)
  @doc "Free a PDF/A result's native handle now (idempotent)."
  def pdf_a_close(%PdfAResult{ref: ref}), do: Native.pdf_a_close(ref)

  @doc "Whether the PDF/UA result is accessible; returns `{:ok, bool}`."
  def pdf_ua_accessible?(%PdfUaResult{ref: ref}), do: Native.pdf_ua_is_accessible(ref)
  @doc "PDF/UA validation errors as a list of strings."
  def pdf_ua_errors(%PdfUaResult{ref: ref}),
    do: result_strings(ref, &Native.pdf_ua_error_count/1, &Native.pdf_ua_get_error/2)

  @doc "PDF/UA validation warnings as a list of strings."
  def pdf_ua_warnings(%PdfUaResult{ref: ref}),
    do: result_strings(ref, &Native.pdf_ua_warning_count/1, &Native.pdf_ua_get_warning/2)

  @doc "Accessibility element counts as a `{:ok, %UaStats{}}`."
  def pdf_ua_stats(%PdfUaResult{ref: ref}) do
    with {:ok, {s, im, t, f, an, pg}} <- Native.pdf_ua_get_stats(ref) do
      {:ok, %UaStats{struct: s, images: im, tables: t, forms: f, annotations: an, pages: pg}}
    end
  end

  @doc "Free a PDF/UA result's native handle now (idempotent)."
  def pdf_ua_close(%PdfUaResult{ref: ref}), do: Native.pdf_ua_close(ref)

  @doc "Whether the PDF/X result is compliant; returns `{:ok, bool}`."
  def pdf_x_compliant?(%PdfXResult{ref: ref}), do: Native.pdf_x_is_compliant(ref)
  @doc "PDF/X validation errors as a list of strings."
  def pdf_x_errors(%PdfXResult{ref: ref}),
    do: result_strings(ref, &Native.pdf_x_error_count/1, &Native.pdf_x_get_error/2)

  @doc "Free a PDF/X result's native handle now (idempotent)."
  def pdf_x_close(%PdfXResult{ref: ref}), do: Native.pdf_x_close(ref)

  # ── log level (phase 6) ────────────────────────────────────────────────────────
  @doc "Set the global log level (0=Off 1=Error 2=Warn 3=Info 4=Debug 5=Trace)."
  def set_log_level(level) when is_integer(level), do: Native.oxide_set_log_level(level)
  @doc "Get the current global log level (0-5)."
  def get_log_level, do: Native.oxide_get_log_level()

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

  defp wrap_font({:ok, ref}), do: {:ok, %EmbeddedFont{ref: ref}}
  defp wrap_font(other), do: other
  defp wrap_doc_builder({:ok, ref}), do: {:ok, %DocumentBuilder{ref: ref}}
  defp wrap_doc_builder(other), do: other
  defp wrap_page_builder({:ok, ref}), do: {:ok, %PageBuilder{ref: ref}}
  defp wrap_page_builder(other), do: other

  defp wrap_certificate({:ok, ref}), do: {:ok, %Certificate{ref: ref}}
  defp wrap_certificate(other), do: other
  defp wrap_timestamp({:ok, ref}), do: {:ok, %Timestamp{ref: ref}}
  defp wrap_timestamp(other), do: other
  defp wrap_tsa({:ok, ref}), do: {:ok, %TsaClient{ref: ref}}
  defp wrap_tsa(other), do: other
  defp wrap_pdf_a({:ok, ref}), do: {:ok, %PdfAResult{ref: ref}}
  defp wrap_pdf_a(other), do: other
  defp wrap_pdf_ua({:ok, ref}), do: {:ok, %PdfUaResult{ref: ref}}
  defp wrap_pdf_ua(other), do: other
  defp wrap_pdf_x({:ok, ref}), do: {:ok, %PdfXResult{ref: ref}}
  defp wrap_pdf_x(other), do: other

  # Collect `count`/`get(index)` validation strings into a plain list. `count`
  # returns a bare int; each `get` returns {:ok, binary} | {:error, code}.
  defp result_strings(ref, count_fun, get_fun) do
    n = count_fun.(ref)
    n = if is_integer(n) and n > 0, do: n, else: 0

    for i <- 0..(n - 1)//1 do
      case get_fun.(ref, i) do
        {:ok, s} -> s
        _ -> ""
      end
    end
  end
end

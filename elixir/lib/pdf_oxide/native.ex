defmodule PdfOxide.Native do
  @moduledoc false
  # NIF loader. The compiled NIF (priv/pdf_oxide_nif) bridges to the pdf_oxide
  # C ABI; every text-producing function runs on a dirty CPU scheduler. All
  # functions here are replaced at load time — the bodies raise if the NIF
  # failed to load.
  @on_load :load_nif

  def load_nif do
    path = :filename.join(:code.priv_dir(:pdf_oxide), ~c"pdf_oxide_nif")
    :erlang.load_nif(path, 0)
  end

  defp nif_error, do: :erlang.nif_error(:nif_not_loaded)

  def from_markdown(_md), do: nif_error()
  def from_html(_html), do: nif_error()
  def from_text(_text), do: nif_error()
  def pdf_save(_pdf, _path), do: nif_error()
  def pdf_save_to_bytes(_pdf), do: nif_error()
  def doc_open(_path), do: nif_error()
  def doc_open_bytes(_bytes), do: nif_error()
  def doc_open_pw(_path, _pw), do: nif_error()
  def doc_page_count(_doc), do: nif_error()
  def doc_version(_doc), do: nif_error()
  def doc_is_encrypted(_doc), do: nif_error()
  def doc_has_structure_tree(_doc), do: nif_error()
  def doc_extract_text(_doc, _page), do: nif_error()
  def doc_to_plain_text(_doc, _page), do: nif_error()
  def doc_to_markdown(_doc, _page), do: nif_error()
  def doc_to_html(_doc, _page), do: nif_error()
  def doc_to_markdown_all(_doc), do: nif_error()
  def doc_to_html_all(_doc), do: nif_error()
  def doc_to_plain_text_all(_doc), do: nif_error()
  def doc_authenticate(_doc, _pw), do: nif_error()
  def doc_extract_structured_json(_doc, _page), do: nif_error()
  def doc_extract_chars(_doc, _page), do: nif_error()
  def doc_extract_words(_doc, _page), do: nif_error()
  def doc_extract_text_lines(_doc, _page), do: nif_error()
  def doc_extract_tables(_doc, _page), do: nif_error()
  def doc_embedded_fonts(_doc, _page), do: nif_error()
  def doc_embedded_images(_doc, _page), do: nif_error()
  def doc_page_annotations(_doc, _page), do: nif_error()
  def doc_extract_paths(_doc, _page), do: nif_error()
  def doc_search_page(_doc, _page, _term, _case_sensitive), do: nif_error()
  def doc_search_all(_doc, _term, _case_sensitive), do: nif_error()
  def doc_close(_doc), do: nif_error()
  def pdf_close(_pdf), do: nif_error()
end

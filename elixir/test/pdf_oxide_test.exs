# One test per public function — mirrors the api_coverage convention used by
# every pdf_oxide binding. Self-contained: builds its own PDF from Markdown.
defmodule PdfOxideTest do
  use ExUnit.Case

  defp sample_pdf do
    {:ok, p} =
      PdfOxide.from_markdown("# Coverage Doc\n\nAlpha bravo charlie. Some **bold** text.\n")

    {:ok, bytes} = PdfOxide.to_bytes(p)
    bytes
  end

  test "builder: from_markdown/from_html/from_text + to_bytes" do
    for {:ok, p} <- [
          PdfOxide.from_markdown("# md\n\nbody\n"),
          PdfOxide.from_html("<h1>h</h1><p>b</p>"),
          PdfOxide.from_text("plain text body")
        ] do
      assert {:ok, bytes} = PdfOxide.to_bytes(p)
      assert byte_size(bytes) > 100
    end
  end

  test "save" do
    path = Path.join(System.tmp_dir!(), "pdfoxide_ex_#{System.unique_integer([:positive])}.pdf")
    {:ok, p} = PdfOxide.from_markdown("# f\n\nx\n")
    assert :ok = PdfOxide.save(p, path)
    assert File.exists?(path)
    File.rm(path)
  end

  describe "document" do
    setup do
      {:ok, doc} = PdfOxide.open_from_bytes(sample_pdf())
      {:ok, doc: doc}
    end

    test "open_from_bytes + page_count", %{doc: doc} do
      assert {:ok, n} = PdfOxide.page_count(doc)
      assert n >= 1
    end

    test "open (path)" do
      path =
        Path.join(System.tmp_dir!(), "pdfoxide_ex_open_#{System.unique_integer([:positive])}.pdf")

      {:ok, p} = PdfOxide.from_markdown("# f\n\nx\n")
      :ok = PdfOxide.save(p, path)
      assert {:ok, doc} = PdfOxide.open(path)
      assert {:ok, n} = PdfOxide.page_count(doc)
      assert n >= 1
      File.rm(path)
    end

    test "version", %{doc: doc} do
      assert %{major: maj} = PdfOxide.version(doc)
      assert maj >= 1
    end

    test "close (idempotent) + open_with_password exists" do
      {:ok, doc} = PdfOxide.open_from_bytes(sample_pdf())
      assert :ok = PdfOxide.close(doc)
      assert function_exported?(PdfOxide, :open_with_password, 2)
    end

    test "encrypted?/structure_tree?", %{doc: doc} do
      assert PdfOxide.encrypted?(doc) == false
      _ = PdfOxide.structure_tree?(doc)
    end

    test "extraction", %{doc: doc} do
      assert {:ok, text} = PdfOxide.extract_text(doc, 0)
      assert text =~ "Alpha"
      assert {:ok, pt} = PdfOxide.to_plain_text(doc, 0)
      assert byte_size(pt) > 0
      assert {:ok, md} = PdfOxide.to_markdown(doc, 0)
      assert byte_size(md) > 0
      assert {:ok, html} = PdfOxide.to_html(doc, 0)
      assert html =~ "<"
      assert {:ok, mdall} = PdfOxide.to_markdown_all(doc)
      assert byte_size(mdall) > 0
      assert {:ok, htmlall} = PdfOxide.to_html_all(doc)
      assert byte_size(htmlall) > 0
      assert htmlall =~ "<"
      assert {:ok, ptall} = PdfOxide.to_plain_text_all(doc)
      assert byte_size(ptall) > 0
      assert {:ok, json} = PdfOxide.extract_structured_json(doc, 0)
      assert byte_size(json) > 0
    end

    test "element extraction (phase 1)", %{doc: doc} do
      assert {:ok, words} = PdfOxide.extract_words(doc, 0)
      assert is_list(words)
      assert length(words) > 0
      w = hd(words)
      assert is_binary(w.text) and byte_size(w.text) > 0
      assert %PdfOxide.Bbox{} = w.bbox
      assert is_number(w.bbox.x) and is_number(w.bbox.width)
      assert is_boolean(w.bold)

      assert {:ok, chars} = PdfOxide.extract_chars(doc, 0)
      assert is_list(chars)
      assert length(chars) > 0
      assert is_integer(hd(chars).character)
      assert %PdfOxide.Bbox{} = hd(chars).bbox

      assert {:ok, lines} = PdfOxide.extract_text_lines(doc, 0)
      assert is_list(lines)
      assert length(lines) > 0
      assert is_binary(hd(lines).text)
      assert is_integer(hd(lines).word_count)

      assert {:ok, tables} = PdfOxide.extract_tables(doc, 0)
      assert is_list(tables)
    end

    test "authenticate returns a bool", %{doc: doc} do
      assert {:ok, result} = PdfOxide.authenticate(doc, "")
      assert is_boolean(result)
    end

    test "page model", %{doc: doc} do
      page = PdfOxide.page(doc, 0)
      assert {:ok, text} = PdfOxide.text(page)
      assert text =~ "Alpha"
      assert {:ok, md} = PdfOxide.markdown(page)
      assert byte_size(md) > 0
      assert {:ok, html} = PdfOxide.html(page)
      assert byte_size(html) > 0
      assert {:ok, pt} = PdfOxide.plain_text(page)
      assert byte_size(pt) > 0
    end
  end

  test "error path: open nonexistent" do
    assert {:error, _code} = PdfOxide.open("/nonexistent/nope.pdf")
  end
end

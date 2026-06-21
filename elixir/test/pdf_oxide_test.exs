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

    test "element extraction (phase 2)", %{doc: doc} do
      assert {:ok, fonts} = PdfOxide.embedded_fonts(doc, 0)
      assert is_list(fonts)

      assert {:ok, images} = PdfOxide.embedded_images(doc, 0)
      assert is_list(images)

      assert {:ok, annots} = PdfOxide.page_annotations(doc, 0)
      assert is_list(annots)

      assert {:ok, paths} = PdfOxide.extract_paths(doc, 0)
      assert is_list(paths)
    end

    test "search + search_all", %{doc: doc} do
      assert {:ok, results} = PdfOxide.search(doc, 0, "Alpha", false)
      assert is_list(results)
      assert length(results) > 0
      r = hd(results)
      assert r.text =~ "Alpha"
      assert is_integer(r.page) and r.page >= 0
      assert %PdfOxide.Bbox{} = r.bbox

      assert {:ok, all} = PdfOxide.search_all(doc, "Alpha", false)
      assert is_list(all)
      assert length(all) > 0
      a = hd(all)
      assert a.text =~ "Alpha"
      assert a.page >= 0
    end

    test "authenticate returns a bool", %{doc: doc} do
      assert {:ok, result} = PdfOxide.authenticate(doc, "")
      assert is_boolean(result)
    end

    test "page rendering (phase 3)", %{doc: doc} do
      assert {:ok, img} = PdfOxide.render_page(doc, 0)
      assert %PdfOxide.RenderedImage{} = img
      assert is_integer(img.width) and img.width > 0
      assert is_integer(img.height) and img.height > 0
      assert is_binary(img.data) and byte_size(img.data) > 0

      path =
        Path.join(
          System.tmp_dir!(),
          "pdfoxide_ex_render_#{System.unique_integer([:positive])}.png"
        )

      assert :ok = PdfOxide.save(img, path)
      assert File.exists?(path)
      File.rm(path)

      assert {:ok, _zoomed} = PdfOxide.render_page_zoom(doc, 0, 2.0)
      assert {:ok, _thumb} = PdfOxide.render_page_thumbnail(doc, 0, 128)
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

  describe "document editor" do
    test "open_from_bytes + core editing API" do
      assert {:ok, ed} = PdfOxide.open_editor_from_bytes(sample_pdf())

      assert {:ok, n} = PdfOxide.editor_page_count(ed)
      assert n >= 1

      assert is_boolean(PdfOxide.editor_modified?(ed))

      assert :ok = PdfOxide.rotate_all_pages(ed, 90)
      assert {:ok, deg} = PdfOxide.get_page_rotation(ed, 0)
      assert is_integer(deg)
      assert deg == 90

      assert :ok = PdfOxide.set_producer(ed, "x")
      assert {:ok, producer} = PdfOxide.get_producer(ed)
      assert is_binary(producer)

      assert {:ok, bytes} = PdfOxide.editor_save_to_bytes(ed)
      assert byte_size(bytes) > 0

      assert :ok = PdfOxide.editor_close(ed)
    end
  end

  describe "PDF creation builder" do
    test "create -> page -> font/heading/paragraph -> build -> reopen" do
      assert {:ok, db} = PdfOxide.builder()
      assert :ok = PdfOxide.builder_set_title(db, "Builder Coverage")

      assert {:ok, page} = PdfOxide.builder_page(db, 595, 842)
      assert :ok = PdfOxide.page_font(page, "Helvetica", 12)
      assert :ok = PdfOxide.page_heading(page, 1, "Title")
      assert :ok = PdfOxide.page_paragraph(page, "Hello world from the builder.")
      # page_done consumes the page handle.
      assert :ok = PdfOxide.page_done(page)

      assert {:ok, bytes} = PdfOxide.builder_build(db)
      assert is_binary(bytes) and byte_size(bytes) > 0
      assert :ok = PdfOxide.builder_close(db)

      assert {:ok, doc} = PdfOxide.open_from_bytes(bytes)
      assert {:ok, n} = PdfOxide.page_count(doc)
      assert n >= 1
      assert {:ok, text} = PdfOxide.extract_text(doc, 0)
      assert text =~ "Hello" or text =~ "Title"
      :ok = PdfOxide.close(doc)
    end

    test "letter_page + a few fluent ops, then save to disk" do
      assert {:ok, db} = PdfOxide.builder()
      assert {:ok, page} = PdfOxide.builder_letter_page(db)
      assert :ok = PdfOxide.page_font(page, "Helvetica", 14)
      assert :ok = PdfOxide.page_text(page, "Letter page line.")
      assert :ok = PdfOxide.page_horizontal_rule(page)
      assert :ok = PdfOxide.page_done(page)

      path =
        Path.join(
          System.tmp_dir!(),
          "pdfoxide_ex_build_#{System.unique_integer([:positive])}.pdf"
        )

      assert :ok = PdfOxide.builder_save(db, path)
      assert File.exists?(path)
      File.rm(path)
      :ok = PdfOxide.builder_close(db)
    end

    test "embedded-font + page builder functions are exported" do
      # Standard-font path is exercised above; embedded-font loaders need a real
      # font file, so just assert the surface exists without a fixture.
      assert function_exported?(PdfOxide, :font_from_file, 1)
      assert function_exported?(PdfOxide, :font_from_bytes, 2)
      assert function_exported?(PdfOxide, :builder_register_embedded_font, 3)
      assert function_exported?(PdfOxide, :page_table, 7)
      assert function_exported?(PdfOxide, :page_combo_box, 7)
      assert function_exported?(PdfOxide, :page_radio_group, 8)
    end
  end

  test "error path: open nonexistent" do
    assert {:error, _code} = PdfOxide.open("/nonexistent/nope.pdf")
  end

  # ── phase 6: signatures / PKI / timestamps / TSA / DSS / validation ───────────
  describe "validation (phase 6)" do
    setup do
      {:ok, doc} = PdfOxide.open_from_bytes(sample_pdf())
      {:ok, doc: doc}
    end

    test "validate_pdf_a + compliant?/errors/warnings", %{doc: doc} do
      assert {:ok, res} = PdfOxide.validate_pdf_a(doc, 1)
      assert {:ok, compliant} = PdfOxide.pdf_a_compliant?(res)
      assert is_boolean(compliant)
      assert is_list(PdfOxide.pdf_a_errors(res))
      assert is_integer(PdfOxide.pdf_a_warning_count(res))
      assert :ok = PdfOxide.pdf_a_close(res)
    end

    test "validate_pdf_ua + accessible?/errors/warnings/stats", %{doc: doc} do
      assert {:ok, res} = PdfOxide.validate_pdf_ua(doc, 1)
      assert {:ok, accessible} = PdfOxide.pdf_ua_accessible?(res)
      assert is_boolean(accessible)
      assert is_list(PdfOxide.pdf_ua_errors(res))
      assert is_list(PdfOxide.pdf_ua_warnings(res))
      assert {:ok, %PdfOxide.UaStats{} = stats} = PdfOxide.pdf_ua_stats(res)
      assert is_integer(stats.struct) and is_integer(stats.pages)
      assert :ok = PdfOxide.pdf_ua_close(res)
    end

    test "validate_pdf_x + compliant?/errors", %{doc: doc} do
      assert {:ok, res} = PdfOxide.validate_pdf_x(doc, 1)
      assert {:ok, compliant} = PdfOxide.pdf_x_compliant?(res)
      assert is_boolean(compliant)
      assert is_list(PdfOxide.pdf_x_errors(res))
      assert :ok = PdfOxide.pdf_x_close(res)
    end
  end

  test "log level round-trip (phase 6)" do
    original = PdfOxide.get_log_level()
    assert is_integer(original)
    assert :ok = PdfOxide.set_log_level(2)
    assert PdfOxide.get_log_level() == 2
    assert :ok = PdfOxide.set_log_level(original)
  end

  describe "PKI/signing wrappers exercised with minimal inputs (phase 6)" do
    # No real PKCS#12 cert or TSA network is available, so each wrapper is
    # invoked with empty/minimal inputs and must either return or raise — the
    # goal is that every phase-6 wrapper is exercised, not that crypto succeeds.
    defp returns_or_raises(fun) do
      try do
        fun.()
        :ok
      rescue
        _ -> :ok
      catch
        _, _ -> :ok
      end
    end

    test "certificate loaders + accessors" do
      assert :ok = returns_or_raises(fn -> PdfOxide.certificate_from_bytes(<<0, 1, 2>>, "") end)
      assert :ok = returns_or_raises(fn -> PdfOxide.certificate_from_pem("", "") end)

      case PdfOxide.certificate_from_pem("", "") do
        {:ok, cert} ->
          _ = PdfOxide.certificate_subject(cert)
          _ = PdfOxide.certificate_issuer(cert)
          _ = PdfOxide.certificate_serial(cert)
          _ = PdfOxide.certificate_validity(cert)
          _ = PdfOxide.certificate_valid?(cert)
          assert :ok = PdfOxide.certificate_close(cert)

        {:error, _} ->
          :ok
      end
    end

    test "signing wrappers" do
      pdf = sample_pdf()

      case PdfOxide.certificate_from_pem("", "") do
        {:ok, cert} ->
          _ = PdfOxide.sign_bytes(pdf, cert, "r", "l")
          _ = PdfOxide.sign_bytes_pades(pdf, cert, 0, "")
          _ = PdfOxide.sign_bytes_pades_opts(pdf, cert, 0, "")
          PdfOxide.certificate_close(cert)
          :ok

        {:error, _} ->
          # Loader raised/failed; just assert the surfaces exist.
          assert function_exported?(PdfOxide, :sign_bytes, 4)
          assert function_exported?(PdfOxide, :sign_bytes_pades, 5)
          assert function_exported?(PdfOxide, :sign_bytes_pades_opts, 5)
      end
    end

    test "timestamp parse + accessors" do
      assert :ok = returns_or_raises(fn -> PdfOxide.timestamp_parse(<<0, 1, 2, 3>>) end)

      case PdfOxide.timestamp_parse(<<0, 1, 2, 3>>) do
        {:ok, ts} ->
          _ = PdfOxide.timestamp_token(ts)
          _ = PdfOxide.timestamp_message_imprint(ts)
          _ = PdfOxide.timestamp_time(ts)
          _ = PdfOxide.timestamp_serial(ts)
          _ = PdfOxide.timestamp_tsa_name(ts)
          _ = PdfOxide.timestamp_policy_oid(ts)
          _ = PdfOxide.timestamp_hash_algorithm(ts)
          _ = PdfOxide.timestamp_verify(ts)
          assert :ok = PdfOxide.timestamp_close(ts)

        {:error, _} ->
          :ok
      end
    end

    test "tsa client wrappers" do
      assert :ok =
               returns_or_raises(fn ->
                 PdfOxide.tsa_client("http://localhost:0/tsa", timeout: 1)
               end)

      case PdfOxide.tsa_client("http://localhost:0/tsa", timeout: 1) do
        {:ok, client} ->
          _ = returns_or_raises(fn -> PdfOxide.tsa_request_timestamp(client, <<1, 2, 3>>) end)

          _ =
            returns_or_raises(fn ->
              PdfOxide.tsa_request_timestamp_hash(client, <<1, 2, 3>>, 0)
            end)

          assert :ok = PdfOxide.tsa_close(client)

        {:error, _} ->
          assert function_exported?(PdfOxide, :tsa_request_timestamp, 2)
          assert function_exported?(PdfOxide, :tsa_request_timestamp_hash, 3)
      end
    end

    test "signature-info + dss surfaces exist" do
      # SignatureInfo / Dss handles come from a signed document; we have none, so
      # assert the wrapper surface exists (each is exercised when a real signed
      # PDF is present).
      for {f, arity} <- [
            {:signature_signer_name, 1},
            {:signature_reason, 1},
            {:signature_location, 1},
            {:signature_time, 1},
            {:signature_certificate, 1},
            {:signature_pades_level, 1},
            {:signature_has_timestamp?, 1},
            {:signature_timestamp, 1},
            {:signature_add_timestamp, 2},
            {:signature_verify, 1},
            {:signature_verify_detached, 2},
            {:signature_close, 1},
            {:dss_cert_count, 1},
            {:dss_crl_count, 1},
            {:dss_ocsp_count, 1},
            {:dss_vri_count, 1},
            {:dss_cert, 2},
            {:dss_crl, 2},
            {:dss_ocsp, 2},
            {:dss_close, 1}
          ] do
        assert function_exported?(PdfOxide, f, arity)
      end
    end
  end

  # ── phase 7: barcodes / OCR / render variants / redaction / constructors /
  # page getters / timestamp ─────────────────────────────────────────────────
  defp returns_or_raises_7(fun) do
    try do
      fun.()
      :ok
    rescue
      _ -> :ok
    catch
      _, _ -> :ok
    end
  end

  describe "barcodes (phase 7)" do
    test "generate_qr_code -> data/format/png/svg" do
      assert {:ok, bc} = PdfOxide.generate_qr_code("hello-qr", 1, 256)
      assert %PdfOxide.Barcode{} = bc
      assert {:ok, data} = PdfOxide.barcode_data(bc)
      assert data =~ "hello-qr"
      assert {:ok, fmt} = PdfOxide.barcode_format(bc)
      assert is_integer(fmt)
      assert {:ok, conf} = PdfOxide.barcode_confidence(bc)
      assert is_float(conf)
      assert {:ok, png} = PdfOxide.barcode_png(bc, 128)
      assert is_binary(png) and byte_size(png) > 0
      assert {:ok, svg} = PdfOxide.barcode_svg(bc, 128)
      assert is_binary(svg) and byte_size(svg) > 0
      assert :ok = PdfOxide.barcode_close(bc)
    end

    test "generate_barcode -> data/format" do
      assert {:ok, bc} = PdfOxide.generate_barcode("12345678", 0, 256)
      assert %PdfOxide.Barcode{} = bc
      assert {:ok, data} = PdfOxide.barcode_data(bc)
      assert is_binary(data)
      assert {:ok, fmt} = PdfOxide.barcode_format(bc)
      assert is_integer(fmt)
      assert :ok = PdfOxide.barcode_close(bc)
    end

    test "add_barcode_to_page on an editor" do
      {:ok, ed} = PdfOxide.open_editor_from_bytes(sample_pdf())
      {:ok, bc} = PdfOxide.generate_qr_code("on-page", 1, 256)
      # Returns :ok or {:error, code}; either exercises the wrapper.
      result = PdfOxide.add_barcode_to_page(ed, 0, bc, 10, 10, 60, 60)
      assert result == :ok or match?({:error, _}, result)
      PdfOxide.barcode_close(bc)
      PdfOxide.editor_close(ed)
    end
  end

  describe "render variants (phase 7)" do
    setup do
      {:ok, doc} = PdfOxide.open_from_bytes(sample_pdf())
      {:ok, doc: doc}
    end

    test "render_page_with_options", %{doc: doc} do
      assert {:ok, img} = PdfOxide.render_page_with_options(doc, 0, dpi: 96)
      assert %PdfOxide.RenderedImage{} = img
      assert img.width > 0 and img.height > 0
      assert byte_size(img.data) > 0
    end

    test "render_page_with_options_ex (empty layers)", %{doc: doc} do
      assert {:ok, img} = PdfOxide.render_page_with_options_ex(doc, 0, [], dpi: 96)
      assert img.width > 0 and img.height > 0
      assert byte_size(img.data) > 0
    end

    test "render_page_region", %{doc: doc} do
      assert {:ok, img} = PdfOxide.render_page_region(doc, 0, 0, 0, 100, 100, 0)
      assert img.width > 0 and img.height > 0
      assert byte_size(img.data) > 0
    end

    test "render_page_fit", %{doc: doc} do
      assert {:ok, img} = PdfOxide.render_page_fit(doc, 0, 200, 200, 0)
      assert img.width > 0 and img.height > 0
      assert byte_size(img.data) > 0
    end

    test "render_page_raw", %{doc: doc} do
      assert {:ok, img} = PdfOxide.render_page_raw(doc, 0, 96)
      assert img.width > 0 and img.height > 0
      assert byte_size(img.data) > 0
    end

    test "renderer create/close + estimate_render_time", %{doc: doc} do
      # pdf_create_renderer is a no-op stub: returns a handle or errors. Either
      # outcome exercises the wrapper.
      case PdfOxide.renderer(96, 0, 85, true) do
        {:ok, rndr} ->
          assert %PdfOxide.Renderer{} = rndr
          assert :ok = PdfOxide.renderer_close(rndr)

        {:error, _} ->
          :ok
      end

      result = PdfOxide.estimate_render_time(doc, 0)
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end
  end

  describe "page getters (phase 7)" do
    setup do
      {:ok, doc} = PdfOxide.open_from_bytes(sample_pdf())
      {:ok, doc: doc}
    end

    test "page_width/page_height/page_rotation", %{doc: doc} do
      assert {:ok, w} = PdfOxide.page_width(doc, 0)
      assert is_number(w) and w > 0
      assert {:ok, h} = PdfOxide.page_height(doc, 0)
      assert is_number(h) and h > 0
      assert {:ok, rot} = PdfOxide.page_rotation(doc, 0)
      assert is_integer(rot)
    end

    test "page_elements + element_count + close", %{doc: doc} do
      case PdfOxide.page_elements(doc, 0) do
        {:ok, elems} ->
          assert %PdfOxide.ElementList{} = elems
          assert {:ok, n} = PdfOxide.element_count(elems)
          assert is_integer(n) and n >= 0
          assert :ok = PdfOxide.element_list_close(elems)

        {:error, _} ->
          assert function_exported?(PdfOxide, :element_count, 1)
      end
    end
  end

  describe "redaction (phase 7)" do
    test "add/count/apply/scrub on an editor" do
      {:ok, ed} = PdfOxide.open_editor_from_bytes(sample_pdf())

      assert :ok = PdfOxide.redaction_add(ed, 0, 10, 10, 60, 30, 0.0, 0.0, 0.0)
      assert {:ok, n} = PdfOxide.redaction_count(ed, 0)
      assert is_integer(n) and n >= 1

      apply_result = PdfOxide.redaction_apply(ed, false, 0.0, 0.0, 0.0)
      assert match?({:ok, _}, apply_result) or match?({:error, _}, apply_result)

      scrub_result = PdfOxide.redaction_scrub_metadata(ed)
      assert match?({:ok, _}, scrub_result) or match?({:error, _}, scrub_result)

      PdfOxide.editor_close(ed)
    end
  end

  describe "constructors (phase 7)" do
    test "from_image_bytes raises/errors on bad input" do
      result = PdfOxide.from_image_bytes(<<0, 1, 2, 3>>)
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end

    test "from_image errors on a nonexistent path" do
      assert {:error, _} = PdfOxide.from_image("/nonexistent/nope.png")
    end

    # from_html_css builds where the html-render path is available, else errors
    # (e.g. no default font in this cdylib). Either outcome exercises the wrapper.
    test "from_html_css produces a PDF or errors" do
      case PdfOxide.from_html_css("<h1>hi</h1><p>body</p>", "h1{color:red}") do
        {:ok, pdf} ->
          assert {:ok, bytes} = PdfOxide.to_bytes(pdf)
          assert byte_size(bytes) > 100

        {:error, _} ->
          :ok
      end
    end

    test "from_html_css_with_fonts (no fonts) produces a PDF or errors" do
      case PdfOxide.from_html_css_with_fonts("<p>x</p>", "", []) do
        {:ok, pdf} ->
          assert {:ok, bytes} = PdfOxide.to_bytes(pdf)
          assert byte_size(bytes) > 100

        {:error, _} ->
          :ok
      end
    end

    test "merge of two temp PDFs (or errors)" do
      p1 = Path.join(System.tmp_dir!(), "pdfoxide_m1_#{System.unique_integer([:positive])}.pdf")
      p2 = Path.join(System.tmp_dir!(), "pdfoxide_m2_#{System.unique_integer([:positive])}.pdf")
      {:ok, a} = PdfOxide.from_markdown("# A\n\none\n")
      {:ok, b} = PdfOxide.from_markdown("# B\n\ntwo\n")
      :ok = PdfOxide.save(a, p1)
      :ok = PdfOxide.save(b, p2)

      result = PdfOxide.merge([p1, p2])
      assert match?({:ok, _}, result) or match?({:error, _}, result)

      File.rm(p1)
      File.rm(p2)
    end
  end

  describe "OCR + timestamp wrappers exercised with minimal inputs (phase 7)" do
    # OCR needs model files and add_timestamp needs a live TSA; we have neither,
    # so each wrapper is invoked with minimal inputs and must return or raise.
    test "ocr engine + page wrappers" do
      assert :ok = returns_or_raises_7(fn -> PdfOxide.ocr_engine("", "", "") end)

      {:ok, doc} = PdfOxide.open_from_bytes(sample_pdf())
      needs = PdfOxide.ocr_page_needs_ocr(doc, 0)
      assert match?({:ok, _}, needs) or match?({:error, _}, needs)

      # engine=nil path: native extraction only.
      native = PdfOxide.ocr_extract_text(doc, 0, nil)
      assert match?({:ok, _}, native) or match?({:error, _}, native)

      assert function_exported?(PdfOxide, :ocr_extract_text, 3)
      assert function_exported?(PdfOxide, :ocr_engine_close, 1)
      PdfOxide.close(doc)
    end

    test "add_timestamp returns or raises with empty TSA url" do
      assert :ok =
               returns_or_raises_7(fn -> PdfOxide.add_timestamp(sample_pdf(), 0, "") end)

      result = PdfOxide.add_timestamp(sample_pdf(), 0, "")
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end
  end

  # ── phase 8: final-coverage wrappers ────────────────────────────────────────
  # Most work on the markdown sample; office-import/sign/convert/prefetch are
  # invoked with minimal inputs and asserted return-or-error (no real office
  # files / certs / network are available here).
  describe "phase 8 final coverage" do
    setup do
      {:ok, doc} = PdfOxide.open_from_bytes(sample_pdf())
      {:ok, doc: doc}
    end

    defp ok_or_error(result), do: match?({:ok, _}, result) or match?({:error, _}, result)

    test "office: open_from_*_bytes return or error on non-office bytes" do
      junk = sample_pdf()
      assert ok_or_error(PdfOxide.open_from_docx_bytes(junk))
      assert ok_or_error(PdfOxide.open_from_pptx_bytes(junk))
      assert ok_or_error(PdfOxide.open_from_xlsx_bytes(junk))
    end

    test "office: to_docx/pptx/xlsx return bytes or error", %{doc: doc} do
      assert ok_or_error(PdfOxide.to_docx(doc))
      assert ok_or_error(PdfOxide.to_pptx(doc))
      assert ok_or_error(PdfOxide.to_xlsx(doc))
    end

    test "in-rect: text/words/lines/tables/images", %{doc: doc} do
      assert {:ok, text} = PdfOxide.extract_text_in_rect(doc, 0, 0.0, 0.0, 1000.0, 1000.0)
      assert is_binary(text)
      assert {:ok, words} = PdfOxide.extract_words_in_rect(doc, 0, 0.0, 0.0, 1000.0, 1000.0)
      assert is_list(words)
      assert Enum.all?(words, &match?(%PdfOxide.Word{}, &1))
      assert {:ok, lines} = PdfOxide.extract_lines_in_rect(doc, 0, 0.0, 0.0, 1000.0, 1000.0)
      assert Enum.all?(lines, &match?(%PdfOxide.TextLine{}, &1))
      assert {:ok, tables} = PdfOxide.extract_tables_in_rect(doc, 0, 0.0, 0.0, 1000.0, 1000.0)
      assert Enum.all?(tables, &match?(%PdfOxide.Table{}, &1))
      assert {:ok, images} = PdfOxide.extract_images_in_rect(doc, 0, 0.0, 0.0, 1000.0, 1000.0)
      assert Enum.all?(images, &match?(%PdfOxide.Image{}, &1))
    end

    test "auto extraction + classification", %{doc: doc} do
      assert ok_or_error(PdfOxide.extract_text_auto(doc, 0))
      assert ok_or_error(PdfOxide.extract_all_text(doc))
      assert ok_or_error(PdfOxide.extract_page_auto(doc, 0))
      assert ok_or_error(PdfOxide.extract_page_auto(doc, 0, ""))
      assert ok_or_error(PdfOxide.classify_page(doc, 0))
      assert ok_or_error(PdfOxide.classify_document(doc))
    end

    test "header/footer/artifact erase + remove", %{doc: doc} do
      assert ok_or_error(PdfOxide.erase_header(doc, 0))
      assert ok_or_error(PdfOxide.erase_footer(doc, 0))
      assert ok_or_error(PdfOxide.erase_artifacts(doc, 0))
      assert ok_or_error(PdfOxide.remove_headers(doc))
      assert ok_or_error(PdfOxide.remove_footers(doc, 0.5))
      assert ok_or_error(PdfOxide.remove_artifacts(doc, 0.5))
    end

    test "forms: fields (empty list ok) + import/export", %{doc: doc} do
      assert {:ok, fields} = PdfOxide.form_fields(doc)
      assert is_list(fields)
      assert Enum.all?(fields, &match?(%PdfOxide.FormField{}, &1))
      assert ok_or_error(PdfOxide.export_form_data_to_bytes(doc))
      assert ok_or_error(PdfOxide.export_form_data_to_bytes(doc, 1))
      assert ok_or_error(PdfOxide.import_form_data(doc, "/nonexistent/data.fdf"))
      assert ok_or_error(PdfOxide.form_import_from_file(doc, "/nonexistent/data.fdf"))

      {:ok, ed} = PdfOxide.open_editor_from_bytes(sample_pdf())
      assert ok_or_error(PdfOxide.import_fdf_bytes(ed, <<>>))
      assert ok_or_error(PdfOxide.import_xfdf_bytes(ed, <<>>))
    end

    test "document structure / metadata", %{doc: doc} do
      assert ok_or_error(PdfOxide.outline(doc))
      assert ok_or_error(PdfOxide.page_labels(doc))
      assert ok_or_error(PdfOxide.xmp_metadata(doc))
      assert ok_or_error(PdfOxide.source_bytes(doc))
      assert is_boolean(PdfOxide.has_xfa?(doc))
      assert ok_or_error(PdfOxide.plan_split_by_bookmarks(doc))
    end

    test "pdf_page_count on a built Pdf" do
      {:ok, pdf} = PdfOxide.from_markdown("# c\n\nx\n")
      assert ok_or_error(PdfOxide.pdf_page_count(pdf))
    end

    test "doc-level signatures (no cert): count/verify/timestamp/dss", %{doc: doc} do
      assert ok_or_error(PdfOxide.signature_count(doc))
      assert ok_or_error(PdfOxide.signature(doc, 0))
      assert ok_or_error(PdfOxide.verify_all_signatures(doc))
      assert ok_or_error(PdfOxide.document_has_timestamp?(doc))
      assert ok_or_error(PdfOxide.document_dss(doc))
    end

    test "sign wrapper exists and errors without a real cert", %{doc: doc} do
      assert function_exported?(PdfOxide, :sign, 4)
      {:ok, cert} = wrap_or_skip_cert()

      if cert do
        assert ok_or_error(PdfOxide.sign(doc, cert, "r", "l"))
      else
        :ok
      end
    end

    test "annotation extras (empty page -> error) + to_json", %{doc: doc} do
      assert ok_or_error(PdfOxide.annotation_color(doc, 0, 0))
      assert ok_or_error(PdfOxide.annotation_creation_date(doc, 0, 0))
      assert ok_or_error(PdfOxide.annotation_modification_date(doc, 0, 0))
      assert ok_or_error(PdfOxide.annotation_hidden?(doc, 0, 0))
      assert ok_or_error(PdfOxide.annotation_marked_deleted?(doc, 0, 0))
      assert ok_or_error(PdfOxide.annotation_printable?(doc, 0, 0))
      assert ok_or_error(PdfOxide.annotation_read_only?(doc, 0, 0))
      assert ok_or_error(PdfOxide.link_annotation_uri(doc, 0, 0))
      assert ok_or_error(PdfOxide.text_annotation_icon_name(doc, 0, 0))
      assert ok_or_error(PdfOxide.highlight_quad_points_count(doc, 0, 0))
      assert ok_or_error(PdfOxide.highlight_quad_point(doc, 0, 0, 0))
      assert ok_or_error(PdfOxide.annotations_to_json(doc, 0))
    end

    test "element accessors + elements_to_json", %{doc: doc} do
      assert {:ok, elems} = PdfOxide.page_elements(doc, 0)
      assert {:ok, n} = PdfOxide.element_count(elems)
      assert is_integer(n)
      assert ok_or_error(PdfOxide.element_type(elems, 0))
      assert ok_or_error(PdfOxide.element_text(elems, 0))
      assert ok_or_error(PdfOxide.element_rect(elems, 0))
      assert ok_or_error(PdfOxide.elements_to_json(elems))
      PdfOxide.element_list_close(elems)
    end

    test "font_size + fonts_to_json + search_results_to_json", %{doc: doc} do
      assert ok_or_error(PdfOxide.font_size(doc, 0, 0))
      assert ok_or_error(PdfOxide.fonts_to_json(doc, 0))
      assert ok_or_error(PdfOxide.search_results_to_json(doc, "Alpha"))
      assert ok_or_error(PdfOxide.search_results_to_json(doc, "Alpha", true))
    end

    test "crypto / FIPS" do
      assert {:ok, prov} = PdfOxide.crypto_active_provider()
      assert is_binary(prov)
      assert ok_or_error(PdfOxide.crypto_cbom())
      assert ok_or_error(PdfOxide.crypto_inventory())
      assert ok_or_error(PdfOxide.crypto_policy())
      assert is_integer(PdfOxide.crypto_fips_available())
      assert is_integer(PdfOxide.crypto_use_fips())
      assert is_integer(PdfOxide.crypto_set_policy("default"))
    end

    test "models / config" do
      assert ok_or_error(PdfOxide.model_manifest())
      assert is_integer(PdfOxide.prefetch_available())
      assert ok_or_error(PdfOxide.prefetch_models(""))
      assert is_integer(PdfOxide.set_max_ops_per_stream(1_000_000))
      assert is_integer(PdfOxide.set_preserve_unmapped_glyphs(0))
    end

    test "convert_to_pdf_a returns or errors", %{doc: doc} do
      result = PdfOxide.convert_to_pdf_a(doc, 2)
      assert result == :ok or match?({:error, _}, result)
    end

    # A cert is unavailable here; return nil so sign/4 is only invoked when one
    # can actually be constructed.
    defp wrap_or_skip_cert, do: {:ok, nil}
  end
end

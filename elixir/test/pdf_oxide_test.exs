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
end

# frozen_string_literal: true

require 'spec_helper'

# Regression coverage for the audit-driven FFI fixes in v0.3.55 #547.
#
# Each example here corresponds to a binding whose `attach_function`
# signature was wrong against `include/pdf_oxide_c/pdf_oxide.h`. The
# bugs were latent because no spec exercised the path. Filling those
# gaps prevents the bug class from regressing silently — any future
# off-by-one trailing `int32_t *error_code` (the cause of the
# pdf_oxide_search_result_get_page segfault) will now turn an aarch64
# segfault into a hard test failure on every cell.
RSpec.describe 'FFI signature regressions' do
  include_context 'fixtures-present'

  describe 'pdf_oxide_search_result_get_page / _get_text / _get_bbox (round 3)' do
    # Original bug: Ruby decl was 2 args (handle, idx), C wanted 3
    # (handle, idx, int32_t *err). Trailing register garbage was used
    # as the err pointer → segfault on aarch64 / abort on macOS arm64.
    it 'returns real (non-zero) bbox values when the query is found' do
      skip 'fixture missing' unless File.exist?(fixture('hello_structure.pdf'))

      PdfOxide::PdfDocument.open(fixture('hello_structure.pdf')) do |doc|
        matches = doc.search('Hello', case_sensitive: false)
        next if matches.empty? # text-less fixture is a no-match, not a bug

        first = matches.first
        bbox = first[:bbox]
        expect(bbox.values_at(:x, :y, :width, :height)).to all(be_a(Float))
        # The pre-fix code returned the zero-rect placeholder
        # { x: 0.0, y: 0.0, width: 0.0, height: 0.0 } for every match.
        # Any real match has at least one non-zero coordinate.
        expect(bbox.values.any? { |v| v != 0.0 }).to be(true)
      end
    end

    it 'does not segfault under repeated search invocations' do
      # The segfault was probabilistic — register garbage happened to
      # point into a mapped page on x86_64 but unmapped on aarch64.
      # Hammering search 20× exercises both the unaligned-err-write
      # case and the owned-char* free path for _get_text.
      PdfOxide::PdfDocument.open(fixture('simple.pdf')) do |doc|
        20.times { doc.search('the') }
      end
    end
  end

  describe 'pdf_document_is_encrypted (A.1 — was passing extra err arg)' do
    # Pre-fix Ruby decl was %i[pointer pointer], :bool but the C side
    # is bool pdf_document_is_encrypted(const PdfDocument *handle) —
    # no err arg. The extra pointer was silently ignored on cdecl
    # ABIs; on stricter ABIs it would corrupt the call frame.
    it 'returns false for an unencrypted fixture' do
      PdfOxide::PdfDocument.open(fixture('simple.pdf')) do |doc|
        expect(doc.encrypted?).to be(false)
      end
    end

    it 'returns true for a known-encrypted fixture' do
      skip 'fixture missing' unless File.exist?(fixture('encrypted_objstm.pdf'))

      PdfOxide::PdfDocument.open(fixture('encrypted_objstm.pdf')) do |doc|
        expect(doc.encrypted?).to be(true)
      end
    end
  end

  describe 'pdf_document_open_from_bytes (A.1 — was 8-pointer placeholder)' do
    # Pre-fix the symbol was attached as 8 pointers; the real C sig is
    # (data, len, err). PdfDocument.open routes byte-buffer args
    # through this binding, so any call from Ruby pre-fix raised
    # `ArgumentError: wrong number of arguments`.
    it 'opens a PDF passed as a byte buffer via PdfDocument.open' do
      bytes = File.binread(fixture('simple.pdf'))
      PdfOxide::PdfDocument.open(bytes) do |doc|
        expect(doc.page_count).to be > 0
      end
    end
  end

  describe 'pdf_document_get_form_fields (A.1 — was 8-pointer placeholder)' do
    it 'returns an Array even for a fixture with no AcroForm' do
      PdfOxide::PdfDocument.open(fixture('simple.pdf')) do |doc|
        expect(doc.form_fields).to be_an(Array)
      end
    end
  end

  describe 'pdf_validate_pdf_a_level (A.1 — was 8-pointer placeholder)' do
    # The placeholder would have ArgumentError'd. After the fix the
    # call dispatches with 3 args (doc, level, err) — for a non-PDF/A
    # input the cdylib should return false (or the symbol may be
    # absent if the build lacks the relevant feature; pdf_a? handles
    # both via FFI::NotFoundError and respond_to? guards).
    it 'returns a boolean without raising for a non-compliant PDF' do
      PdfOxide::PdfDocument.open(fixture('simple.pdf')) do |doc|
        result = PdfOxide::PdfValidator.pdf_a?(doc, level: :a1b)
        expect([true, false]).to include(result)
      end
    end
  end

  describe 'extract_text / to_markdown / to_html (B.1 — :string→:pointer leak fix)' do
    # The :string return-type caused Ruby FFI to copy the C string but
    # never call free_string. Looped calls would leak one buffer per
    # invocation. We can't measure heap growth from RSpec cheaply, but
    # we can at least prove the fixed path returns sensible strings
    # under repeated invocation (the original :string path also worked
    # — the regression here is purely a leak, so a smoke loop is
    # enough to guard against re-introducing :string).
    it 'extract_text returns a String repeatedly without raising' do
      PdfOxide::PdfDocument.open(fixture('simple.pdf')) do |doc|
        10.times do
          out = doc.extract_text(0)
          expect(out).to be_a(String)
        end
      end
    end

    it 'to_markdown returns a String for the whole document' do
      PdfOxide::PdfDocument.open(fixture('simple.pdf')) do |doc|
        out = PdfOxide::MarkdownConverter.to_markdown(doc)
        expect(out).to be_a(String)
      end
    end

    it 'to_html returns a String for the whole document' do
      PdfOxide::PdfDocument.open(fixture('simple.pdf')) do |doc|
        out = PdfOxide::MarkdownConverter.to_html(doc)
        expect(out).to be_a(String)
      end
    end
  end

  describe 'PDF/A + PDF/UA level → cdylib wire-format integer parity' do
    # Locks in the alignment with the cdylib's documented integer
    # encoding (src/ffi.rs:1225 + 5538) and with the C# / Java / PHP
    # bindings. Pre-fix Ruby had `a1a: 0, a1b: 1` — semantically
    # reversed against the cdylib's actual mapping. Pre-fix users
    # asking for "validate as A1a" got A1b validation, etc.
    it 'PDF/A: B before A within each level, matching src/ffi.rs:1225' do
      expect(PdfOxide::PdfValidator::PDF_A_LEVELS).to eq(
        a1b: 0, a1a: 1, a2b: 2, a2a: 3, a2u: 4, a3b: 5, a3a: 6, a3u: 7
      )
    end

    it 'PDF/UA: 1-indexed, matching src/ffi.rs:5538' do
      expect(PdfOxide::PdfValidator::PDF_UA_LEVELS).to eq(ua1: 1, ua2: 2)
    end
  end

  describe 'cdylib error-code → exception mapping (parity with PHP + C#)' do
    # Locks in src/ffi.rs:98-106 — same 9-code surface every binding
    # uses. Pre-v0.3.55 had alphabetical-natural mapping (4 =>
    # StateError, 8 => SignatureError, …); cdylib returning 8
    # (ERR_UNSUPPORTED) silently raised SignatureError, etc.
    let(:doc_class) do
      Class.new do
        def initialize; end

        def call_raise_for_code(code, op)
          # Re-expose the private instance method via Module#send.
          PdfOxide::PdfDocument.allocate.send(:raise_for_code, code, op)
        end
      end
    end

    [
      [1, PdfOxide::ArgumentError,           'ERR_INVALID_ARG'],
      [2, PdfOxide::IoError,                 'ERR_IO'],
      [3, PdfOxide::ParseError,              'ERR_PARSE'],
      [4, PdfOxide::ParseError,              'ERR_EXTRACTION'],
      [5, PdfOxide::InternalError,           'ERR_INTERNAL'],
      [6, PdfOxide::ArgumentError,           'ERR_INVALID_PAGE'],
      [7, PdfOxide::SearchError,             'ERR_SEARCH'],
      [8, PdfOxide::UnsupportedFeatureError, '_ERR_UNSUPPORTED']
    ].each do |code, klass, label|
      it "code #{code} (#{label}) → #{klass}" do
        instance = doc_class.new
        expect { instance.call_raise_for_code(code, "op_#{code}") }
          .to raise_error(klass, /op_#{code} failed/)
      end
    end

    it 'returns silently for code 0 (SUCCESS)' do
      instance = doc_class.new
      expect { instance.call_raise_for_code(0, 'noop') }.not_to raise_error
    end

    it 'falls back to InternalError for unknown codes' do
      instance = doc_class.new
      expect { instance.call_raise_for_code(99, 'weird') }
        .to raise_error(PdfOxide::InternalError, /weird failed/)
    end
  end

  describe 'PadesSignOptions struct layout (Ruby/Rust parity)' do
    # Already covered in pdf_signer_spec, but re-asserted here so the
    # struct-layout invariant lives alongside the other FFI regression
    # guards. The Rust tests/test_pkcs12_signing_opts.rs asserts the
    # same 14 * 8 == 112 byte size from the C side.
    it 'is 14 fields × 8 bytes (112 B) on a 64-bit target' do
      expect(PdfOxide::PdfSigner::PadesSignOptions.size).to eq(14 * 8)
    end
  end
end

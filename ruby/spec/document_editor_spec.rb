# frozen_string_literal: true

require 'spec_helper'
require 'tmpdir'

RSpec.describe PdfOxide::DocumentEditor do
  include_context 'fixtures-present'

  let(:source) { fixture('simple.pdf') }

  describe '#add_redaction + #apply_redactions! + #to_bytes' do
    it 'happy-path: queue, apply, save bytes' do
      PdfOxide::DocumentEditor.open(source) do |ed|
        ed.add_redaction(page: 0, rect: [100.0, 200.0, 300.0, 250.0])
        expect(ed.redaction_count(0)).to eq(1)
        ed.apply_redactions!
        bytes = ed.to_bytes
        expect(bytes[0, 5]).to eq('%PDF-')
        expect(bytes.bytesize).to be > 50
      end
    end

    it 'rejects malformed rects (4 numeric values required)' do
      PdfOxide::DocumentEditor.open(source) do |ed|
        expect { ed.add_redaction(page: 0, rect: [1.0, 2.0]) }
          .to raise_error(PdfOxide::ArgumentError, /4 numeric/)
      end
    end
  end

  describe '#save_to' do
    it 'writes a destructively redacted file to disk' do
      Dir.mktmpdir do |tmp|
        out = File.join(tmp, 'redacted.pdf')
        PdfOxide::DocumentEditor.open(source) do |ed|
          ed.add_redaction(page: 0, rect: [100.0, 200.0, 300.0, 250.0])
          ed.apply_redactions!
          ed.save_to(out)
        end
        expect(File).to exist(out)
        File.open(out, 'rb') { |f| expect(f.read(5)).to eq('%PDF-') }
      end
    end
  end

  describe '#close' do
    it 'is idempotent' do
      ed = PdfOxide::DocumentEditor.open(source)
      ed.close
      expect(ed.closed?).to be(true)
      ed.close
      ed.close
      expect(ed.closed?).to be(true)
    end

    it 'raises on use-after-close' do
      ed = PdfOxide::DocumentEditor.open(source)
      ed.close
      expect { ed.redaction_count(0) }.to raise_error(PdfOxide::InvalidStateError)
    end
  end
end

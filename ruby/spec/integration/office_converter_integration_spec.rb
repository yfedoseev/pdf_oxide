# frozen_string_literal: true

# Phase 3 acceptance test for OfficeConverter (v0.3.48 #159).
#
# The cdylib supports DOCX / PPTX / XLSX → PDF in-memory.  The
# v0.3.55 Ruby fixture set doesn't ship sample Office files (the
# full happy-path lands in Phase 4 with a curated fixture pack);
# this spec asserts the surface is plumbed and the security
# contract holds — wrong bytes fail-closed with a typed error.

require 'spec_helper'

RSpec.describe PdfOxide::OfficeConverter, :skip_mock do
  it 'rejects empty / nil byte inputs' do
    expect { PdfOxide::OfficeConverter.from_docx(nil) }
      .to raise_error(PdfOxide::ArgumentError, /nil\/empty/)
    expect { PdfOxide::OfficeConverter.from_pptx('') }
      .to raise_error(PdfOxide::ArgumentError, /nil\/empty/)
    expect { PdfOxide::OfficeConverter.from_xlsx('') }
      .to raise_error(PdfOxide::ArgumentError, /nil\/empty/)
  end

  it 'fails closed on malformed bytes (cdylib reports an error code)' do
    # The C ABI rejects non-zip / unrecognised payloads with a non-
    # zero error code.  Confirm the binding surfaces it as a typed
    # PdfOxide error rather than crashing.
    expect {
      PdfOxide::OfficeConverter.from_docx('this is not a docx archive')
    }.to raise_error(PdfOxide::Error)
  end

  it 'infers format from a non-existent file path (file-not-found guard)' do
    expect {
      PdfOxide::OfficeConverter.from_file('/no/such/file.docx')
    }.to raise_error(PdfOxide::FileNotFoundError)
  end
end

# frozen_string_literal: true

require 'spec_helper'

RSpec.describe PdfOxide::Managers::Barcode do
  let(:mock_handle) { double(null?: false) }
  let(:mock_document) { instance_double(PdfOxide::Document, handle: mock_handle, closed?: false, page_count: 5) }
  let(:manager) { described_class.new(mock_document) }

  describe 'barcode format constants' do
    it 'defines all barcode formats' do
      expect(described_class::BARCODE_FORMAT_CODE128).to eq(0)
      expect(described_class::BARCODE_FORMAT_EAN13).to eq(3)
      expect(described_class::BARCODE_FORMAT_UPCA).to eq(5)
    end
  end

  describe '#add_qr_code' do
    it 'adds QR code to page' do
      expect(FFI::Bindings).to receive(:pdf_document_add_qr_code).and_return(true)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('https://example.com')

      result = manager.add_qr_code(0, 50, 50, 100, 'https://example.com')
      expect(result).to be true
    end

    it 'validates page index' do
      expect { manager.add_qr_code(-1, 50, 50, 100, 'data') }.to raise_error(PdfOxide::ArgumentError)
    end

    it 'converts coordinates to floats' do
      allow(FFI::Bindings).to receive(:pdf_document_add_qr_code)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('data')

      manager.add_qr_code(0, 50, 50, 100, 'data')
      expect(FFI::Bindings).to have_received(:pdf_document_add_qr_code)
        .with(mock_handle, 0, 50.0, 50.0, 100.0, anything, anything, anything)
    end
  end

  describe '#add_barcode' do
    it 'adds barcode to page' do
      expect(FFI::Bindings).to receive(:pdf_document_add_barcode).and_return(true)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('12345678')

      result = manager.add_barcode(0, 50, 50, 100, 50, '12345678', :code128)
      expect(result).to be true
    end

    it 'converts format symbol to integer' do
      allow(FFI::Bindings).to receive(:pdf_document_add_barcode)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('data')

      manager.add_barcode(0, 50, 50, 100, 50, 'data', :ean13)
      expect(FFI::Bindings).to have_received(:pdf_document_add_barcode)
        .with(mock_handle, 0, anything, anything, anything, anything, anything, 3, anything)
    end

    it 'supports integer format parameter' do
      allow(FFI::Bindings).to receive(:pdf_document_add_barcode)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('data')

      manager.add_barcode(0, 50, 50, 100, 50, 'data', 3)
      expect(FFI::Bindings).to have_received(:pdf_document_add_barcode)
        .with(mock_handle, 0, anything, anything, anything, anything, anything, 3, anything)
    end
  end

  describe '#generate_qr_code' do
    it 'generates QR code as PNG image' do
      png_data = 'PNG binary data'
      allow(FFI::Bindings).to receive(:pdf_generate_qr_code).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_barcode_get_image_png).and_return(png_data)
      allow(FFI::Bindings).to receive(:pdf_barcode_free)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('data')

      result = manager.generate_qr_code('https://example.com', 200)
      expect(result).to eq(png_data)
    end
  end

  describe '#generate_barcode' do
    it 'generates barcode as PNG image' do
      png_data = 'PNG binary data'
      allow(FFI::Bindings).to receive(:pdf_generate_barcode).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_barcode_get_image_png).and_return(png_data)
      allow(FFI::Bindings).to receive(:pdf_barcode_free)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('data')

      result = manager.generate_barcode('12345678', :code128, 200, 100)
      expect(result).to eq(png_data)
    end
  end

  describe '#extract_barcodes' do
    it 'extracts barcodes from page' do
      allow(FFI::Bindings).to receive(:pdf_document_extract_barcodes).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_oxide_barcode_count).and_return(0)
      allow(FFI::Bindings).to receive(:pdf_oxide_barcode_list_free)

      result = manager.extract_barcodes(0)
      expect(result).to be_an(Array)
      expect(result).to be_empty
    end

    it 'validates page index' do
      expect { manager.extract_barcodes(-1) }.to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '#get_all_barcodes' do
    it 'returns all barcodes in document' do
      allow(manager).to receive(:extract_barcodes).and_return([
        { data: '123', format: 0 }
      ])

      result = manager.get_all_barcodes
      expect(result).to be_an(Array)
      expect(result.first).to have_key(:page)
    end
  end

  describe '#barcode_statistics' do
    it 'returns barcode statistics' do
      allow(manager).to receive(:get_all_barcodes).and_return([
        { data: '123', format: 0, page: 0, format_name: 'code128' }
      ])

      result = manager.barcode_statistics
      expect(result).to be_a(Hash)
      expect(result).to have_key(:total_barcodes)
      expect(result).to have_key(:by_format)
    end
  end

  # Phase 3: Barcode Completion Tests

  describe '#generate_ean13' do
    it 'generates EAN-13 barcode' do
      allow(FFI::Bindings).to receive(:pdf_generate_ean13).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_barcode_get_image_png).and_return('ean13_png')
      allow(FFI::Bindings).to receive(:pdf_barcode_free)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('5901234123457')

      result = manager.generate_ean13('5901234123457')
      expect(result).to eq('ean13_png')
    end

    it 'validates EAN-13 data format' do
      expect { manager.generate_ean13('123') }.to raise_error(PdfOxide::ArgumentError, /13 digits/)
    end

    it 'validates EAN-13 contains only digits' do
      expect { manager.generate_ean13('590123412345A') }.to raise_error(PdfOxide::ArgumentError, /only digits/)
    end
  end

  describe '#generate_ean8' do
    it 'generates EAN-8 barcode' do
      allow(FFI::Bindings).to receive(:pdf_generate_ean8).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_barcode_get_image_png).and_return('ean8_png')
      allow(FFI::Bindings).to receive(:pdf_barcode_free)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('96385074')

      result = manager.generate_ean8('96385074')
      expect(result).to eq('ean8_png')
    end

    it 'validates EAN-8 data format' do
      expect { manager.generate_ean8('12345') }.to raise_error(PdfOxide::ArgumentError, /8 digits/)
    end
  end

  describe '#generate_upc_a' do
    it 'generates UPC-A barcode' do
      allow(FFI::Bindings).to receive(:pdf_generate_upc_a).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_barcode_get_image_png).and_return('upca_png')
      allow(FFI::Bindings).to receive(:pdf_barcode_free)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('123456789012')

      result = manager.generate_upc_a('123456789012')
      expect(result).to eq('upca_png')
    end

    it 'validates UPC-A data is 12 digits' do
      expect { manager.generate_upc_a('1234567890') }.to raise_error(PdfOxide::ArgumentError, /12 digits/)
    end
  end

  describe '#generate_code128' do
    it 'generates Code128 barcode' do
      allow(FFI::Bindings).to receive(:pdf_generate_code128).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_barcode_get_image_png).and_return('code128_png')
      allow(FFI::Bindings).to receive(:pdf_barcode_free)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('ABC123456')

      result = manager.generate_code128('ABC123456')
      expect(result).to eq('code128_png')
    end

    it 'validates Code128 data is not empty' do
      expect { manager.generate_code128('') }.to raise_error(PdfOxide::ArgumentError, /cannot be empty/)
    end
  end

  describe '#generate_code39' do
    it 'generates Code39 barcode' do
      allow(FFI::Bindings).to receive(:pdf_generate_code39).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_barcode_get_image_png).and_return('code39_png')
      allow(FFI::Bindings).to receive(:pdf_barcode_free)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('ABC-1234')

      result = manager.generate_code39('ABC-1234')
      expect(result).to eq('code39_png')
    end

    it 'validates Code39 data is not empty' do
      expect { manager.generate_code39(nil) }.to raise_error(PdfOxide::ArgumentError, /cannot be empty/)
    end
  end

  describe '#barcode_to_base64' do
    it 'converts barcode to Base64' do
      barcode_ptr = double(null?: false)
      allow(FFI::Bindings).to receive(:pdf_barcode_get_image_base64).and_return('iVBORw0K...')
      allow(FFI::StringMarshaller).to receive(:from_c).and_return('base64_string')

      result = manager.barcode_to_base64(barcode_ptr)
      expect(result).to eq('base64_string')
    end

    it 'validates barcode handle' do
      expect { manager.barcode_to_base64(nil) }.to raise_error(PdfOxide::ArgumentError, /Invalid barcode handle/)
    end
  end

  describe '#add_barcode_fit' do
    it 'adds barcode to page with auto-fit' do
      barcode_ptr = double(null?: false)
      allow(FFI::Bindings).to receive(:pdf_add_barcode_to_page_fit).and_return(true)

      result = manager.add_barcode_fit(0, barcode_ptr, 50, 50, 200, 100)
      expect(result).to be true
    end

    it 'validates page index' do
      barcode_ptr = double(null?: false)
      expect { manager.add_barcode_fit(-1, barcode_ptr, 50, 50, 200, 100) }
        .to raise_error(PdfOxide::ArgumentError)
    end

    it 'validates barcode handle' do
      expect { manager.add_barcode_fit(0, nil, 50, 50, 200, 100) }
        .to raise_error(PdfOxide::ArgumentError, /Invalid barcode handle/)
    end
  end

  describe '#add_qr_with_label' do
    it 'adds QR code with label to page' do
      allow(FFI::Bindings).to receive(:pdf_add_qr_code_with_label).and_return(true)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('encoded')

      result = manager.add_qr_with_label(0, 'https://example.com', 50, 50, 100, 'My Website')
      expect(result).to be true
    end

    it 'validates page index' do
      expect { manager.add_qr_with_label(-1, 'data', 50, 50, 100) }
        .to raise_error(PdfOxide::ArgumentError)
    end

    it 'handles optional label' do
      allow(FFI::Bindings).to receive(:pdf_add_qr_code_with_label).and_return(true)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('encoded')

      result = manager.add_qr_with_label(0, 'data', 50, 50, 100)
      expect(result).to be true
    end
  end

  describe '#detect_barcodes' do
    it 'detects barcodes on page with full information' do
      allow(FFI::Bindings).to receive(:pdf_detect_barcodes_on_page).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_oxide_barcode_count).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_oxide_barcode_get_data).and_return('barcode_data')
      allow(FFI::Bindings).to receive(:pdf_barcode_get_data).and_return('barcode_ptr')
      allow(FFI::Bindings).to receive(:pdf_barcode_get_format).and_return(3)
      allow(FFI::Bindings).to receive(:pdf_barcode_get_confidence).and_return(0.95)
      allow(FFI::Bindings).to receive(:pdf_barcode_get_bounds)
      allow(FFI::Bindings).to receive(:pdf_oxide_barcode_list_free)
      allow(FFI::StringMarshaller).to receive(:from_c).and_return('barcode_data')

      result = manager.detect_barcodes(0)
      expect(result).to be_an(Array)
      expect(result.first).to have_key(:confidence)
      expect(result.first).to have_key(:bounds)
    end

    it 'validates page index' do
      expect { manager.detect_barcodes(-1) }.to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '#detect_all_barcodes' do
    it 'detects barcodes across all pages' do
      allow(manager).to receive(:detect_barcodes).and_return([
        { data: '123', confidence: 0.95, bounds: {} }
      ])

      result = manager.detect_all_barcodes
      expect(result).to be_an(Array)
      expect(result.first).to have_key(:page)
    end
  end

  describe '#barcode_detection_stats' do
    it 'returns detection statistics with confidence data' do
      allow(manager).to receive(:detect_all_barcodes).and_return([
        { data: '123', format: 0, format_name: 'code128', confidence: 0.95, page: 0, bounds: {} }
      ])

      result = manager.barcode_detection_stats
      expect(result).to be_a(Hash)
      expect(result).to have_key(:total_detections)
      expect(result).to have_key(:confidence_stats)
    end
  end

  describe '#export_barcodes_json' do
    it 'exports barcodes as JSON' do
      allow(manager).to receive(:detect_all_barcodes).and_return([
        { data: '123', format: 0, format_name: 'code128', confidence: 0.95, page: 0, bounds: {} }
      ])

      result = manager.export_barcodes_json
      expect(result).to be_a(String)
      parsed = JSON.parse(result)
      expect(parsed).to be_an(Array)
      expect(parsed.first).to have_key('data')
    end
  end

  # Integration tests

  describe 'Integration: Complete barcode workflow' do
    it 'generates and detects barcodes' do
      allow(FFI::Bindings).to receive(:pdf_generate_ean13).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_barcode_get_image_png).and_return('png_data')
      allow(FFI::Bindings).to receive(:pdf_barcode_free)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('encoded')

      # Generate
      barcode_data = manager.generate_ean13('5901234123457')
      expect(barcode_data).to eq('png_data')
    end

    it 'generates multiple barcode formats' do
      formats = {
        ean13: '5901234123457',
        ean8: '96385074',
        code128: 'ABC123456',
        code39: 'CODE-39'
      }

      formats.each do |format, data|
        allow(FFI::Bindings).to receive(:"pdf_generate_#{format}").and_return(mock_handle)
        allow(FFI::Bindings).to receive(:pdf_barcode_get_image_png).and_return("#{format}_png")
        allow(FFI::Bindings).to receive(:pdf_barcode_free)
        allow(FFI::StringMarshaller).to receive(:to_utf8).and_return(data)

        if format == :ean13
          result = manager.generate_ean13(data)
        elsif format == :ean8
          result = manager.generate_ean8(data)
        elsif format == :code128
          result = manager.generate_code128(data)
        elsif format == :code39
          result = manager.generate_code39(data)
        end

        expect(result).to eq("#{format}_png")
      end
    end
  end
end

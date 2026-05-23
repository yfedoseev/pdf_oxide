# frozen_string_literal: true

require 'spec_helper'

RSpec.describe PdfOxide::Managers::Ocr do
  let(:mock_handle) { double(null?: false) }
  let(:mock_document) { instance_double(PdfOxide::Document, handle: mock_handle, closed?: false, page_count: 5) }
  let(:manager) { described_class.new(mock_document) }

  describe '#initialize' do
    it 'initializes with nil engine handle' do
      expect(manager.engine_handle).to be_nil
    end
  end

  describe '#available?' do
    it 'returns true when engine can be initialized' do
      allow(FFI::Bindings).to receive(:pdf_ocr_engine_create).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_ocr_engine_free)

      result = manager.available?
      expect(result).to be true
    end

    it 'returns false when engine initialization fails' do
      allow(FFI::Bindings).to receive(:pdf_ocr_engine_create).and_raise(StandardError)

      result = manager.available?
      expect(result).to be false
    end
  end

  describe '#initialize_engine' do
    it 'creates OCR engine' do
      expect(FFI::Bindings).to receive(:pdf_ocr_engine_create).and_return(mock_handle)

      result = manager.initialize_engine
      expect(result).to be true
      expect(manager.engine_handle).not_to be_nil
    end

    it 'returns true if engine already initialized' do
      manager.instance_variable_set(:@engine_handle, mock_handle)

      result = manager.initialize_engine
      expect(result).to be true
    end
  end

  describe '#engine_version' do
    it 'returns engine version string' do
      manager.instance_variable_set(:@engine_handle, mock_handle)
      allow(FFI::Bindings).to receive(:pdf_ocr_engine_get_version).and_return('Tesseract 5.0')
      allow(FFI::StringMarshaller).to receive(:from_c_string).with('Tesseract 5.0').and_return('Tesseract 5.0')

      result = manager.engine_version
      expect(result).to eq('Tesseract 5.0')
    end

    it 'returns unknown if version retrieval fails' do
      manager.instance_variable_set(:@engine_handle, mock_handle)
      allow(FFI::Bindings).to receive(:pdf_ocr_engine_get_version).and_return(nil)
      allow(FFI::StringMarshaller).to receive(:from_c_string).with(nil).and_return(nil)

      result = manager.engine_version
      expect(result).to eq('unknown')
    end
  end

  describe '#page_needs_ocr?' do
    before do
      manager.instance_variable_set(:@engine_handle, mock_handle)
    end

    it 'checks if page needs OCR' do
      allow(FFI::Bindings).to receive(:pdf_ocr_page_needs_ocr).and_return(true)

      result = manager.page_needs_ocr?(0)
      expect(result).to be true
    end

    it 'validates page index' do
      expect { manager.page_needs_ocr?(-1) }.to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '#ocr_page' do
    before do
      manager.instance_variable_set(:@engine_handle, mock_handle)
    end

    it 'extracts text from page using OCR' do
      allow(FFI::Bindings).to receive(:pdf_ocr_extract_text).and_return('Extracted text')
      allow(FFI::StringMarshaller).to receive(:from_c_string).with('Extracted text').and_return('Extracted text')

      result = manager.ocr_page(0)
      expect(result).to eq('Extracted text')
    end

    it 'returns empty string if OCR fails' do
      allow(FFI::Bindings).to receive(:pdf_ocr_extract_text).and_return(nil)
      allow(FFI::StringMarshaller).to receive(:from_c_string).with(nil).and_return(nil)

      result = manager.ocr_page(0)
      expect(result).to eq('')
    end
  end

  describe '#ocr_document' do
    before do
      manager.instance_variable_set(:@engine_handle, mock_handle)
    end

    it 'extracts text from all pages' do
      allow(manager).to receive(:ocr_page).and_return('Page text')

      result = manager.ocr_document
      expect(result).to include('Page text')
    end

    it 'joins pages with double newline' do
      allow(manager).to receive(:ocr_page).and_return('Text')

      result = manager.ocr_document
      expect(result).to include("\n\n")
    end
  end

  describe '#page_is_scanned?' do
    before do
      manager.instance_variable_set(:@engine_handle, mock_handle)
    end

    it 'returns true for scanned pages' do
      allow(manager).to receive(:page_needs_ocr?).and_return(true)

      result = manager.page_is_scanned?(0)
      expect(result).to be true
    end
  end

  describe '#release_engine' do
    it 'frees OCR engine resources' do
      manager.instance_variable_set(:@engine_handle, mock_handle)
      expect(FFI::Bindings).to receive(:pdf_ocr_engine_free)

      manager.release_engine
      expect(manager.engine_handle).to be_nil
    end

    it 'handles nil engine handle' do
      manager.instance_variable_set(:@engine_handle, nil)

      expect { manager.release_engine }.not_to raise_error
    end
  end

  describe '#ocr_statistics' do
    before do
      manager.instance_variable_set(:@engine_handle, mock_handle)
    end

    it 'returns OCR statistics' do
      allow(manager).to receive(:engine_version).and_return('Tesseract 5.0')
      allow(manager).to receive(:page_needs_ocr?).and_return(true, false)

      result = manager.ocr_statistics
      expect(result).to be_a(Hash)
      expect(result).to have_key(:engine_version)
      expect(result).to have_key(:pages_to_ocr)
    end
  end

  describe '#detect_language' do
    it 'returns language code (currently defaults to English)' do
      result = manager.detect_language(0)
      expect(result).to eq('en')
    end
  end

  # Phase 4: OCR Enhancement Tests

  describe '#initialize_engine_with_config' do
    it 'initializes engine with custom configuration' do
      config = PdfOxide::Types::OcrConfig.new(
        detection_threshold: 0.7,
        recognition_threshold: 0.6,
        use_gpu: true
      )

      allow(FFI::Bindings).to receive(:pdf_ocr_config_create).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_ocr_config_set_detection_threshold).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_ocr_config_set_recognition_threshold).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_ocr_config_set_max_side_len).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_ocr_config_set_use_gpu).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_ocr_config_set_gpu_device_id).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_ocr_engine_create_with_config).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_ocr_config_free)

      result = manager.initialize_engine_with_config(config)
      expect(result).to be true
    end

    it 'validates config parameter' do
      expect { manager.initialize_engine_with_config('invalid') }
        .to raise_error(PdfOxide::ArgumentError, /OcrConfig/)
    end
  end

  describe '#get_character_confidences' do
    it 'returns character-level confidence scores' do
      span_handle = double(null?: false)
      allow(FFI::Bindings).to receive(:pdf_ocr_results_get_text).and_return('Hello')
      allow(FFI::StringMarshaller).to receive(:from_c).with('Hello').and_return('Hello')
      allow(FFI::Bindings).to receive(:pdf_ocr_span_get_char_confidence)
        .and_return(0.95, 0.92, 0.98, 0.87, 0.94)

      result = manager.get_character_confidences(span_handle)
      expect(result).to be_an(Array)
      expect(result.length).to eq(5)
      expect(result.first).to have_key(:character)
      expect(result.first).to have_key(:confidence)
      expect(result.first[:character]).to eq('H')
    end

    it 'returns empty array for nil handle' do
      result = manager.get_character_confidences(nil)
      expect(result).to eq([])
    end
  end

  describe '#extract_text_spans' do
    before do
      manager.instance_variable_set(:@engine_handle, mock_handle)
    end

    it 'extracts text spans with confidence' do
      span_handle = double(null?: false)
      allow(FFI::Bindings).to receive(:pdf_ocr_extract_spans).and_return(span_handle)
      allow(FFI::Bindings).to receive(:pdf_ocr_results_count).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_ocr_results_get_text).and_return('Test text')
      allow(FFI::StringMarshaller).to receive(:from_c).with('Test text').and_return('Test text')
      allow(FFI::Bindings).to receive(:pdf_ocr_span_get_bbox)
      allow(FFI::Bindings).to receive(:pdf_ocr_results_get_span).and_return(span_handle)
      allow(FFI::Bindings).to receive(:pdf_ocr_span_get_char_confidence).and_return(0.9)
      allow(FFI::Bindings).to receive(:pdf_ocr_results_free)

      result = manager.extract_text_spans(0)
      expect(result).to be_an(Array)
      expect(result.first).to have_key(:text)
      expect(result.first).to have_key(:confidence)
    end
  end

  describe '#extract_pages_detailed' do
    before do
      manager.instance_variable_set(:@engine_handle, mock_handle)
    end

    it 'extracts detailed OCR results for page range' do
      allow(manager).to receive(:page_needs_ocr?).and_return(true)
      allow(manager).to receive(:extract_text_spans).and_return([])
      allow(manager).to receive(:ocr_page).and_return('Text')
      allow(manager).to receive(:ocr_result_confidence).and_return(0.95)

      result = manager.extract_pages_detailed(0, 2)
      expect(result).to be_an(Array)
      expect(result.first).to have_key(:page)
      expect(result.first).to have_key(:text)
      expect(result.first).to have_key(:confidence)
      expect(result.first).to have_key(:spans)
    end

    it 'skips non-scanned pages when option enabled' do
      allow(manager).to receive(:page_needs_ocr?).and_return(false, true, false)

      result = manager.extract_pages_detailed(0, 2, true)
      expect(result.length).to eq(1)
    end
  end

  describe '#gpu_available?' do
    before do
      manager.instance_variable_set(:@engine_handle, mock_handle)
    end

    it 'returns whether GPU is available' do
      allow(FFI::Bindings).to receive(:pdf_ocr_gpu_available).and_return(true)

      result = manager.gpu_available?
      expect(result).to be true
    end
  end

  describe '#gpu_device_count' do
    before do
      manager.instance_variable_set(:@engine_handle, mock_handle)
    end

    it 'returns number of available GPU devices' do
      allow(FFI::Bindings).to receive(:pdf_ocr_gpu_device_count).and_return(2)

      result = manager.gpu_device_count
      expect(result).to eq(2)
    end
  end

  describe '#engine_capabilities' do
    before do
      manager.instance_variable_set(:@engine_handle, mock_handle)
    end

    it 'returns engine capabilities information' do
      allow(manager).to receive(:engine_version).and_return('5.0')
      allow(manager).to receive(:engine_status).and_return('ready')
      allow(manager).to receive(:gpu_available?).and_return(true)
      allow(manager).to receive(:gpu_device_count).and_return(1)

      result = manager.engine_capabilities
      expect(result).to be_a(Hash)
      expect(result).to have_key(:version)
      expect(result).to have_key(:gpu_available)
      expect(result).to have_key(:gpu_device_count)
      expect(result).to have_key(:supported_languages)
    end
  end

  describe '#get_supported_languages' do
    it 'returns list of supported languages' do
      result = manager.get_supported_languages
      expect(result).to be_an(Array)
      expect(result).to include('en', 'es', 'fr')
    end
  end

  describe '#batch_ocr_detailed' do
    before do
      manager.instance_variable_set(:@engine_handle, mock_handle)
    end

    it 'performs batch OCR with detailed results' do
      allow(manager).to receive(:page_needs_ocr?).and_return(true)
      allow(manager).to receive(:ocr_page).and_return('Text')
      allow(manager).to receive(:ocr_result_confidence).and_return(0.95)
      allow(manager).to receive(:extract_text_spans).and_return([])

      result = manager.batch_ocr_detailed([0, 1, 2])
      expect(result).to be_an(Array)
      expect(result.length).to eq(3)
      expect(result.first).to have_key(:page)
      expect(result.first).to have_key(:text)
      expect(result.first).to have_key(:confidence)
    end

    it 'supports progress callback' do
      allow(manager).to receive(:page_needs_ocr?).and_return(true)
      allow(manager).to receive(:ocr_page).and_return('Text')
      allow(manager).to receive(:ocr_result_confidence).and_return(0.95)
      allow(manager).to receive(:extract_text_spans).and_return([])

      callback_results = []
      manager.batch_ocr_detailed([0, 1]) do |page, result|
        callback_results << [page, result]
      end

      expect(callback_results.length).to eq(2)
    end

    it 'skips non-scanned pages with option' do
      allow(manager).to receive(:page_needs_ocr?).and_return(false, true, false)

      result = manager.batch_ocr_detailed([0, 1, 2], skip_non_scanned: true)
      expect(result.length).to eq(1)
    end
  end

  describe '#document_ocr_summary' do
    before do
      manager.instance_variable_set(:@engine_handle, mock_handle)
    end

    it 'returns document OCR summary' do
      allow(manager).to receive(:page_needs_ocr?).and_return(true, false, true)
      allow(manager).to receive(:engine_version).and_return('5.0')
      allow(manager).to receive(:gpu_available?).and_return(true)
      allow(manager).to receive(:estimate_ocr_time).and_return(5)

      result = manager.document_ocr_summary
      expect(result).to be_a(Hash)
      expect(result).to have_key(:total_pages)
      expect(result).to have_key(:pages_needing_ocr)
      expect(result).to have_key(:pages_with_text)
      expect(result).to have_key(:engine_version)
      expect(result).to have_key(:gpu_available)
      expect(result).to have_key(:estimated_processing_time_seconds)
    end
  end

  describe '#estimate_ocr_time' do
    it 'estimates OCR processing time' do
      allow(manager).to receive(:page_needs_ocr?).and_return(true, true, false)
      allow(manager).to receive(:gpu_available?).and_return(false)

      result = manager.estimate_ocr_time
      expect(result).to be_an(Integer)
      expect(result).to be >= 0
    end

    it 'applies GPU discount to estimate' do
      allow(manager).to receive(:page_needs_ocr?).and_return(true, true, false)
      allow(manager).to receive(:gpu_available?).and_return(true)

      gpu_result = manager.estimate_ocr_time

      allow(manager).to receive(:gpu_available?).and_return(false)
      cpu_result = manager.estimate_ocr_time

      expect(gpu_result).to be < cpu_result
    end
  end

  # Integration tests

  describe 'Integration: Complete OCR workflow' do
    before do
      manager.instance_variable_set(:@engine_handle, mock_handle)
    end

    it 'performs complete OCR workflow with configuration' do
      config = PdfOxide::Types::OcrConfig.balanced

      allow(FFI::Bindings).to receive(:pdf_ocr_config_create).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_ocr_config_set_detection_threshold).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_ocr_config_set_recognition_threshold).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_ocr_config_set_max_side_len).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_ocr_config_set_use_gpu).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_ocr_config_set_gpu_device_id).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_ocr_engine_create_with_config).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_ocr_config_free)

      # Reinitialize with config
      manager.initialize_engine_with_config(config)
      expect(manager.engine_handle).not_to be_nil
    end

    it 'performs OCR config preset presets' do
      presets = [:balanced, :high_accuracy, :fast, :low_resource]

      presets.each do |preset|
        config = PdfOxide::Types::OcrConfig.send(preset)
        expect(config).to be_a(PdfOxide::Types::OcrConfig)
      end
    end
  end
end

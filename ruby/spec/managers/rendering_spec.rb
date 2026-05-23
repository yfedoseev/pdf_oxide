# frozen_string_literal: true

require 'spec_helper'

RSpec.describe PdfOxide::Managers::Rendering do
  let(:mock_handle) { double(null?: false) }
  let(:mock_document) { instance_double(PdfOxide::Document, handle: mock_handle, closed?: false, page_count: 5) }
  let(:manager) { described_class.new(mock_document) }

  before do
    allow(FFI::StringMarshaller).to receive(:from_c).and_return('mock_data')
    allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('mock_utf8')
  end

  describe '#render_page_to_file' do
    it 'renders page to file successfully' do
      allow(FFI::Bindings).to receive(:pdf_render_page_to_file).and_return(true)
      allow(FFI::Types).to receive(:enum_to_value).and_return(0)

      result = manager.render_page_to_file(0, '/tmp/page.png')
      expect(result).to be true
    end

    it 'validates page index' do
      expect { manager.render_page_to_file(10, '/tmp/page.png') }.to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '#render_page_to_bytes' do
    it 'renders page to image bytes' do
      image_handle = double(null?: false)
      allow(FFI::Bindings).to receive(:pdf_render_page_to_bytes).and_return(image_handle)
      allow(FFI::Bindings).to receive(:pdf_rendered_image_size).and_return(1024)
      allow(FFI::Bindings).to receive(:pdf_rendered_image_data).and_return(double(read_bytes: 'image_data'))
      allow(FFI::Bindings).to receive(:pdf_rendered_image_free)
      allow(FFI::Types).to receive(:enum_to_value).and_return(0)

      result = manager.render_page_to_bytes(0)
      expect(result).to eq('image_data')
    end

    it 'handles nil image handle' do
      allow(FFI::Bindings).to receive(:pdf_render_page_to_bytes).and_return(nil)
      allow(FFI::Types).to receive(:enum_to_value).and_return(0)

      result = manager.render_page_to_bytes(0)
      expect(result).to be_nil
    end
  end

  describe '#render_page_range' do
    it 'renders range of pages' do
      allow(FFI::Bindings).to receive(:pdf_render_page_range).and_return(3)
      allow(FFI::Types).to receive(:enum_to_value).and_return(0)

      result = manager.render_page_range(0, 2, '/tmp/output')
      expect(result).to eq(3)
    end

    it 'validates page indices' do
      expect { manager.render_page_range(2, 1, '/tmp/output') }
        .to raise_error(PdfOxide::ArgumentError, /start_page must be <= end_page/)
    end
  end

  describe '#render_all' do
    it 'renders all pages' do
      allow(FFI::Bindings).to receive(:pdf_render_page_range).and_return(5)
      allow(FFI::Types).to receive(:enum_to_value).and_return(0)

      result = manager.render_all('/tmp/output')
      expect(result).to eq(5)
    end
  end

  describe '#render_thumbnail' do
    it 'renders page thumbnail' do
      image_handle = double(null?: false)
      allow(FFI::Bindings).to receive(:pdf_render_page_thumbnail).and_return(image_handle)
      allow(FFI::Bindings).to receive(:pdf_rendered_image_size).and_return(512)
      allow(FFI::Bindings).to receive(:pdf_rendered_image_data).and_return(double(read_bytes: 'thumb_data'))
      allow(FFI::Bindings).to receive(:pdf_rendered_image_free)

      result = manager.render_thumbnail(0, 100)
      expect(result).to eq('thumb_data')
    end
  end

  describe '#page_dimensions' do
    it 'returns page dimensions' do
      allow(FFI::Bindings).to receive(:pdf_document_get_page_width).and_return(612.0)
      allow(FFI::Bindings).to receive(:pdf_document_get_page_height).and_return(792.0)

      result = manager.page_dimensions(0)
      expect(result.width).to eq(612)
      expect(result.height).to eq(792)
    end
  end

  describe '#render_with_zoom' do
    it 'renders page with zoom level' do
      image_handle = double(null?: false)
      allow(FFI::Bindings).to receive(:pdf_render_page_zoom).and_return(image_handle)
      allow(FFI::Bindings).to receive(:pdf_rendered_image_size).and_return(2048)
      allow(FFI::Bindings).to receive(:pdf_rendered_image_data).and_return(double(read_bytes: 'zoomed_image'))
      allow(FFI::Bindings).to receive(:pdf_rendered_image_free)
      allow(FFI::Types).to receive(:enum_to_value).and_return(0)

      result = manager.render_with_zoom(0, 1.5)
      expect(result).to eq('zoomed_image')
    end

    it 'validates zoom level is positive' do
      expect { manager.render_with_zoom(0, 0) }
        .to raise_error(PdfOxide::ArgumentError, /Zoom level must be positive/)
    end
  end

  # Phase 2: Rendering Completion Methods

  describe '#render_region' do
    let(:region) { { x: 100, y: 100, width: 200, height: 200 } }

    it 'renders specific region of page' do
      image_handle = double(null?: false)
      allow(FFI::Bindings).to receive(:pdf_render_page_region).and_return(image_handle)
      allow(FFI::Bindings).to receive(:pdf_rendered_image_size).and_return(1024)
      allow(FFI::Bindings).to receive(:pdf_rendered_image_data).and_return(double(read_bytes: 'region_data'))
      allow(FFI::Bindings).to receive(:pdf_rendered_image_free)
      allow(FFI::Types).to receive(:enum_to_value).and_return(0)

      result = manager.render_region(0, region)
      expect(result).to eq('region_data')
    end

    it 'validates region hash structure' do
      expect { manager.render_region(0, { x: 100 }) }
        .to raise_error(PdfOxide::ArgumentError, /region must have :y key/)
    end

    it 'validates region dimensions are positive' do
      expect { manager.render_region(0, { x: 100, y: 100, width: 0, height: 200 }) }
        .to raise_error(PdfOxide::ArgumentError, /width must be positive/)
    end
  end

  describe '#render_fit' do
    it 'renders page fitted to dimensions' do
      image_handle = double(null?: false)
      allow(FFI::Bindings).to receive(:pdf_render_page_fit).and_return(image_handle)
      allow(FFI::Bindings).to receive(:pdf_rendered_image_size).and_return(1600)
      allow(FFI::Bindings).to receive(:pdf_rendered_image_data).and_return(double(read_bytes: 'fit_data'))
      allow(FFI::Bindings).to receive(:pdf_rendered_image_free)
      allow(FFI::Types).to receive(:enum_to_value).and_return(0)

      result = manager.render_fit(0, 400, 600)
      expect(result).to eq('fit_data')
    end

    it 'validates max dimensions are positive' do
      allow(FFI::Types).to receive(:enum_to_value).and_return(0)
      expect { manager.render_fit(0, 0, 600) }
        .to raise_error(PdfOxide::ArgumentError, /max_width must be positive/)
    end
  end

  describe '#render_to_base64' do
    it 'renders page to base64-encoded string' do
      allow(FFI::Bindings).to receive(:pdf_render_page_to_base64).and_return('base64_string_ptr')
      allow(FFI::StringMarshaller).to receive(:from_c).with('base64_string_ptr').and_return('iVBORw0KGgo...')
      allow(FFI::Types).to receive(:enum_to_value).and_return(0)

      result = manager.render_to_base64(0)
      expect(result).to eq('iVBORw0KGgo...')
    end
  end

  describe '#render_region_to_base64' do
    it 'renders region to base64-encoded string' do
      region = { x: 50, y: 50, width: 100, height: 100 }
      allow(FFI::Bindings).to receive(:pdf_rendered_image_to_base64).and_return('base64_ptr')
      allow(FFI::StringMarshaller).to receive(:from_c).with('base64_ptr').and_return('/9j/4AAQ...')
      allow(FFI::Types).to receive(:enum_to_value).and_return(1)

      result = manager.render_region_to_base64(0, region, format: :jpeg)
      expect(result).to eq('/9j/4AAQ...')
    end
  end

  describe '#estimate_render_time' do
    it 'estimates rendering time' do
      allow(FFI::Bindings).to receive(:pdf_estimate_render_time).and_return(150)
      allow(FFI::Types).to receive(:enum_to_value).and_return(0)

      result = manager.estimate_render_time(0)
      expect(result).to eq(150)
    end
  end

  describe '#renderer_statistics' do
    it 'returns renderer statistics' do
      stats_ptr = double(null?: false)
      allow(FFI::Bindings).to receive(:pdf_renderer_get_statistics).and_return(stats_ptr)
      allow(FFI::Bindings).to receive(:pdf_renderer_statistics_pages_rendered).and_return(10)
      allow(FFI::Bindings).to receive(:pdf_renderer_statistics_total_time).and_return(5000)
      allow(FFI::Bindings).to receive(:pdf_renderer_statistics_avg_time).and_return(500.0)

      result = manager.renderer_statistics
      expect(result[:pages_rendered]).to eq(10)
      expect(result[:total_time_ms]).to eq(5000)
      expect(result[:avg_time_per_page_ms]).to eq(500.0)
    end

    it 'returns empty hash if stats is nil' do
      allow(FFI::Bindings).to receive(:pdf_renderer_get_statistics).and_return(nil)

      result = manager.renderer_statistics
      expect(result).to eq({})
    end
  end

  describe '#reset_renderer_statistics' do
    it 'resets renderer statistics' do
      allow(FFI::Bindings).to receive(:pdf_renderer_reset_statistics).and_return(true)

      result = manager.reset_renderer_statistics
      expect(result).to be true
    end
  end

  describe '#convert_image_format' do
    it 'converts image between formats' do
      allow(FFI::Bindings).to receive(:pdf_rendered_image_convert).and_return(double(null?: false))
      allow(FFI::Bindings).to receive(:pdf_rendered_image_size).and_return(2048)
      allow(FFI::Bindings).to receive(:pdf_rendered_image_data).and_return(double(read_bytes: 'converted_data'))
      allow(FFI::Bindings).to receive(:pdf_rendered_image_free)
      allow(FFI::Types).to receive(:enum_to_value).and_return(1)

      result = manager.convert_image_format('png_data', :png, :jpeg)
      expect(result).to eq('converted_data')
    end

    it 'validates image_bytes is not nil' do
      expect { manager.convert_image_format(nil, :png, :jpeg) }
        .to raise_error(PdfOxide::ArgumentError, /image_bytes cannot be nil/)
    end

    it 'validates from_format is required' do
      expect { manager.convert_image_format('data', nil, :jpeg) }
        .to raise_error(PdfOxide::ArgumentError, /from_format is required/)
    end
  end

  describe '#image_to_base64' do
    it 'converts image bytes to base64' do
      allow(FFI::Bindings).to receive(:pdf_rendered_image_to_base64).and_return('base64_ptr')
      allow(FFI::StringMarshaller).to receive(:from_c).with('base64_ptr').and_return('UklGRi...')
      allow(FFI::Types).to receive(:enum_to_value).and_return(2)

      result = manager.image_to_base64('webp_data', :webp)
      expect(result).to eq('UklGRi...')
    end

    it 'validates image_bytes is not nil' do
      expect { manager.image_to_base64(nil, :png) }
        .to raise_error(PdfOxide::ArgumentError, /image_bytes cannot be nil/)
    end
  end

  describe '#mime_type_for' do
    it 'returns MIME type for PNG format' do
      allow(FFI::Bindings).to receive(:pdf_image_format_mime_type).and_return('mime_ptr')
      allow(FFI::StringMarshaller).to receive(:from_c).with('mime_ptr').and_return('image/png')
      allow(FFI::Types).to receive(:enum_to_value).and_return(0)

      result = manager.mime_type_for(:png)
      expect(result).to eq('image/png')
    end

    it 'returns MIME type for JPEG format' do
      allow(FFI::Bindings).to receive(:pdf_image_format_mime_type).and_return('mime_ptr')
      allow(FFI::StringMarshaller).to receive(:from_c).with('mime_ptr').and_return('image/jpeg')
      allow(FFI::Types).to receive(:enum_to_value).and_return(1)

      result = manager.mime_type_for(:jpeg)
      expect(result).to eq('image/jpeg')
    end
  end

  describe '#extension_for' do
    it 'returns file extension for PNG format' do
      allow(FFI::Bindings).to receive(:pdf_image_format_extension).and_return('ext_ptr')
      allow(FFI::StringMarshaller).to receive(:from_c).with('ext_ptr').and_return('.png')
      allow(FFI::Types).to receive(:enum_to_value).and_return(0)

      result = manager.extension_for(:png)
      expect(result).to eq('.png')
    end

    it 'returns file extension for WebP format' do
      allow(FFI::Bindings).to receive(:pdf_image_format_extension).and_return('ext_ptr')
      allow(FFI::StringMarshaller).to receive(:from_c).with('ext_ptr').and_return('.webp')
      allow(FFI::Types).to receive(:enum_to_value).and_return(2)

      result = manager.extension_for(:webp)
      expect(result).to eq('.webp')
    end
  end

  # Integration tests

  describe 'Integration: Complete rendering workflow' do
    it 'renders page in multiple formats' do
      # Setup mocks for all format conversions
      formats = %i[png jpeg webp]
      formats.each do |fmt|
        allow(FFI::Types).to receive(:enum_to_value).with(FFI::Types::IMAGE_FORMATS, fmt).and_return(formats.index(fmt))
      end

      allow(FFI::Bindings).to receive(:pdf_render_page_to_bytes).and_return(double(null?: false))
      allow(FFI::Bindings).to receive(:pdf_rendered_image_size).and_return(2048)
      allow(FFI::Bindings).to receive(:pdf_rendered_image_data).and_return(double(read_bytes: 'image_bytes'))
      allow(FFI::Bindings).to receive(:pdf_rendered_image_free)

      # Render in all formats
      formats.each do |fmt|
        result = manager.render_page_to_bytes(0, format: fmt)
        expect(result).to eq('image_bytes')
      end
    end

    it 'handles rendering options with presets' do
      image_handle = double(null?: false)
      allow(FFI::Bindings).to receive(:pdf_render_page_to_bytes).and_return(image_handle)
      allow(FFI::Bindings).to receive(:pdf_rendered_image_size).and_return(1024)
      allow(FFI::Bindings).to receive(:pdf_rendered_image_data).and_return(double(read_bytes: 'data'))
      allow(FFI::Bindings).to receive(:pdf_rendered_image_free)
      allow(FFI::Types).to receive(:enum_to_value).and_return(0)

      # Use preset options
      options = PdfOxide::Types::RenderOptions.high
      result = manager.render_page_to_bytes(0, options)
      expect(result).to eq('data')
    end
  end
end

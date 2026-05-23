# frozen_string_literal: true

require 'spec_helper'

RSpec.describe PdfOxide::Managers::Xfa do
  let(:mock_handle) { double(null?: false) }
  let(:mock_document) { instance_double(PdfOxide::Document, handle: mock_handle, closed?: false, page_count: 5) }
  let(:manager) { described_class.new(mock_document) }

  describe 'XFA form type constants' do
    it 'defines form type constants' do
      expect(described_class::XFA_FORM_TYPE_STATIC).to eq(1)
      expect(described_class::XFA_FORM_TYPE_DYNAMIC).to eq(2)
    end
  end

  describe '#has_xfa_forms?' do
    it 'returns true when document has XFA forms' do
      allow(FFI::Bindings).to receive(:pdf_document_has_xfa_form).and_return(true)

      result = manager.has_xfa_forms?
      expect(result).to be true
    end

    it 'returns false when document has no XFA forms' do
      allow(FFI::Bindings).to receive(:pdf_document_has_xfa_form).and_return(false)

      result = manager.has_xfa_forms?
      expect(result).to be false
    end
  end

  describe '#parse_xfa_form' do
    it 'returns XFA form data' do
      allow(manager).to receive(:has_xfa_forms?).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_parse_xfa_form).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_field_count).and_return(3)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_free)

      result = manager.parse_xfa_form
      expect(result).to be_a(Hash)
      expect(result[:field_count]).to eq(3)
    end

    it 'raises error when document has no XFA forms' do
      allow(manager).to receive(:has_xfa_forms?).and_return(false)

      expect { manager.parse_xfa_form }
        .to raise_error(PdfOxide::OperationError)
    end
  end

  describe '#get_xfa_form_type' do
    it 'returns form type as symbol' do
      allow(manager).to receive(:has_xfa_forms?).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_parse_xfa_form).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_get_xfa_form_type).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_free)

      result = manager.get_xfa_form_type
      expect(result).to eq(:static)
    end

    it 'returns :dynamic for dynamic forms' do
      allow(manager).to receive(:has_xfa_forms?).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_parse_xfa_form).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_get_xfa_form_type).and_return(2)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_free)

      result = manager.get_xfa_form_type
      expect(result).to eq(:dynamic)
    end

    it 'returns :unknown for unknown types' do
      allow(manager).to receive(:has_xfa_forms?).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_parse_xfa_form).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_get_xfa_form_type).and_return(99)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_free)

      result = manager.get_xfa_form_type
      expect(result).to eq(:unknown)
    end

    it 'raises error when document has no XFA forms' do
      allow(manager).to receive(:has_xfa_forms?).and_return(false)

      expect { manager.get_xfa_form_type }
        .to raise_error(PdfOxide::OperationError)
    end
  end

  describe '#get_xfa_form_title' do
    it 'returns form title' do
      allow(manager).to receive(:has_xfa_forms?).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_parse_xfa_form).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_get_title).and_return('Form Title')
      allow(FFI::Bindings).to receive(:pdf_xfa_form_free)
      allow(FFI::StringMarshaller).to receive(:from_c_string).and_return('Form Title')

      result = manager.get_xfa_form_title
      expect(result).to eq('Form Title')
    end

    it 'returns empty string when title is nil' do
      allow(manager).to receive(:has_xfa_forms?).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_parse_xfa_form).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_get_title).and_return(nil)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_free)
      allow(FFI::StringMarshaller).to receive(:from_c_string).and_return(nil)

      result = manager.get_xfa_form_title
      expect(result).to eq('')
    end
  end

  describe '#get_xfa_page_count' do
    it 'returns page count' do
      allow(manager).to receive(:has_xfa_forms?).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_parse_xfa_form).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_page_count).and_return(10)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_free)

      result = manager.get_xfa_page_count
      expect(result).to eq(10)
    end

    it 'returns 0 when no form' do
      allow(manager).to receive(:has_xfa_forms?).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_parse_xfa_form).and_return(nil)

      result = manager.get_xfa_page_count
      expect(result).to eq(0)
    end
  end

  describe '#get_xfa_field_label' do
    it 'returns field label' do
      allow(manager).to receive(:has_xfa_forms?).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_parse_xfa_form).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_field_count).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_get_field).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_xfa_field_get_name).and_return('field1')
      allow(FFI::Bindings).to receive(:pdf_xfa_field_get_label).and_return('Field Label')
      allow(FFI::Bindings).to receive(:pdf_xfa_field_free)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_free)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('field1')
      allow(FFI::StringMarshaller).to receive(:read_c_string).and_return('field1')
      allow(FFI::StringMarshaller).to receive(:from_c_string).and_return('Field Label')

      result = manager.get_xfa_field_label('field1')
      expect(result).to eq('Field Label')
    end

    it 'validates field name' do
      expect { manager.get_xfa_field_label('') }
        .to raise_error(PdfOxide::ArgumentError)
    end

    it 'raises error when document has no XFA forms' do
      allow(manager).to receive(:has_xfa_forms?).and_return(false)

      expect { manager.get_xfa_field_label('field1') }
        .to raise_error(PdfOxide::OperationError)
    end
  end

  describe '#is_xfa_field_required?' do
    it 'returns true for required fields' do
      allow(manager).to receive(:has_xfa_forms?).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_parse_xfa_form).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_field_count).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_get_field).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_xfa_field_get_name).and_return('required_field')
      allow(FFI::Bindings).to receive(:pdf_xfa_field_is_required).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_xfa_field_free)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_free)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('required_field')
      allow(FFI::StringMarshaller).to receive(:read_c_string).and_return('required_field')

      result = manager.is_xfa_field_required?('required_field')
      expect(result).to be true
    end

    it 'returns false for optional fields' do
      allow(manager).to receive(:has_xfa_forms?).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_parse_xfa_form).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_field_count).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_get_field).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_xfa_field_get_name).and_return('optional_field')
      allow(FFI::Bindings).to receive(:pdf_xfa_field_is_required).and_return(false)
      allow(FFI::Bindings).to receive(:pdf_xfa_field_free)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_free)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('optional_field')
      allow(FFI::StringMarshaller).to receive(:read_c_string).and_return('optional_field')

      result = manager.is_xfa_field_required?('optional_field')
      expect(result).to be false
    end
  end

  describe '#is_xfa_field_readonly?' do
    it 'returns true for read-only fields' do
      allow(manager).to receive(:has_xfa_forms?).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_parse_xfa_form).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_field_count).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_get_field).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_xfa_field_get_name).and_return('readonly_field')
      allow(FFI::Bindings).to receive(:pdf_xfa_field_is_readonly).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_xfa_field_free)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_free)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('readonly_field')
      allow(FFI::StringMarshaller).to receive(:read_c_string).and_return('readonly_field')

      result = manager.is_xfa_field_readonly?('readonly_field')
      expect(result).to be true
    end

    it 'returns false for editable fields' do
      allow(manager).to receive(:has_xfa_forms?).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_parse_xfa_form).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_field_count).and_return(1)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_get_field).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_xfa_field_get_name).and_return('editable_field')
      allow(FFI::Bindings).to receive(:pdf_xfa_field_is_readonly).and_return(false)
      allow(FFI::Bindings).to receive(:pdf_xfa_field_free)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_free)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('editable_field')
      allow(FFI::StringMarshaller).to receive(:read_c_string).and_return('editable_field')

      result = manager.is_xfa_field_readonly?('editable_field')
      expect(result).to be false
    end
  end

  describe '#xfa_dataset_to_json' do
    it 'returns dataset as JSON hash' do
      allow(manager).to receive(:has_xfa_forms?).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_parse_xfa_form).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_get_dataset).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_xfa_dataset_to_json).and_return('{"field1": "value1"}')
      allow(FFI::Bindings).to receive(:pdf_xfa_dataset_free)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_free)
      allow(FFI::StringMarshaller).to receive(:from_c_string).and_return('{"field1": "value1"}')

      result = manager.xfa_dataset_to_json
      expect(result).to be_a(Hash)
      expect(result['field1']).to eq('value1')
    end

    it 'returns empty hash on error' do
      allow(manager).to receive(:has_xfa_forms?).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_parse_xfa_form).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_get_dataset).and_return(nil)
      allow(FFI::Bindings).to receive(:pdf_xfa_form_free)

      result = manager.xfa_dataset_to_json
      expect(result).to eq({})
    end

    it 'raises error when document has no XFA forms' do
      allow(manager).to receive(:has_xfa_forms?).and_return(false)

      expect { manager.xfa_dataset_to_json }
        .to raise_error(PdfOxide::OperationError)
    end
  end

  describe '#extract_xfa_as_fdf' do
    it 'returns FDF data' do
      allow(manager).to receive(:has_xfa_forms?).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_extract_xfa_as_fdf).and_return('%FDF-1.2')
      allow(FFI::StringMarshaller).to receive(:from_c_string).and_return('%FDF-1.2')

      result = manager.extract_xfa_as_fdf
      expect(result).to include('%FDF')
    end

    it 'raises error when document has no XFA forms' do
      allow(manager).to receive(:has_xfa_forms?).and_return(false)

      expect { manager.extract_xfa_as_fdf }
        .to raise_error(PdfOxide::OperationError)
    end
  end

  describe '#get_xfa_template_xml' do
    it 'returns template XML' do
      allow(manager).to receive(:has_xfa_forms?).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_get_xfa_template_xml).and_return('<?xml version="1.0"?>')
      allow(FFI::StringMarshaller).to receive(:from_c_string).and_return('<?xml version="1.0"?>')

      result = manager.get_xfa_template_xml
      expect(result).to include('<?xml')
    end

    it 'raises error when document has no XFA forms' do
      allow(manager).to receive(:has_xfa_forms?).and_return(false)

      expect { manager.get_xfa_template_xml }
        .to raise_error(PdfOxide::OperationError)
    end
  end

  describe '#xfa_statistics' do
    it 'returns statistics' do
      allow(manager).to receive(:has_xfa_forms?).and_return(true)
      allow(manager).to receive(:get_xfa_field_count).and_return(3)
      allow(manager).to receive(:get_xfa_field_names).and_return(['field1', 'field2', 'field3'])

      result = manager.xfa_statistics
      expect(result[:has_xfa]).to be true
      expect(result[:field_count]).to eq(3)
      expect(result[:field_names]).to have_length(3)
      expect(result).to have_key(:timestamp)
    end
  end
end

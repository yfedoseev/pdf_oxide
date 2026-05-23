# frozen_string_literal: true

require 'spec_helper'

RSpec.describe PdfOxide::Managers::Form do
  let(:mock_handle) { double(null?: false) }
  let(:mock_document) { instance_double(PdfOxide::Document, handle: mock_handle, closed?: false, page_count: 5) }
  let(:manager) { described_class.new(mock_document) }

  describe 'field type constants' do
    it 'defines all field types' do
      expect(described_class::FIELD_TYPE_TEXT).to eq(4)
      expect(described_class::FIELD_TYPE_CHECKBOX).to eq(2)
      expect(described_class::FIELD_TYPE_RADIOBUTTON).to eq(3)
    end
  end

  describe '#form_field_count' do
    it 'returns number of form fields' do
      allow(FFI::Bindings).to receive(:pdf_document_get_form_field_count).and_return(5)

      result = manager.form_field_count
      expect(result).to eq(5)
    end

    it 'returns zero when no form fields' do
      allow(FFI::Bindings).to receive(:pdf_document_get_form_field_count).and_return(0)

      result = manager.form_field_count
      expect(result).to eq(0)
    end
  end

  describe '#has_acro_forms?' do
    it 'returns true when document has AcroForm fields' do
      allow(FFI::Bindings).to receive(:pdf_document_has_acro_forms).and_return(true)

      result = manager.has_acro_forms?
      expect(result).to be true
    end

    it 'returns false when document has no AcroForm fields' do
      allow(FFI::Bindings).to receive(:pdf_document_has_acro_forms).and_return(false)

      result = manager.has_acro_forms?
      expect(result).to be false
    end
  end

  describe '#get_form_field_value' do
    it 'returns form field value' do
      allow(FFI::Bindings).to receive(:pdf_document_get_form_field_value).and_return('John Doe')
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('field_name')
      allow(FFI::StringMarshaller).to receive(:from_c_string).with('John Doe').and_return('John Doe')

      result = manager.get_form_field_value('name_field')
      expect(result).to eq('John Doe')
    end

    it 'raises error when field name is empty' do
      expect {
        manager.get_form_field_value('')
      }.to raise_error(PdfOxide::ArgumentError)
    end

    it 'raises error when field name is nil' do
      expect {
        manager.get_form_field_value(nil)
      }.to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '#set_form_field_value' do
    it 'sets form field value' do
      expect(FFI::Bindings).to receive(:pdf_document_set_form_field_value).and_return(true)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('value')

      result = manager.set_form_field_value('name_field', 'John Doe')
      expect(result).to be true
    end

    it 'converts field name and value to UTF-8' do
      allow(FFI::Bindings).to receive(:pdf_document_set_form_field_value)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_call_original

      manager.set_form_field_value('field', 'value')
      expect(FFI::StringMarshaller).to have_received(:to_utf8).with('field')
      expect(FFI::StringMarshaller).to have_received(:to_utf8).with('value')
    end

    it 'converts numeric values to string' do
      allow(FFI::Bindings).to receive(:pdf_document_set_form_field_value)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('42')

      manager.set_form_field_value('age_field', 42)
      expect(FFI::StringMarshaller).to have_received(:to_utf8).with('42')
    end
  end

  describe '#get_form_field_type' do
    it 'returns form field type' do
      allow(FFI::Bindings).to receive(:pdf_document_get_form_field_type).and_return(4)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('field_name')

      result = manager.get_form_field_type('name_field')
      expect(result).to eq(:text)
    end

    it 'returns integer for unknown field type' do
      allow(FFI::Bindings).to receive(:pdf_document_get_form_field_type).and_return(99)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('field_name')

      result = manager.get_form_field_type('field')
      expect(result).to eq(99)
    end
  end

  describe '#reset_form_fields' do
    it 'resets all form fields to default values' do
      expect(FFI::Bindings).to receive(:pdf_document_reset_form_fields).and_return(true)

      result = manager.reset_form_fields
      expect(result).to be true
    end
  end

  describe '#flatten_forms' do
    it 'flattens form fields' do
      expect(FFI::Bindings).to receive(:pdf_document_flatten_forms).and_return(true)

      result = manager.flatten_forms
      expect(result).to be true
    end
  end

  describe '#get_all_values' do
    it 'returns hash of all field values' do
      allow(manager).to receive(:form_field_names).and_return(['name', 'email'])
      allow(manager).to receive(:get_form_field_value).and_return('value')

      result = manager.get_all_values
      expect(result).to be_a(Hash)
      expect(result).to have_key('name')
      expect(result).to have_key('email')
    end
  end

  describe '#set_all_values' do
    it 'sets multiple form field values' do
      allow(manager).to receive(:set_form_field_value).and_return(true)

      result = manager.set_all_values({ 'name' => 'John', 'email' => 'john@example.com' })
      expect(result).to be true
      expect(manager).to have_received(:set_form_field_value).twice
    end
  end

  describe '#form_info' do
    it 'returns form information' do
      allow(manager).to receive(:has_acro_forms?).and_return(true)
      allow(manager).to receive(:has_xfa_forms?).and_return(false)
      allow(manager).to receive(:form_field_count).and_return(3)
      allow(manager).to receive(:get_all_form_fields).and_return([])

      result = manager.form_info
      expect(result).to be_a(Hash)
      expect(result).to have_key(:has_acro_forms)
      expect(result).to have_key(:field_count)
    end
  end

  describe '#form_statistics' do
    it 'returns form statistics' do
      allow(manager).to receive(:get_all_form_fields).and_return([
        double(type: 4, value: 'test'),
        double(type: 2, value: nil),
        double(type: 4, value: '')
      ])

      result = manager.form_statistics
      expect(result).to be_a(Hash)
      expect(result).to have_key(:total_fields)
      expect(result[:total_fields]).to eq(3)
    end
  end

  describe '#export_to_fdf' do
    it 'exports form data to FDF file' do
      expect(FFI::Bindings).to receive(:pdf_form_export_to_fdf).and_return(true)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('/path/to/output.fdf')

      result = manager.export_to_fdf('/path/to/output.fdf')
      expect(result).to be true
    end

    it 'raises error when output path is empty' do
      expect { manager.export_to_fdf('') }.to raise_error(PdfOxide::ArgumentError)
    end

    it 'raises error when output path is nil' do
      expect { manager.export_to_fdf(nil) }.to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '#export_to_xfdf' do
    it 'exports form data to XFDF file' do
      expect(FFI::Bindings).to receive(:pdf_form_export_to_xfdf).and_return(true)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('/path/to/output.xfdf')

      result = manager.export_to_xfdf('/path/to/output.xfdf')
      expect(result).to be true
    end

    it 'raises error when output path is empty' do
      expect { manager.export_to_xfdf('') }.to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '#import_from_file' do
    it 'imports form data from file' do
      allow(File).to receive(:exist?).and_return(true)
      expect(FFI::Bindings).to receive(:pdf_form_import_from_file).and_return(true)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('/path/to/input.xfdf')

      result = manager.import_from_file('/path/to/input.xfdf')
      expect(result).to be true
    end

    it 'raises error when input path is empty' do
      expect { manager.import_from_file('') }.to raise_error(PdfOxide::ArgumentError)
    end

    it 'raises error when file does not exist' do
      allow(File).to receive(:exist?).and_return(false)

      expect { manager.import_from_file('/nonexistent.xfdf') }.to raise_error(PdfOxide::FileNotFoundError)
    end
  end

  describe '#reset_all_fields' do
    it 'resets all form fields' do
      expect(FFI::Bindings).to receive(:pdf_form_reset_all_fields).and_return(true)

      result = manager.reset_all_fields
      expect(result).to be true
    end
  end

  describe '#find_field_by_name' do
    it 'finds field by name and returns index' do
      allow(FFI::Bindings).to receive(:pdf_form_field_find_by_name).and_return(2)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('field_name')

      result = manager.find_field_by_name('Address.City')
      expect(result).to eq(2)
    end

    it 'returns -1 when field not found' do
      allow(FFI::Bindings).to receive(:pdf_form_field_find_by_name).and_return(-1)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('field_name')

      result = manager.find_field_by_name('NonExistent')
      expect(result).to eq(-1)
    end

    it 'raises error when field name is empty' do
      expect { manager.find_field_by_name('') }.to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '#set_field_value_by_name' do
    it 'sets string value using string function' do
      expect(FFI::Bindings).to receive(:pdf_form_field_set_value_by_name_string).and_return(true)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('value')

      result = manager.set_field_value_by_name('Name', 'John Doe')
      expect(result).to be true
    end

    it 'sets boolean value using boolean function' do
      expect(FFI::Bindings).to receive(:pdf_form_field_set_value_by_name_boolean).and_return(true)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('field_name')

      result = manager.set_field_value_by_name('Checkbox1', true)
      expect(result).to be true
    end

    it 'converts numeric values to string' do
      allow(FFI::Bindings).to receive(:pdf_form_field_set_value_by_name_string).and_return(true)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_call_original

      manager.set_field_value_by_name('Age', 42)
      expect(FFI::StringMarshaller).to have_received(:to_utf8).with('42')
    end

    it 'raises error when field name is empty' do
      expect { manager.set_field_value_by_name('', 'value') }.to raise_error(PdfOxide::ArgumentError)
    end
  end
end

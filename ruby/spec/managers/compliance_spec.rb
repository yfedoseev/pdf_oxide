# frozen_string_literal: true

require 'spec_helper'

RSpec.describe PdfOxide::Managers::Compliance do
  let(:mock_handle) { double(null?: false) }
  let(:mock_document) { instance_double(PdfOxide::Document, handle: mock_handle, closed?: false, page_count: 5) }
  let(:manager) { described_class.new(mock_document) }

  describe 'compliance constants' do
    it 'defines PDF/A levels' do
      expect(described_class::PDF_A_1).to eq(1)
      expect(described_class::PDF_A_2).to eq(2)
      expect(described_class::PDF_A_3).to eq(3)
    end

    it 'defines PDF/X standards' do
      expect(described_class::PDF_X_1).to eq(1)
      expect(described_class::PDF_X_3).to eq(3)
      expect(described_class::PDF_X_4).to eq(4)
    end
  end

  describe '#validate_pdf_a' do
    it 'validates PDF/A compliance' do
      allow(FFI::Bindings).to receive(:pdf_validate_pdf_a).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_pdf_a_is_compliant).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_pdf_a_error_count).and_return(0)
      allow(FFI::Bindings).to receive(:pdf_pdf_a_warning_count).and_return(0)
      allow(FFI::Bindings).to receive(:pdf_pdf_a_get_report).and_return('Report')
      allow(FFI::Bindings).to receive(:pdf_pdf_a_results_free)
      allow(FFI::StringMarshaller).to receive(:from_c_string).and_return('Report')

      result = manager.validate_pdf_a(described_class::PDF_A_1)
      expect(result).to be_a(Hash)
      expect(result[:compliant]).to be true
    end

    it 'validates compliance level' do
      expect { manager.validate_pdf_a(99) }.to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '#is_pdf_a?' do
    it 'returns true when document is PDF/A compliant' do
      allow(manager).to receive(:validate_pdf_a).and_return({ compliant: true })

      result = manager.is_pdf_a?
      expect(result).to be true
    end

    it 'returns false when document is not PDF/A compliant' do
      allow(manager).to receive(:validate_pdf_a).and_return({ compliant: false })

      result = manager.is_pdf_a?
      expect(result).to be false
    end
  end

  describe '#validate_pdf_x' do
    it 'validates PDF/X compliance' do
      allow(FFI::Bindings).to receive(:pdf_validate_pdf_x).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_pdf_x_is_compliant).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_pdf_x_error_count).and_return(0)
      allow(FFI::Bindings).to receive(:pdf_pdf_x_warning_count).and_return(0)
      allow(FFI::Bindings).to receive(:pdf_pdf_x_get_report).and_return('Report')
      allow(FFI::Bindings).to receive(:pdf_pdf_x_results_free)
      allow(FFI::StringMarshaller).to receive(:from_c_string).and_return('Report')

      result = manager.validate_pdf_x(described_class::PDF_X_1)
      expect(result).to be_a(Hash)
      expect(result[:compliant]).to be true
    end
  end

  describe '#is_pdf_x?' do
    it 'returns true when document is PDF/X compliant' do
      allow(manager).to receive(:validate_pdf_x).and_return({ compliant: true })

      result = manager.is_pdf_x?
      expect(result).to be true
    end
  end

  describe '#validate_pdf_ua' do
    it 'validates PDF/UA compliance' do
      allow(FFI::Bindings).to receive(:pdf_validate_pdf_ua).and_return(mock_handle)
      allow(FFI::Bindings).to receive(:pdf_pdf_ua_is_accessible).and_return(true)
      allow(FFI::Bindings).to receive(:pdf_pdf_ua_error_count).and_return(0)
      allow(FFI::Bindings).to receive(:pdf_pdf_ua_results_free)

      result = manager.validate_pdf_ua
      expect(result).to be_a(Hash)
      expect(result[:compliant]).to be true
    end
  end

  describe '#is_pdf_ua?' do
    it 'returns true when document is PDF/UA accessible' do
      allow(manager).to receive(:validate_pdf_ua).and_return({ compliant: true })

      result = manager.is_pdf_ua?
      expect(result).to be true
    end
  end

  describe '#convert_to_pdf_a' do
    it 'converts document to PDF/A' do
      expect(FFI::Bindings).to receive(:pdf_convert_to_pdf_a).and_return(true)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('output.pdf')

      result = manager.convert_to_pdf_a(described_class::PDF_A_1, 'output.pdf')
      expect(result).to be true
    end

    it 'raises error when output path is empty' do
      expect {
        manager.convert_to_pdf_a(described_class::PDF_A_1, '')
      }.to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '#convert_to_pdf_x' do
    it 'converts document to PDF/X' do
      expect(FFI::Bindings).to receive(:pdf_convert_to_pdf_x).and_return(true)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('output.pdf')

      result = manager.convert_to_pdf_x(described_class::PDF_X_1, 'output.pdf')
      expect(result).to be true
    end
  end

  describe '#convert_to_pdf_ua' do
    it 'converts document to PDF/UA' do
      expect(FFI::Bindings).to receive(:pdf_convert_to_pdf_ua).and_return(true)
      allow(FFI::StringMarshaller).to receive(:to_utf8).and_return('output.pdf')

      result = manager.convert_to_pdf_ua('output.pdf')
      expect(result).to be true
    end
  end

  describe '#get_validation_errors' do
    it 'returns validation errors for PDF/A' do
      allow(manager).to receive(:validate_pdf_a).and_return({ errors: ['Error 1'] })

      result = manager.get_validation_errors(:pdf_a)
      expect(result).to be_an(Array)
    end

    it 'raises error for unknown compliance type' do
      expect {
        manager.get_validation_errors(:unknown)
      }.to raise_error(PdfOxide::ArgumentError)
    end
  end

  describe '#compliance_info' do
    it 'returns compliance information' do
      allow(manager).to receive(:is_pdf_a?).and_return(true)
      allow(manager).to receive(:is_pdf_x?).and_return(false)
      allow(manager).to receive(:is_pdf_ua?).and_return(true)

      result = manager.compliance_info
      expect(result).to be_a(Hash)
      expect(result).to have_key(:pdf_a)
      expect(result).to have_key(:pdf_x)
      expect(result).to have_key(:pdf_ua)
    end
  end

  describe '#validate_all_standards' do
    it 'validates all standards at once' do
      allow(manager).to receive(:validate_pdf_a).and_return({ compliant: true })
      allow(manager).to receive(:validate_pdf_x).and_return({ compliant: true })
      allow(manager).to receive(:validate_pdf_ua).and_return({ compliant: true })
      allow(manager).to receive(:is_pdf_a?).and_return(true)
      allow(manager).to receive(:is_pdf_x?).and_return(true)
      allow(manager).to receive(:is_pdf_ua?).and_return(true)
      allow(manager).to receive(:get_validation_errors).and_return([])

      result = manager.validate_all_standards
      expect(result).to be_a(Hash)
      expect(result).to have_key(:pdf_a)
      expect(result).to have_key(:pdf_x)
      expect(result).to have_key(:pdf_ua)
      expect(result).to have_key(:summary)
      expect(result[:summary][:pdf_a_compliant]).to be true
    end

    it 'includes total issues in summary' do
      allow(manager).to receive(:validate_pdf_a).and_return({ compliant: false })
      allow(manager).to receive(:validate_pdf_x).and_return({ compliant: false })
      allow(manager).to receive(:validate_pdf_ua).and_return({ compliant: false })
      allow(manager).to receive(:is_pdf_a?).and_return(false)
      allow(manager).to receive(:is_pdf_x?).and_return(false)
      allow(manager).to receive(:is_pdf_ua?).and_return(false)
      allow(manager).to receive(:get_validation_errors).and_return([{}, {}, {}])

      result = manager.validate_all_standards
      expect(result[:summary][:total_issues]).to eq(9)
    end
  end

  describe '#get_compliance_recommendations' do
    it 'provides recommendations for non-compliant PDF/A' do
      allow(manager).to receive(:is_pdf_a?).and_return(false)
      allow(manager).to receive(:is_pdf_x?).and_return(true)
      allow(manager).to receive(:is_pdf_ua?).and_return(true)
      allow(manager).to receive(:get_validation_errors).with(:pdf_a).and_return([{}])

      result = manager.get_compliance_recommendations
      expect(result).to be_an(Array)
      expect(result.any? { |r| r.include?('PDF/A') }).to be true
    end

    it 'provides recommendations for non-compliant PDF/UA' do
      allow(manager).to receive(:is_pdf_a?).and_return(true)
      allow(manager).to receive(:is_pdf_x?).and_return(true)
      allow(manager).to receive(:is_pdf_ua?).and_return(false)
      allow(manager).to receive(:get_validation_errors).with(:pdf_ua).and_return([{}])

      result = manager.get_compliance_recommendations
      expect(result.any? { |r| r.include?('accessible') }).to be true
    end

    it 'provides positive recommendations when compliant' do
      allow(manager).to receive(:is_pdf_a?).and_return(true)
      allow(manager).to receive(:is_pdf_x?).and_return(true)
      allow(manager).to receive(:is_pdf_ua?).and_return(true)

      result = manager.get_compliance_recommendations
      expect(result.any? { |r| r.include?('compliant') }).to be true
    end
  end

  describe '#pdf_a_level_to_string' do
    it 'converts PDF/A-1 to string' do
      result = manager.pdf_a_level_to_string(described_class::PDF_A_1)
      expect(result).to eq('PDF/A-1')
    end

    it 'converts PDF/A-2 to string' do
      result = manager.pdf_a_level_to_string(described_class::PDF_A_2)
      expect(result).to eq('PDF/A-2')
    end

    it 'converts PDF/A-3 to string' do
      result = manager.pdf_a_level_to_string(described_class::PDF_A_3)
      expect(result).to eq('PDF/A-3')
    end

    it 'handles unknown levels' do
      result = manager.pdf_a_level_to_string(99)
      expect(result).to eq('PDF/A-Unknown')
    end
  end

  describe '#pdf_x_standard_to_string' do
    it 'converts PDF/X-1 to string' do
      result = manager.pdf_x_standard_to_string(described_class::PDF_X_1)
      expect(result).to eq('PDF/X-1')
    end

    it 'converts PDF/X-3 to string' do
      result = manager.pdf_x_standard_to_string(described_class::PDF_X_3)
      expect(result).to eq('PDF/X-3')
    end

    it 'converts PDF/X-4 to string' do
      result = manager.pdf_x_standard_to_string(described_class::PDF_X_4)
      expect(result).to eq('PDF/X-4')
    end

    it 'handles unknown standards' do
      result = manager.pdf_x_standard_to_string(99)
      expect(result).to eq('PDF/X-Unknown')
    end
  end

  describe '#pdf_ua_to_string' do
    it 'returns PDF/UA-1' do
      result = manager.pdf_ua_to_string
      expect(result).to eq('PDF/UA-1')
    end
  end
end

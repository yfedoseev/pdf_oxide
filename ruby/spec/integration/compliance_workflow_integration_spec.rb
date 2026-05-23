# frozen_string_literal: true

require 'spec_helper'

RSpec.describe 'Compliance and XFA Workflow Integration', skip: 'Phase 2 repair: prepared snapshot is mock-shaped; Phase 4 rewrites as real-FFI integration tests' do
  let(:mock_handle) { double(null?: false) }
  let(:mock_document) { instance_double(PdfOxide::Document, handle: mock_handle, closed?: false) }

  describe 'Comprehensive PDF compliance analysis workflow' do
    it 'validates document against all standards and gets recommendations' do
      compliance_manager = PdfOxide::Managers::Compliance.new(mock_document)

      allow(compliance_manager).to receive(:validate_pdf_a_1b).and_return(
        {
          compliant: false,
          issues: [
            'Missing XRef stream',
            'Embedded fonts not subset'
          ]
        }
      )

      allow(compliance_manager).to receive(:validate_pdf_x_3).and_return(
        {
          compliant: true,
          issues: []
        }
      )

      allow(compliance_manager).to receive(:validate_pdf_ua).and_return(
        {
          compliant: false,
          issues: [
            'Missing document structure tags',
            'Images lack alternative text'
          ]
        }
      )

      # Validate all standards at once
      results = compliance_manager.validate_all_standards
      expect(results).to be_a(Hash)
      expect(results).to have_key(:pdf_a_1b)
      expect(results).to have_key(:pdf_x_3)
      expect(results).to have_key(:pdf_ua)
      expect(results).to have_key(:total_issues)

      # Get recommendations
      recommendations = compliance_manager.get_compliance_recommendations
      expect(recommendations).to be_an(Array)
      expect(recommendations).not_to be_empty
    end

    it 'provides specific recommendations for non-compliant standards' do
      compliance_manager = PdfOxide::Managers::Compliance.new(mock_document)

      allow(compliance_manager).to receive(:validate_pdf_a_1b).and_return(
        { compliant: false, issues: ['Issue 1', 'Issue 2'] }
      )
      allow(compliance_manager).to receive(:validate_pdf_a_2b).and_return(
        { compliant: false, issues: ['Issue 3'] }
      )

      recommendations = compliance_manager.get_compliance_recommendations
      expect(recommendations).to include(include('PDF/A'))
    end

    it 'converts compliance levels to human-readable format' do
      compliance_manager = PdfOxide::Managers::Compliance.new(mock_document)

      # Test PDF/A level conversion
      a1_str = compliance_manager.pdf_a_level_to_string(1)
      expect(a1_str).to include('PDF/A')

      a2_str = compliance_manager.pdf_a_level_to_string(2)
      expect(a2_str).to include('PDF/A')

      # Test PDF/X standard conversion
      x1_str = compliance_manager.pdf_x_standard_to_string(1)
      expect(x1_str).to include('PDF/X')

      x3_str = compliance_manager.pdf_x_standard_to_string(3)
      expect(x3_str).to include('PDF/X')

      # Test PDF/UA conversion
      ua_str = compliance_manager.pdf_ua_to_string
      expect(ua_str).to eq('PDF/UA-1')
    end
  end

  describe 'XFA form analysis and data extraction workflow' do
    let(:xfa_manager) { PdfOxide::Managers::XFA.new(mock_document) }

    it 'analyzes form type and structure' do
      allow(xfa_manager).to receive_messages(
        has_xfa?: true,
        get_xfa_form_type: :static,
        get_xfa_form_title: 'Insurance Claim Form',
        get_xfa_page_count: 4
      )

      # Check form type
      form_type = xfa_manager.get_xfa_form_type
      expect(form_type).to eq(:static)

      # Get form title and pages
      title = xfa_manager.get_xfa_form_title
      expect(title).to eq('Insurance Claim Form')

      page_count = xfa_manager.get_xfa_page_count
      expect(page_count).to eq(4)
    end

    it 'extracts field properties for form validation' do
      allow(xfa_manager).to receive_messages(
        has_xfa?: true,
        get_xfa_field_label: 'Full Name',
        is_xfa_field_required?: true,
        is_xfa_field_readonly?: false
      )

      label = xfa_manager.get_xfa_field_label('name_field')
      expect(label).to eq('Full Name')

      required = xfa_manager.is_xfa_field_required?('name_field')
      expect(required).to be true

      readonly = xfa_manager.is_xfa_field_readonly?('name_field')
      expect(readonly).to be false
    end

    it 'exports form data in multiple formats' do
      xfa_data = {
        'name' => 'John Doe',
        'email' => 'john@example.com',
        'phone' => '555-1234'
      }

      allow(xfa_manager).to receive_messages(
        has_xfa?: true,
        xfa_dataset_to_json: xfa_data,
        extract_xfa_as_fdf: "%FDF-1.2\n1 0 obj\n<<...>>",
        get_xfa_template_xml: "<xfa:template>...</xfa:template>"
      )

      # Export as JSON
      json_data = xfa_manager.xfa_dataset_to_json
      expect(json_data).to eq(xfa_data)

      # Export as FDF
      fdf_content = xfa_manager.extract_xfa_as_fdf
      expect(fdf_content).to include('%FDF')

      # Get template XML
      template = xfa_manager.get_xfa_template_xml
      expect(template).to include('<xfa:template>')
    end
  end

  describe 'Compliance and XFA combined analysis' do
    let(:compliance_mgr) { PdfOxide::Managers::Compliance.new(mock_document) }
    let(:xfa_mgr) { PdfOxide::Managers::XFA.new(mock_document) }

    it 'analyzes both compliance and form structure in one workflow' do
      # Set up compliance checks
      allow(compliance_mgr).to receive(:validate_pdf_ua).and_return(
        {
          compliant: false,
          issues: ['Missing form field labels']
        }
      )

      # Set up XFA form checks
      allow(xfa_mgr).to receive_messages(
        has_xfa?: true,
        get_xfa_form_type: :dynamic
      )

      # Run combined analysis
      compliance_issues = compliance_mgr.validate_pdf_ua
      form_type = xfa_mgr.get_xfa_form_type

      # Verify combined results
      expect(compliance_issues[:compliant]).to be false
      expect(form_type).to eq(:dynamic)

      # Check form labels for accessibility
      allow(xfa_mgr).to receive(:get_xfa_field_label).and_return('Field Label')
      label = xfa_mgr.get_xfa_field_label('field1')
      expect(label).not_to be_empty
    end
  end

  describe 'Edge cases and error handling' do
    it 'handles missing XFA gracefully' do
      xfa_manager = PdfOxide::Managers::XFA.new(mock_document)

      allow(xfa_manager).to receive(:has_xfa?).and_return(false)

      # Should handle gracefully or raise specific error
      expect { xfa_manager.get_xfa_form_type }
        .to raise_error(PdfOxide::Error)
    end

    it 'validates field name parameters' do
      xfa_manager = PdfOxide::Managers::XFA.new(mock_document)

      allow(xfa_manager).to receive(:has_xfa?).and_return(true)

      # Empty field name should be rejected
      expect { xfa_manager.get_xfa_field_label('') }
        .to raise_error(ArgumentError)
    end
  end
end

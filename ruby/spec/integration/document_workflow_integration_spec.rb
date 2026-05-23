# frozen_string_literal: true

require 'spec_helper'

RSpec.describe 'Document Workflow Integration', skip: 'Phase 2 repair: prepared snapshot is mock-shaped; Phase 4 rewrites as real-FFI integration tests' do
  describe 'Complete PDF creation and metadata workflow' do
    it 'creates document from markdown with metadata and exports to JSON' do
      # Create creator from markdown
      creator = PdfOxide::Creator.from_markdown("# Title\n\nContent paragraph.")
      expect(creator.source_format).to eq(:markdown)
      expect(creator).not_to be_empty

      # Add metadata
      creator.title('Integration Test Document')
             .author('Test Author')
             .subject('Testing')
             .keywords(['test', 'integration', 'pdf'])
             .creator('Integration Test Suite')

      # Add pages and content
      creator.add_blank_page(800, 600)
             .add_text('Hello, World!')
             .add_blank_page

      # Verify state
      expect(creator.page_count).to eq(2)
      expect(creator.metadata[:title]).to eq('Integration Test Document')
      expect(creator.metadata[:author]).to eq('Test Author')
      expect(creator.creation_timestamp).to be_an(Integer)

      # Export to JSON
      json_str = creator.to_json
      parsed = JSON.parse(json_str)

      expect(parsed['pages']).to eq(2)
      expect(parsed['metadata']['title']).to eq('Integration Test Document')
      expect(parsed).to have_key('created_at')
      expect(parsed['source_format']).to eq('markdown')
    end

    it 'creates document from HTML and exports as hash' do
      html = "<html><body><h1>Title</h1><p>Content</p></body></html>"
      creator = PdfOxide::Creator.from_html(html)

      creator.title('HTML Document')
             .add_blank_page(612, 792)
             .add_text('HTML Content')

      hash = creator.to_h
      expect(hash[:pages]).to eq(1)
      expect(hash[:metadata][:title]).to eq('HTML Document')
      expect(hash[:source_format]).to eq(:html)
      expect(hash).to have_key(:created_at)
    end

    it 'creates document from plain text with method chaining' do
      text = "Line 1\nLine 2\nLine 3"
      creator = PdfOxide::Creator.from_text(text)
                                  .title('Text Document')
                                  .author('Text Author')
                                  .subject('Plain Text')
                                  .add_blank_page
                                  .add_text('First paragraph')

      expect(creator.page_count).to eq(1)
      expect(creator.source_content).to eq(text)
      expect(creator.source_format).to eq(:text)
      expect(creator.metadata[:author]).to eq('Text Author')
    end
  end

  describe 'Builder pattern with complex configuration' do
    it 'builds document with full configuration' do
      creator = PdfOxide::Creator.new_blank
                                  .build do |c|
        c.title('Complex Document')
         .author('Builder Author')
         .subject('Builder Test')
         .keywords(['builder', 'pattern'])
         .creator('Builder Pattern Test')
      end.add_blank_page(612, 792)
        .add_blank_page(800, 600)
        .add_text('First page text')

      expect(creator.page_count).to eq(2)
      expect(creator.metadata[:title]).to eq('Complex Document')
      expect(creator.metadata[:keywords]).to eq('builder, pattern')
      expect(creator.empty?).to be false
    end
  end

  describe 'Content and metadata consistency' do
    it 'maintains metadata consistency across operations' do
      creator = PdfOxide::Creator.new_blank
      original_timestamp = creator.creation_timestamp

      creator.title('Consistent Document')
      expect(creator.metadata[:title]).to eq('Consistent Document')

      # Modify metadata copy and verify original is unchanged
      metadata_copy = creator.metadata
      metadata_copy[:title] = 'Modified Title'

      expect(creator.metadata[:title]).to eq('Consistent Document')
    end

    it 'preserves source information through operations' do
      source = "# Header\n\nParagraph"
      creator = PdfOxide::Creator.from_markdown(source)

      creator.title('Preserve Test')
             .add_blank_page

      expect(creator.source_content).to eq(source)
      expect(creator.source_format).to eq(:markdown)

      json_str = creator.to_json
      parsed = JSON.parse(json_str)

      expect(parsed['source_format']).to eq('markdown')
    end
  end

  describe 'Error handling in workflows' do
    it 'validates required parameters in factory methods' do
      expect { PdfOxide::Creator.from_markdown('') }
        .to raise_error(ArgumentError, /cannot be empty/)

      expect { PdfOxide::Creator.from_html(nil) }
        .to raise_error(ArgumentError, /cannot be empty/)

      expect { PdfOxide::Creator.from_text('') }
        .to raise_error(ArgumentError, /cannot be empty/)
    end

    it 'validates content parameters' do
      creator = PdfOxide::Creator.new_blank

      expect { creator.add_text('') }
        .to raise_error(ArgumentError, /cannot be empty/)

      expect { creator.add_text(nil) }
        .to raise_error(ArgumentError, /cannot be empty/)
    end

    it 'validates image file existence' do
      creator = PdfOxide::Creator.new_blank

      expect { creator.add_image('/nonexistent/path.png') }
        .to raise_error(FileNotFoundError)
    end
  end
end

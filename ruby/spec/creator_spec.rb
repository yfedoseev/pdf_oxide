# frozen_string_literal: true

require 'spec_helper'

RSpec.describe PdfOxide::Creator do
  describe '#new_blank' do
    it 'creates a blank PDF creator' do
      creator = described_class.new_blank
      expect(creator).to be_a(described_class)
      expect(creator.page_count).to eq(0)
      expect(creator.empty?).to be true
    end
  end

  describe '.from_markdown' do
    it 'creates creator from Markdown content' do
      markdown = "# Title\n\nSome paragraph."
      creator = described_class.from_markdown(markdown)

      expect(creator).to be_a(described_class)
      expect(creator.source_format).to eq(:markdown)
      expect(creator.source_content).to eq(markdown)
    end

    it 'validates Markdown content is not empty' do
      expect { described_class.from_markdown('') }
        .to raise_error(ArgumentError, /cannot be empty/)

      expect { described_class.from_markdown(nil) }
        .to raise_error(ArgumentError, /cannot be empty/)
    end

    it 'sets markdown creator metadata' do
      creator = described_class.from_markdown("# Title")
      expect(creator.metadata[:creator]).to include('Markdown')
    end
  end

  describe '.from_html' do
    it 'creates creator from HTML content' do
      html = "<html><body><h1>Title</h1></body></html>"
      creator = described_class.from_html(html)

      expect(creator).to be_a(described_class)
      expect(creator.source_format).to eq(:html)
      expect(creator.source_content).to eq(html)
    end

    it 'validates HTML content is not empty' do
      expect { described_class.from_html('') }
        .to raise_error(ArgumentError, /cannot be empty/)

      expect { described_class.from_html(nil) }
        .to raise_error(ArgumentError, /cannot be empty/)
    end

    it 'sets HTML creator metadata' do
      creator = described_class.from_html("<h1>Test</h1>")
      expect(creator.metadata[:creator]).to include('HTML')
    end
  end

  describe '.from_text' do
    it 'creates creator from plain text' do
      text = "Line 1\nLine 2\nLine 3"
      creator = described_class.from_text(text)

      expect(creator).to be_a(described_class)
      expect(creator.source_format).to eq(:text)
      expect(creator.source_content).to eq(text)
    end

    it 'validates text content is not empty' do
      expect { described_class.from_text('') }
        .to raise_error(ArgumentError, /cannot be empty/)

      expect { described_class.from_text(nil) }
        .to raise_error(ArgumentError, /cannot be empty/)
    end

    it 'sets text creator metadata' do
      creator = described_class.from_text("Some text")
      expect(creator.metadata[:creator]).to include('Text')
    end
  end

  describe '#add_page_from_template' do
    let(:temp_file) { Tempfile.new(['test', '.pdf']) }

    after do
      temp_file.close
      temp_file.unlink
    end

    it 'adds page from template' do
      creator = described_class.new_blank
      result = creator.add_page_from_template(temp_file.path)

      expect(result).to equal(creator) # Method chaining
      expect(creator.page_count).to eq(1)
    end

    it 'raises error for missing template file' do
      creator = described_class.new_blank
      expect { creator.add_page_from_template('/nonexistent/file.pdf') }
        .to raise_error(FileNotFoundError)
    end
  end

  describe '#add_blank_page' do
    it 'adds blank page with default size' do
      creator = described_class.new_blank
      creator.add_blank_page

      expect(creator.page_count).to eq(1)
    end

    it 'adds blank page with custom size' do
      creator = described_class.new_blank
      creator.add_blank_page(800, 600)

      expect(creator.page_count).to eq(1)
    end

    it 'supports method chaining' do
      creator = described_class.new_blank
      result = creator.add_blank_page(800, 600)

      expect(result).to equal(creator)
    end
  end

  describe '#add_page_from_document' do
    let(:temp_file) { Tempfile.new(['test', '.pdf']) }

    after do
      temp_file.close
      temp_file.unlink
    end

    it 'adds page from document' do
      creator = described_class.new_blank
      creator.add_page_from_document(temp_file.path)

      expect(creator.page_count).to eq(1)
    end

    it 'raises error for missing document' do
      creator = described_class.new_blank
      expect { creator.add_page_from_document('/nonexistent/file.pdf') }
        .to raise_error(FileNotFoundError)
    end
  end

  describe '#title' do
    it 'sets document title' do
      creator = described_class.new_blank
      creator.title('My Document')

      expect(creator.metadata[:title]).to eq('My Document')
    end

    it 'supports method chaining' do
      creator = described_class.new_blank
      result = creator.title('Test')

      expect(result).to equal(creator)
    end
  end

  describe '#author' do
    it 'sets document author' do
      creator = described_class.new_blank
      creator.author('John Doe')

      expect(creator.metadata[:author]).to eq('John Doe')
    end

    it 'supports method chaining' do
      creator = described_class.new_blank
      result = creator.author('Test Author')

      expect(result).to equal(creator)
    end
  end

  describe '#subject' do
    it 'sets document subject' do
      creator = described_class.new_blank
      creator.subject('Test Subject')

      expect(creator.metadata[:subject]).to eq('Test Subject')
    end

    it 'supports method chaining' do
      creator = described_class.new_blank
      result = creator.subject('Test')

      expect(result).to equal(creator)
    end
  end

  describe '#keywords' do
    it 'sets keywords as string' do
      creator = described_class.new_blank
      creator.keywords('ruby, pdf, test')

      expect(creator.metadata[:keywords]).to eq('ruby, pdf, test')
    end

    it 'sets keywords as array' do
      creator = described_class.new_blank
      creator.keywords(['ruby', 'pdf', 'test'])

      expect(creator.metadata[:keywords]).to eq('ruby, pdf, test')
    end

    it 'supports method chaining' do
      creator = described_class.new_blank
      result = creator.keywords(['test'])

      expect(result).to equal(creator)
    end
  end

  describe '#creator' do
    it 'sets creator application' do
      creator = described_class.new_blank
      creator.creator('MyApp 1.0')

      expect(creator.metadata[:creator]).to eq('MyApp 1.0')
    end

    it 'supports method chaining' do
      creator = described_class.new_blank
      result = creator.creator('Test')

      expect(result).to equal(creator)
    end
  end

  describe '#add_text' do
    it 'adds text content' do
      creator = described_class.new_blank
      creator.add_text('Hello, World!')

      expect(creator.to_h[:pages]).to eq(0) # Still 0 pages until saved
    end

    it 'validates text is not empty' do
      creator = described_class.new_blank
      expect { creator.add_text('') }
        .to raise_error(ArgumentError, /cannot be empty/)

      expect { creator.add_text(nil) }
        .to raise_error(ArgumentError, /cannot be empty/)
    end

    it 'accepts options' do
      creator = described_class.new_blank
      creator.add_text('Styled Text', font_size: 14, color: 'black')

      expect(creator).to respond_to(:add_text)
    end

    it 'supports method chaining' do
      creator = described_class.new_blank
      result = creator.add_text('Test')

      expect(result).to equal(creator)
    end
  end

  describe '#add_image' do
    let(:temp_image) { Tempfile.new(['test', '.png']) }

    before do
      # Create a simple PNG file
      temp_image.write("\x89PNG\r\n\x1a\n")
      temp_image.flush
    end

    after do
      temp_image.close
      temp_image.unlink
    end

    it 'adds image content' do
      creator = described_class.new_blank
      creator.add_image(temp_image.path)

      expect(creator).to respond_to(:add_image)
    end

    it 'validates image path is not empty' do
      creator = described_class.new_blank
      expect { creator.add_image('') }
        .to raise_error(ArgumentError, /cannot be empty/)

      expect { creator.add_image(nil) }
        .to raise_error(ArgumentError, /cannot be empty/)
    end

    it 'validates image file exists' do
      creator = described_class.new_blank
      expect { creator.add_image('/nonexistent/image.png') }
        .to raise_error(FileNotFoundError)
    end

    it 'accepts options' do
      creator = described_class.new_blank
      creator.add_image(temp_image.path, width: 200, height: 200)

      expect(creator).to respond_to(:add_image)
    end

    it 'supports method chaining' do
      creator = described_class.new_blank
      result = creator.add_image(temp_image.path)

      expect(result).to equal(creator)
    end
  end

  describe '#metadata' do
    it 'returns copy of metadata' do
      creator = described_class.new_blank
      creator.title('Test')

      metadata = creator.metadata
      expect(metadata).to be_a(Hash)
      expect(metadata[:title]).to eq('Test')
    end

    it 'returns independent copy' do
      creator = described_class.new_blank
      creator.title('Original')

      metadata = creator.metadata
      metadata[:title] = 'Modified'

      # Original should not be affected
      expect(creator.metadata[:title]).to eq('Original')
    end
  end

  describe '#source_format' do
    it 'returns nil for blank creators' do
      creator = described_class.new_blank
      expect(creator.source_format).to be_nil
    end

    it 'returns format for markdown' do
      creator = described_class.from_markdown("# Test")
      expect(creator.source_format).to eq(:markdown)
    end

    it 'returns format for HTML' do
      creator = described_class.from_html("<h1>Test</h1>")
      expect(creator.source_format).to eq(:html)
    end

    it 'returns format for text' do
      creator = described_class.from_text("Test")
      expect(creator.source_format).to eq(:text)
    end
  end

  describe '#source_content' do
    it 'returns content for markdown' do
      content = "# Test"
      creator = described_class.from_markdown(content)
      expect(creator.source_content).to eq(content)
    end

    it 'returns content for HTML' do
      content = "<h1>Test</h1>"
      creator = described_class.from_html(content)
      expect(creator.source_content).to eq(content)
    end

    it 'returns content for text' do
      content = "Test text"
      creator = described_class.from_text(content)
      expect(creator.source_content).to eq(content)
    end
  end

  describe '#empty?' do
    it 'returns true for new blank creator' do
      creator = described_class.new_blank
      expect(creator.empty?).to be true
    end

    it 'returns false after adding page' do
      creator = described_class.new_blank
      creator.add_blank_page

      expect(creator.empty?).to be false
    end

    it 'returns false after adding text' do
      creator = described_class.new_blank
      creator.add_text('Test')

      expect(creator.empty?).to be false
    end
  end

  describe '#to_h' do
    it 'returns hash representation' do
      creator = described_class.new_blank
      creator.title('Test')
      creator.add_blank_page

      hash = creator.to_h
      expect(hash).to be_a(Hash)
      expect(hash[:pages]).to eq(1)
      expect(hash[:metadata]).to be_a(Hash)
      expect(hash).to have_key(:created_at)
    end

    it 'includes source format' do
      creator = described_class.from_markdown("# Test")
      hash = creator.to_h

      expect(hash[:source_format]).to eq(:markdown)
    end
  end

  describe '#to_json' do
    it 'returns JSON string' do
      creator = described_class.new_blank
      creator.title('Test')

      json_str = creator.to_json
      parsed = JSON.parse(json_str)

      expect(parsed['metadata']['title']).to eq('Test')
    end

    it 'includes all fields' do
      creator = described_class.from_text("Test text")
      json_str = creator.to_json
      parsed = JSON.parse(json_str)

      expect(parsed).to have_key('pages')
      expect(parsed).to have_key('metadata')
      expect(parsed).to have_key('source_format')
      expect(parsed).to have_key('created_at')
    end
  end

  describe '#build' do
    it 'accepts block for configuration' do
      creator = described_class.new_blank.build do |c|
        c.title('Test')
        c.author('Author')
      end

      expect(creator.metadata[:title]).to eq('Test')
      expect(creator.metadata[:author]).to eq('Author')
    end

    it 'supports method chaining' do
      creator = described_class.new_blank
      result = creator.build { |c| c.title('Test') }

      expect(result).to equal(creator)
    end
  end

  describe '#to_s' do
    it 'returns string representation' do
      creator = described_class.new_blank
      creator.add_blank_page

      str = creator.to_s
      expect(str).to include('Creator')
      expect(str).to include('1')
    end

    it 'shows page count' do
      creator = described_class.new_blank
      creator.add_blank_page
      creator.add_blank_page

      expect(creator.to_s).to include('2')
    end
  end

  describe '#page_count' do
    it 'returns 0 for new creator' do
      creator = described_class.new_blank
      expect(creator.page_count).to eq(0)
    end

    it 'increments with added pages' do
      creator = described_class.new_blank
      creator.add_blank_page
      expect(creator.page_count).to eq(1)

      creator.add_blank_page
      expect(creator.page_count).to eq(2)
    end
  end

  describe '#creation_timestamp' do
    it 'returns integer timestamp' do
      creator = described_class.new_blank
      expect(creator.creation_timestamp).to be_an(Integer)
    end

    it 'returns current time timestamp' do
      before = Time.now.to_i
      creator = described_class.new_blank
      after = Time.now.to_i

      expect(creator.creation_timestamp).to be >= before
      expect(creator.creation_timestamp).to be <= after
    end
  end

  describe 'method chaining' do
    it 'supports full builder pattern' do
      creator = described_class.new_blank
        .title('My Document')
        .author('John Doe')
        .subject('Test Subject')
        .keywords(['test', 'document'])
        .creator('TestApp')
        .add_blank_page
        .add_blank_page

      expect(creator.page_count).to eq(2)
      expect(creator.metadata[:title]).to eq('My Document')
      expect(creator.metadata[:author]).to eq('John Doe')
    end
  end
end

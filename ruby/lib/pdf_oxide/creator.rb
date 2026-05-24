# frozen_string_literal: true

require 'json'

module PdfOxide
  # Interface for creating and modifying PDF documents
  #
  # The Creator class provides a fluent API for building PDF documents from scratch
  # or from existing content (Markdown, HTML, plain text). It supports:
  # - Creating blank documents
  # - Adding pages with custom dimensions
  # - Setting document metadata (title, author, subject, keywords)
  # - Adding content (text, images)
  # - Exporting to JSON and hash formats
  # - Full method chaining for fluent API design
  #
  # @example Creating a PDF from Markdown
  #   creator = PdfOxide::Creator.from_markdown("# Title\n\nContent")
  #   creator.title("My PDF")
  #          .author("John Doe")
  #          .add_blank_page
  #          .add_text("Hello, World!")
  #
  # @example Creating a blank PDF with metadata
  #   creator = PdfOxide::Creator.new_blank
  #                               .title("Document")
  #                               .author("Author Name")
  #                               .keywords(['test', 'document'])
  #
  class Creator
    def initialize
      @pages = []
      @metadata = {}
      @content = []
      @creation_timestamp = Time.now.to_i
    end

    # Create a new blank PDF
    #
    # Creates an empty Creator instance ready for pages and content to be added.
    # Use this when you want to build a PDF from scratch without source content.
    #
    # @return [Creator] A new blank creator instance
    # @example
    #   creator = PdfOxide::Creator.new_blank
    #   creator.add_blank_page.add_text("Hello")
    def self.new_blank
      new
    end

    # Create from Markdown content
    #
    # Creates a Creator from Markdown source content. The markdown content is stored
    # as the source and can be accessed via {#source_content}. The source format
    # is tracked and can be queried via {#source_format}.
    #
    # @param markdown [String] Markdown content (must not be empty)
    # @return [Creator] A new creator with markdown source
    # @raise [ArgumentError] if markdown is nil or empty
    # @example
    #   md = "# Title\n\nParagraph with **bold** text."
    #   creator = PdfOxide::Creator.from_markdown(md)
    #   creator.source_format # => :markdown
    def self.from_markdown(markdown)
      raise ArgumentError, 'Markdown content cannot be empty' if markdown.nil? || markdown.empty?

      creator = new
      creator.instance_variable_set(:@source_format, :markdown)
      creator.instance_variable_set(:@source_content, markdown)

      # Add metadata about source
      creator.creator('PdfOxide Markdown Converter')

      creator
    end

    # Create from HTML content
    #
    # Creates a Creator from HTML source content. The HTML content is stored
    # and accessible via {#source_content}. The source format is set to :html.
    #
    # @param html [String] HTML content (must not be empty)
    # @return [Creator] A new creator with HTML source
    # @raise [ArgumentError] if html is nil or empty
    # @example
    #   html = "<html><body><h1>Title</h1></body></html>"
    #   creator = PdfOxide::Creator.from_html(html)
    #   creator.source_format # => :html
    def self.from_html(html)
      raise ArgumentError, 'HTML content cannot be empty' if html.nil? || html.empty?

      creator = new
      creator.instance_variable_set(:@source_format, :html)
      creator.instance_variable_set(:@source_content, html)

      # Add metadata about source
      creator.creator('PdfOxide HTML Converter')

      creator
    end

    # Create from plain text
    #
    # Creates a Creator from plain text source content. The text content is stored
    # and accessible via {#source_content}. The source format is set to :text.
    #
    # @param text [String] Plain text content (must not be empty)
    # @return [Creator] A new creator with text source
    # @raise [ArgumentError] if text is nil or empty
    # @example
    #   text = "Line 1\nLine 2\nLine 3"
    #   creator = PdfOxide::Creator.from_text(text)
    #   creator.source_format # => :text
    def self.from_text(text)
      raise ArgumentError, 'Text content cannot be empty' if text.nil? || text.empty?

      creator = new
      creator.instance_variable_set(:@source_format, :text)
      creator.instance_variable_set(:@source_content, text)

      # Add metadata about source
      creator.creator('PdfOxide Text Converter')

      creator
    end

    # Add page from template
    # @param template_path [String] Path to template PDF
    # @param page_index [Integer] Page index to use
    # @return [self]
    def add_page_from_template(template_path, page_index = 0)
      raise FileNotFoundError, "Template not found: #{template_path}" unless File.exist?(template_path)

      @pages << { type: :template, path: template_path, page: page_index }
      self
    end

    # Add blank page
    # @param width [Float] Page width in points
    # @param height [Float] Page height in points
    # @return [self]
    def add_blank_page(width = 612, height = 792)
      @pages << { type: :blank, width: width, height: height }
      self
    end

    # Add page from another document
    # @param doc_path [String] Path to document
    # @param page_index [Integer] Page index to copy
    # @return [self]
    def add_page_from_document(doc_path, page_index = 0)
      raise FileNotFoundError, "Document not found: #{doc_path}" unless File.exist?(doc_path)

      @pages << { type: :document, path: doc_path, page: page_index }
      self
    end

    # Set document title
    # @param title [String] Document title
    # @return [self]
    def title(title)
      @metadata[:title] = title
      self
    end

    # Set document author
    # @param author [String] Author name
    # @return [self]
    def author(author)
      @metadata[:author] = author
      self
    end

    # Set document subject
    # @param subject [String] Subject text
    # @return [self]
    def subject(subject)
      @metadata[:subject] = subject
      self
    end

    # Set document keywords
    # @param keywords [String, Array] Keywords
    # @return [self]
    def keywords(keywords)
      @metadata[:keywords] = keywords.is_a?(Array) ? keywords.join(', ') : keywords
      self
    end

    # Set document creator application
    # @param creator [String] Creator application
    # @return [self]
    def creator(creator)
      @metadata[:creator] = creator
      self
    end

    # Save PDF to file.
    #
    # Materialises the queued source (markdown / html / text) via the
    # cdylib's `pdf_from_*` factory, then persists with `pdf_save`.
    # Blank creators (no source content, no merge inputs) raise
    # ArgumentError — the C ABI has no zero-page PDF constructor.
    # @param output_path [String] Output file path
    # @return [Boolean] true on success
    # @raise [ArgumentError] if output_path is blank or no source content
    # @raise [PdfOxide::Error] if the cdylib reports a non-zero error code
    def save(output_path)
      raise ArgumentError, 'Output path cannot be empty' if output_path.nil? || output_path.empty?

      handle = build_pdf_handle
      begin
        error_ptr = ::FFI::MemoryPointer.new(:int32)
        rc = FFI::Bindings.pdf_save(handle, output_path, error_ptr)
        error_code = error_ptr.read_int32
        # pdf_save returns 0 on success, -1 on failure.
        if rc != 0 || error_code != 0
          raise FFI::ErrorHandler.create_error(error_code, 'creator_save', path: output_path)
        end

        true
      ensure
        FFI::Bindings.pdf_free(handle) if handle && !handle.null?
      end
    end

    # Save and return bytes.
    # @return [String] PDF file bytes (binary-encoded)
    # @raise [ArgumentError] if no source content
    # @raise [PdfOxide::Error] if the cdylib reports a non-zero error code
    def to_bytes
      handle = build_pdf_handle
      begin
        len_ptr = ::FFI::MemoryPointer.new(:int32)
        error_ptr = ::FFI::MemoryPointer.new(:int32)
        buf_ptr = FFI::Bindings.pdf_save_to_bytes(handle, len_ptr, error_ptr)
        error_code = error_ptr.read_int32
        if error_code != 0 || buf_ptr.nil? || buf_ptr.null?
          raise FFI::ErrorHandler.create_error(error_code, 'creator_to_bytes')
        end

        len = len_ptr.read_int32
        bytes = buf_ptr.read_string(len)
        FFI::Bindings.free_bytes(buf_ptr)
        bytes.force_encoding(Encoding::BINARY)
      ensure
        FFI::Bindings.pdf_free(handle) if handle && !handle.null?
      end
    end

    private

    # Build a `Pdf` handle from queued source content.
    # @return [FFI::Pointer] PDF handle (caller responsible for pdf_free)
    # @raise [ArgumentError, PdfOxide::Error]
    def build_pdf_handle
      raise ArgumentError, 'Creator has no source content (markdown/html/text)' \
        if @source_format.nil? || @source_content.nil? || @source_content.empty?

      error_ptr = ::FFI::MemoryPointer.new(:int32)
      handle =
        case @source_format
        when :markdown then FFI::Bindings.pdf_from_markdown(@source_content, error_ptr)
        when :html     then FFI::Bindings.pdf_from_html(@source_content, error_ptr)
        when :text     then FFI::Bindings.pdf_from_text(@source_content, error_ptr)
        else
          raise ArgumentError, "Unsupported source format: #{@source_format}"
        end

      error_code = error_ptr.read_int32
      if error_code != 0 || handle.nil? || handle.null?
        raise FFI::ErrorHandler.create_error(error_code, 'creator_build', format: @source_format)
      end

      handle
    end

    public

    # Merge another PDF into this one
    # @param doc_path [String] Path to PDF to merge
    # @return [self]
    def merge(doc_path)
      raise FileNotFoundError, "Document not found: #{doc_path}" unless File.exist?(doc_path)

      @pages << { type: :merge, path: doc_path }
      self
    end

    # Get number of pages
    # @return [Integer]
    def page_count
      @pages.length
    end

    # Builder method for chaining
    # @yield Block for configuration
    # @return [Creator]
    def build
      yield self if block_given?
      self
    end

    # Convert to string representation
    # @return [String]
    def to_s
      "Creator(#{page_count} pages)"
    end

    # Get source format if created from template
    #
    # Returns the source format for creators built from templates. For creators
    # built from Markdown, HTML, or plain text, this returns the corresponding
    # format. For blank creators, returns nil.
    #
    # @return [Symbol, nil] Source format (:markdown, :html, :text) or nil
    # @example
    #   PdfOxide::Creator.from_markdown("# Title").source_format # => :markdown
    #   PdfOxide::Creator.new_blank.source_format # => nil
    attr_reader :source_format

    # Get source content
    #
    # Returns the original source content used to create this Creator.
    # This is only populated for creators built from {.from_markdown},
    # {.from_html}, or {.from_text}. Blank creators return nil.
    #
    # @return [String, nil] Original source content, or nil if not applicable
    # @example
    #   md = "# Title"
    #   creator = PdfOxide::Creator.from_markdown(md)
    #   creator.source_content # => "# Title"
    attr_reader :source_content

    # Check if document is empty
    #
    # Returns true if the creator has no pages and no content queued.
    # A newly created blank creator is empty; adding pages or content marks it non-empty.
    #
    # @return [Boolean] true if no pages or content, false otherwise
    # @example
    #   creator = PdfOxide::Creator.new_blank
    #   creator.empty? # => true
    #   creator.add_blank_page
    #   creator.empty? # => false
    def empty?
      @pages.empty? && @content.empty?
    end

    # Get metadata as hash
    #
    # Returns a copy of the document metadata. Modifying the returned hash
    # does not affect the creator's internal metadata.
    #
    # @return [Hash] Independent copy of document metadata
    # @example
    #   creator.title("My PDF").author("John")
    #   metadata = creator.metadata
    #   metadata[:title] # => "My PDF"
    #   metadata[:author] # => "John"
    def metadata
      @metadata.dup
    end

    # Convert creator to hash (JSON compatible)
    #
    # Returns a hash representation of the creator suitable for serialization.
    # The hash includes metadata, page count, source format, and creation timestamp.
    #
    # @param *args Additional arguments passed to Hash#to_h
    # @return [Hash] Hash representation with keys:
    #   - :pages [Integer] Number of pages
    #   - :metadata [Hash] Document metadata copy
    #   - :source_format [Symbol, nil] Source format if applicable
    #   - :created_at [Integer] Unix timestamp of creation
    # @example
    #   creator = PdfOxide::Creator.from_markdown("# Title")
    #   hash = creator.to_h
    #   hash[:pages] # => 0
    #   hash[:metadata][:creator] # => "PdfOxide Markdown Converter"
    def to_h
      {
        pages: page_count,
        metadata: @metadata.dup,
        source_format: @source_format,
        created_at: @creation_timestamp
      }
    end

    # Convert creator to JSON
    #
    # Serializes the creator to a JSON string using {#to_h} for conversion.
    # The resulting JSON contains all creator information in a structured format.
    #
    # @param *args Additional arguments passed to Hash#to_json
    # @return [String] JSON-formatted string representation
    # @example
    #   creator = PdfOxide::Creator.new_blank.title("Test")
    #   json = creator.to_json
    #   JSON.parse(json)['metadata']['title'] # => "Test"
    def to_json(*args)
      to_h.to_json(*args)
    end

    # Add text content to current page
    #
    # Adds text content to the creator. The text is queued for rendering when
    # the PDF is generated. Options allow customization of font, size, and color.
    #
    # @param text [String] Text to add (must not be empty)
    # @param options [Hash] Text styling options
    # @option options [Float] :font_size Font size in points (default: 12)
    # @option options [String] :font Font name (default: 'Helvetica')
    # @option options [String] :color Text color (default: 'black')
    # @option options [Float] :x X coordinate (default: document left margin)
    # @option options [Float] :y Y coordinate (default: current position)
    # @return [self] For method chaining
    # @raise [ArgumentError] if text is nil or empty
    # @example
    #   creator.add_text("Hello, World!")
    #   creator.add_text("Styled text", font_size: 14, color: 'red')
    def add_text(text, options = {})
      raise ArgumentError, 'Text cannot be empty' if text.nil? || text.empty?

      @content << {
        type: :text,
        content: text,
        options: options
      }
      self
    end

    # Add image content to current page
    #
    # Adds an image to the creator. The image file must exist on disk.
    # Options control image dimensions and positioning.
    #
    # @param image_path [String] Path to image file (must exist and not be empty)
    # @param options [Hash] Image placement and sizing options
    # @option options [Float] :width Image width in points
    # @option options [Float] :height Image height in points
    # @option options [Float] :x X coordinate for image placement
    # @option options [Float] :y Y coordinate for image placement
    # @return [self] For method chaining
    # @raise [ArgumentError] if image_path is nil or empty
    # @raise [FileNotFoundError] if image file does not exist
    # @example
    #   creator.add_image('logo.png', width: 200, height: 100)
    def add_image(image_path, options = {})
      raise ArgumentError, 'Image path cannot be empty' if image_path.nil? || image_path.empty?
      raise FileNotFoundError, "Image file not found: #{image_path}" unless File.exist?(image_path)

      @content << {
        type: :image,
        path: image_path,
        options: options
      }
      self
    end

    # Get creation timestamp
    # @return [Integer] Unix timestamp
    attr_reader :creation_timestamp
  end
end

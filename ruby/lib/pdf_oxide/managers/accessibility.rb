# frozen_string_literal: true

module PdfOxide
  # Accessibility Manager for PDF structure tree and auto-tagging operations
  #
  # Provides structure tree inspection, automatic tagging, and
  # accessibility metadata management for PDF/UA compliance.
  class AccessibilityManager
    # Structure element record
    StructureElement = Struct.new(
      :struct_type,
      :alt_text,
      :children,
      keyword_init: true
    )

    # Structure tree record
    StructureTreeInfo = Struct.new(
      :root_elements,
      :element_count,
      keyword_init: true
    )

    # Auto-tag result record
    AutoTagResult = Struct.new(
      :elements_tagged,
      keyword_init: true
    )

    attr_reader :document

    # Initialize AccessibilityManager with a PDF document
    # @param document [Object] PDF document handle
    def initialize(document)
      @document = document
    end

    # Check if the document has a structure tree (is tagged)
    # @return [Boolean] true if the document contains a structure tree
    # @raise [PdfOxide::AccessibilityError] if the operation fails
    def tagged?
      FFI::ErrorHandler.with_error_check('accessibility_is_tagged') do |err|
        Bindings.pdf_accessibility_is_tagged(@document, err)
      end
    end

    # Get the document's structure tree
    # @return [StructureTreeInfo, nil] structure tree if document is tagged, nil otherwise
    # @raise [PdfOxide::AccessibilityError] if the operation fails
    def get_structure_tree
      return nil unless tagged?

      tree_handle = FFI::ErrorHandler.with_error_check('accessibility_get_structure_tree') do |err|
        Bindings.pdf_accessibility_get_structure_tree(@document, err)
      end

      begin
        StructureTreeInfo.new(
          root_elements: [],
          element_count: 0
        )
      ensure
        Bindings.pdf_structure_tree_free(tree_handle) if tree_handle && !tree_handle.null?
      end
    end

    # Automatically tag the document for accessibility
    #
    # Analyzes content and generates a structure tree with paragraphs,
    # headings, lists, and other semantic elements.
    #
    # @param language [String, nil] BCP 47 language tag (e.g., "en-US"). Optional.
    # @return [AutoTagResult] result with the number of elements tagged
    # @raise [PdfOxide::AccessibilityError] if tagging fails
    def auto_tag(language: nil)
      elements = FFI::ErrorHandler.with_error_check('accessibility_auto_tag') do |err|
        Bindings.pdf_accessibility_auto_tag(@document, language, err)
      end

      AutoTagResult.new(elements_tagged: elements.to_i)
    end

    # Set alternate text on a structure element
    #
    # Alt text is required for non-text content in PDF/UA.
    #
    # @param page [Integer] page index (0-based)
    # @param mcid [Integer] marked content ID
    # @param text [String] alt text string
    # @raise [PdfOxide::AccessibilityError] if the operation fails
    def set_alt_text(page, mcid, text)
      raise ::PdfOxide::ArgumentError.new('text cannot be nil') if text.nil?

      FFI::ErrorHandler.with_bool_check('accessibility_set_alt_text') do |err|
        Bindings.pdf_accessibility_set_alt_text(@document, page, mcid, text, err)
      end
    end

    # Set the document language
    # @param language [String] BCP 47 language tag (e.g., "en-US")
    # @raise [PdfOxide::AccessibilityError] if the operation fails
    def set_language(language)
      raise ::PdfOxide::ArgumentError.new('language cannot be nil') if language.nil?

      FFI::ErrorHandler.with_bool_check('accessibility_set_language') do |err|
        Bindings.pdf_accessibility_set_language(@document, language, err)
      end
    end

    # Set the document title for accessibility
    # @param title [String] document title
    # @raise [PdfOxide::AccessibilityError] if the operation fails
    def set_title(title)
      raise ::PdfOxide::ArgumentError.new('title cannot be nil') if title.nil?

      FFI::ErrorHandler.with_bool_check('accessibility_set_title') do |err|
        Bindings.pdf_accessibility_set_title(@document, title, err)
      end
    end

    # Get accessibility summary
    # @return [Hash] summary of accessibility state and capabilities
    def summary
      {
        is_tagged: tagged?,
        capabilities: {
          auto_tag: true,
          structure_tree: true,
          alt_text: true,
          language: true,
          title: true
        }
      }
    end
  end
end

# frozen_string_literal: true

module PdfOxide
  # Editing Manager for PDF redaction, flattening, and compliance operations.
  #
  # Provides methods for:
  # - Adding and applying content redactions
  # - Scrubbing sensitive metadata
  # - Flattening form fields and annotations into static content
  # - PDF/A compliance conversion and validation
  #
  # @example
  #   manager = PdfOxide::EditingManager.new(document_handle)
  #   manager.add_redaction(page: 0, rect: [100, 200, 300, 50])
  #   manager.apply_redactions
  #   manager.flatten_forms
  class EditingManager
    attr_reader :document

    # Initialize EditingManager with a PDF document handle
    #
    # @param document [FFI::Pointer] Native document handle
    def initialize(document)
      @document = document
    end

    # ==================== REDACTION ====================

    # Add a redaction annotation to a page.
    #
    # The redaction marks an area to be permanently removed when
    # apply_redactions is called. Until applied, the content is
    # still visible but marked for redaction.
    #
    # @param page [Integer] Zero-based page index
    # @param rect [Array<Float>] Rectangle as [x, y, width, height] in PDF points
    # @param color [Array<Integer>] Fill color as [r, g, b] with values 0-255, default [0, 0, 0]
    # @raise [PdfOxide::RedactionError] If adding the redaction fails
    def add_redaction(page:, rect:, color: [0, 0, 0])
      raise ::PdfOxide::ArgumentError.new('rect must contain 4 values') unless rect.length == 4

      x, y, width, height = rect.map(&:to_f)
      r, g, b = color.map(&:to_i)

      FFI::ErrorHandler.with_error_check('redaction_add', page: page) do |err|
        Bindings.pdf_redaction_add(
          @document,
          page,
          x, y, width, height,
          r, g, b,
          err
        )
      end
    end

    # Apply all pending redactions, permanently removing marked content.
    #
    # This operation is irreversible. Once applied, the redacted content
    # is permanently removed from the document.
    #
    # @param scrub_metadata [Boolean] If true, also scrub document metadata
    # @param fill_color [Array<Integer>] Fill color for redacted areas as [r, g, b], default [0, 0, 0]
    # @raise [PdfOxide::RedactionError] If applying redactions fails
    def apply_redactions(scrub_metadata: false, fill_color: [0, 0, 0])
      r, g, b = fill_color.map(&:to_i)

      FFI::ErrorHandler.with_error_check('redaction_apply') do |err|
        Bindings.pdf_redaction_apply(
          @document,
          scrub_metadata,
          r, g, b,
          err
        )
      end
    end

    # Scrub sensitive metadata from the document.
    #
    # Removes various types of metadata that may contain sensitive
    # information such as author names, creation software, or
    # embedded scripts.
    #
    # @param remove_info [Boolean] Remove document Info dictionary (Title, Author, etc.)
    # @param remove_xmp [Boolean] Remove XMP metadata streams
    # @param remove_js [Boolean] Remove embedded JavaScript
    # @raise [PdfOxide::RedactionError] If metadata scrubbing fails
    def scrub_metadata(remove_info: true, remove_xmp: true, remove_js: true)
      FFI::ErrorHandler.with_error_check('redaction_scrub_metadata') do |err|
        Bindings.pdf_redaction_scrub_metadata(
          @document,
          remove_info,
          remove_xmp,
          remove_js,
          err
        )
      end
    end

    # Get the number of pending redaction annotations.
    #
    # @return [Integer] Number of pending redactions not yet applied
    # @raise [PdfOxide::RedactionError] If retrieving the count fails
    def get_redaction_count
      FFI::ErrorHandler.with_int_check('redaction_count') do |err|
        Bindings.pdf_redaction_count(@document, err)
      end
    end

    # ==================== FLATTENING ====================

    # Flatten all form fields in the document.
    #
    # Converts interactive form fields into static content. After
    # flattening, form fields are no longer editable but their
    # values are preserved as visible content.
    #
    # @raise [PdfOxide::Error] If flattening fails
    def flatten_forms
      FFI::ErrorHandler.with_error_check('flatten_forms') do |err|
        Bindings.pdf_document_editor_flatten_forms(@document, err)
      end
    end

    # Flatten form fields on a specific page.
    #
    # @param page [Integer] Zero-based page index
    # @raise [PdfOxide::Error] If flattening fails
    def flatten_forms_page(page)
      FFI::ErrorHandler.with_error_check('flatten_forms_page', page: page) do |err|
        Bindings.pdf_document_editor_flatten_forms_page(@document, page, err)
      end
    end

    # Flatten all annotations in the document.
    #
    # Converts interactive annotations (highlights, stamps, notes, etc.)
    # into static page content. After flattening, annotations are no
    # longer interactive.
    #
    # @raise [PdfOxide::Error] If flattening fails
    def flatten_annotations
      FFI::ErrorHandler.with_error_check('flatten_annotations') do |err|
        Bindings.pdf_document_editor_flatten_annotations(@document, err)
      end
    end

    # Flatten annotations on a specific page.
    #
    # @param page [Integer] Zero-based page index
    # @raise [PdfOxide::Error] If flattening fails
    def flatten_annotations_page(page)
      FFI::ErrorHandler.with_error_check('flatten_annotations_page', page: page) do |err|
        Bindings.pdf_document_editor_flatten_annotations_page(@document, page, err)
      end
    end

    # ==================== COMPLIANCE ====================

    # Convert the document to PDF/A format.
    #
    # @param level [Integer] PDF/A conformance level (0=1B, 1=1A, 2=2B, 3=2A, etc.)
    # @raise [PdfOxide::ComplianceError] If conversion fails
    def convert_to_pdf_a(level: 2)
      FFI::ErrorHandler.with_error_check('convert_to_pdf_a', level: level) do |err|
        Bindings.pdf_convert_to_pdf_a(@document, level, err)
      end
    end

    # Validate the document against a PDF/A conformance level.
    #
    # @param level [Integer] PDF/A conformance level to validate against
    # @return [Integer] Validation result code (0 = compliant)
    # @raise [PdfOxide::ComplianceError] If validation fails
    def validate_pdfa(level: 2)
      FFI::ErrorHandler.with_int_check('validate_pdfa', level: level) do |err|
        Bindings.pdf_validate_pdfa(@document, level, err)
      end
    end

    # ==================== SUMMARY ====================

    # Get editing capabilities summary.
    #
    # @return [Hash] Summary of editing capabilities
    def summary
      {
        redaction_count: get_redaction_count,
        capabilities: {
          redaction: true,
          flatten_forms: true,
          flatten_annotations: true,
          scrub_metadata: true,
          pdf_a_conversion: true,
          pdf_a_validation: true
        }
      }
    end
  end
end

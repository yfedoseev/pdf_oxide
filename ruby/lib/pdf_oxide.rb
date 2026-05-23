# frozen_string_literal: true

require 'ffi'
require_relative 'pdf_oxide/version'
require_relative 'pdf_oxide/errors'
require_relative 'pdf_oxide/ffi/library'
require_relative 'pdf_oxide/ffi/bindings'
require_relative 'pdf_oxide/ffi/types'
require_relative 'pdf_oxide/ffi/string_marshaller'
require_relative 'pdf_oxide/ffi/handle_manager'
require_relative 'pdf_oxide/ffi/error_handler'

# Data types
require_relative 'pdf_oxide/types/bounding_box'
require_relative 'pdf_oxide/types/search_result'
require_relative 'pdf_oxide/types/page_dimensions'
require_relative 'pdf_oxide/types/render_options'
require_relative 'pdf_oxide/types/annotation'
require_relative 'pdf_oxide/types/font_info'
require_relative 'pdf_oxide/types/image_info'
require_relative 'pdf_oxide/types/form_field'
require_relative 'pdf_oxide/types/search_options'
require_relative 'pdf_oxide/types/conversion_options'
require_relative 'pdf_oxide/types/ocr_config'
require_relative 'pdf_oxide/types/signature'
require_relative 'pdf_oxide/types/signing_credentials'
require_relative 'pdf_oxide/types/certificate'

# Managers
require_relative 'pdf_oxide/managers/base'
require_relative 'pdf_oxide/managers/search'
require_relative 'pdf_oxide/managers/rendering'
require_relative 'pdf_oxide/managers/annotation'
require_relative 'pdf_oxide/managers/form'
require_relative 'pdf_oxide/managers/page'
require_relative 'pdf_oxide/managers/metadata'
require_relative 'pdf_oxide/managers/outline'
require_relative 'pdf_oxide/managers/layer'
require_relative 'pdf_oxide/managers/cache'
require_relative 'pdf_oxide/managers/extraction'
require_relative 'pdf_oxide/managers/ocr'
require_relative 'pdf_oxide/managers/compliance'
require_relative 'pdf_oxide/managers/signature'
require_relative 'pdf_oxide/managers/barcode'
require_relative 'pdf_oxide/managers/analysis'

# Phase 2 repair: managers that were present on disk but not wired in.
require_relative 'pdf_oxide/managers/accessibility'
require_relative 'pdf_oxide/managers/certificate'
require_relative 'pdf_oxide/managers/document'
require_relative 'pdf_oxide/managers/editing'
require_relative 'pdf_oxide/managers/enterprise'
require_relative 'pdf_oxide/managers/extraction_strategy'
require_relative 'pdf_oxide/managers/optimization'
require_relative 'pdf_oxide/managers/signature_manager'
require_relative 'pdf_oxide/managers/xfa'

# Main entry points
require_relative 'pdf_oxide/document'
require_relative 'pdf_oxide/creator'

# Phase 3 (v0.3.50–v0.3.54 feature surface) — v0.3.55 Ruby workstream.
require_relative 'pdf_oxide/extract_reason'
require_relative 'pdf_oxide/auto_extract_result'
require_relative 'pdf_oxide/auto_extractor'
require_relative 'pdf_oxide/office_converter'
require_relative 'pdf_oxide/redaction_manager'
require_relative 'pdf_oxide/pades_signer'
require_relative 'pdf_oxide/models'

module PdfOxide
  class << self
    # Open a PDF document
    # @param path [String] Path to PDF file
    # @return [Document] PDF document instance
    def open(path, &block)
      Document.open(path, &block)
    end

    # Create a new PDF document
    # @return [Creator] PDF creator instance
    def create
      Creator.new
    end

    # Get library version
    # @return [String] Version string
    def version
      VERSION
    end
  end
end

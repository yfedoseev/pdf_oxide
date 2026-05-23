# frozen_string_literal: true

require_relative 'base'
require 'json'

module PdfOxide
  module Managers
    # Manager for PDF annotation operations
    # Provides methods to add, list, delete, and manage annotations (comments, highlights, etc.)
    class Annotation < Base
      # ANNOTATION TYPES
      ANNOTATION_TYPE_TEXT = 0
      ANNOTATION_TYPE_HIGHLIGHT = 1
      ANNOTATION_TYPE_UNDERLINE = 2
      ANNOTATION_TYPE_STRIKEOUT = 3
      ANNOTATION_TYPE_SQUIGGLY = 4
      ANNOTATION_TYPE_LINK = 5
      ANNOTATION_TYPE_POPUP = 6

      ANNOTATION_TYPES = {
        text: ANNOTATION_TYPE_TEXT,
        highlight: ANNOTATION_TYPE_HIGHLIGHT,
        underline: ANNOTATION_TYPE_UNDERLINE,
        strikeout: ANNOTATION_TYPE_STRIKEOUT,
        squiggly: ANNOTATION_TYPE_SQUIGGLY,
        link: ANNOTATION_TYPE_LINK,
        popup: ANNOTATION_TYPE_POPUP
      }.freeze

      TYPE_NAMES = ANNOTATION_TYPES.invert.freeze

      # Get count of annotations on page
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Integer] Number of annotations
      def annotation_count(page_index)
        check_document!
        validate_page_index!(page_index)

        with_error_check('annotation_count', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_get_annotation_count(@document.handle, page_index, error_ptr)
        end
      end

      # Check if page has annotations
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Boolean] Whether page has annotations
      def has_annotations?(page_index)
        check_document!
        validate_page_index!(page_index)
        annotation_count(page_index) > 0
      end

      # List all annotations on page
      # @param page_index [Integer] Page index (0-indexed)
      # @return [Array<Types::Annotation>] Array of annotations
      def list_annotations(page_index)
        check_document!
        validate_page_index!(page_index)

        annots_handle = with_error_check('list_annotations', page: page_index) do |error_ptr|
          FFI::Bindings.pdf_document_get_annotations(@document.handle, page_index, error_ptr)
        end

        parse_annotation_list(annots_handle, page_index)
      end

      # Get specific annotation
      # @param page_index [Integer] Page index (0-indexed)
      # @param annotation_index [Integer] Annotation index on page
      # @return [Types::Annotation] Annotation object
      def get_annotation(page_index, annotation_index)
        check_document!
        validate_page_index!(page_index)
        raise ::PdfOxide::ArgumentError, 'Annotation index must be >= 0' if annotation_index < 0

        annotations = list_annotations(page_index)
        raise ::PdfOxide::ArgumentError, "Annotation index #{annotation_index} out of range" if annotation_index >= annotations.count

        annotations[annotation_index]
      end

      # Add highlight annotation
      # @param page_index [Integer] Page index (0-indexed)
      # @param x [Float] Left coordinate
      # @param y [Float] Top coordinate
      # @param width [Float] Width
      # @param height [Float] Height
      # @param options [Hash] Additional options
      # @return [Boolean] Whether operation succeeded
      def add_highlight(page_index, x, y, width, height, options = {})
        check_document!
        validate_page_index!(page_index)

        with_error_check('add_highlight', page: page_index, rect: { x: x, y: y, width: width, height: height }) do |error_ptr|
          FFI::Bindings.pdf_document_add_highlight(
            @document.handle,
            page_index,
            x.to_f,
            y.to_f,
            width.to_f,
            height.to_f,
            error_ptr
          )
        end
        true
      end

      # Add underline annotation
      # @param page_index [Integer] Page index (0-indexed)
      # @param x [Float] Left coordinate
      # @param y [Float] Top coordinate
      # @param width [Float] Width
      # @param height [Float] Height
      # @param options [Hash] Additional options
      # @return [Boolean] Whether operation succeeded
      def add_underline(page_index, x, y, width, height, options = {})
        check_document!
        validate_page_index!(page_index)

        with_error_check('add_underline', page: page_index, rect: { x: x, y: y, width: width, height: height }) do |error_ptr|
          FFI::Bindings.pdf_document_add_underline(
            @document.handle,
            page_index,
            x.to_f,
            y.to_f,
            width.to_f,
            height.to_f,
            error_ptr
          )
        end
        true
      end

      # Add strikeout annotation
      # @param page_index [Integer] Page index (0-indexed)
      # @param x [Float] Left coordinate
      # @param y [Float] Top coordinate
      # @param width [Float] Width
      # @param height [Float] Height
      # @param options [Hash] Additional options
      # @return [Boolean] Whether operation succeeded
      def add_strikeout(page_index, x, y, width, height, options = {})
        check_document!
        validate_page_index!(page_index)

        with_error_check('add_strikeout', page: page_index, rect: { x: x, y: y, width: width, height: height }) do |error_ptr|
          FFI::Bindings.pdf_document_add_strikeout(
            @document.handle,
            page_index,
            x.to_f,
            y.to_f,
            width.to_f,
            height.to_f,
            error_ptr
          )
        end
        true
      end

      # Add text/comment annotation
      # @param page_index [Integer] Page index (0-indexed)
      # @param x [Float] Left coordinate
      # @param y [Float] Top coordinate
      # @param text [String] Comment text
      # @param options [Hash] Additional options (author, subject, etc.)
      # @return [Boolean] Whether operation succeeded
      def add_comment(page_index, x, y, text, options = {})
        check_document!
        validate_page_index!(page_index)

        text_utf8 = FFI::StringMarshaller.to_utf8(text)
        subject_utf8 = FFI::StringMarshaller.to_utf8(options.fetch(:subject, ''))

        with_error_check('add_comment', page: page_index, text: text) do |error_ptr|
          FFI::Bindings.pdf_document_add_text_annotation(
            @document.handle,
            page_index,
            x.to_f,
            y.to_f,
            text_utf8,
            subject_utf8,
            error_ptr
          )
        end
        true
      end

      # Delete annotation
      # @param page_index [Integer] Page index (0-indexed)
      # @param annotation_index [Integer] Annotation index on page
      # @return [Boolean] Whether operation succeeded
      def delete_annotation(page_index, annotation_index)
        check_document!
        validate_page_index!(page_index)
        raise ::PdfOxide::ArgumentError, 'Annotation index must be >= 0' if annotation_index < 0

        with_error_check('delete_annotation', page: page_index, annotation: annotation_index) do |error_ptr|
          FFI::Bindings.pdf_document_delete_annotation(@document.handle, page_index, annotation_index, error_ptr)
        end
        true
      end

      # Flatten annotations on page or all pages
      # @param page_index [Integer, nil] Page index or nil for all pages
      # @return [Boolean] Whether operation succeeded
      def flatten_annotations(page_index = nil)
        check_document!
        validate_page_index!(page_index) if page_index

        with_error_check('flatten_annotations', page: page_index) do |error_ptr|
          if page_index.nil?
            # Flatten all pages
            (0...@document.page_count).each do |i|
              FFI::Bindings.pdf_document_flatten_annotations(@document.handle, i, error_ptr)
            end
            true
          else
            FFI::Bindings.pdf_document_flatten_annotations(@document.handle, page_index, error_ptr)
          end
        end
        true
      end

      # Get all annotations in document
      # @return [Array<Types::Annotation>] All annotations
      def get_all_annotations
        check_document!
        all_annots = []
        (0...@document.page_count).each do |page_idx|
          all_annots.concat(list_annotations(page_idx))
        end
        all_annots
      end

      # Get annotations by type
      # @param annotation_type [Symbol, Integer] Type of annotation
      # @return [Array<Types::Annotation>] Annotations of specified type
      def annotations_by_type(annotation_type)
        check_document!

        type_int = annotation_type.is_a?(Symbol) ? ANNOTATION_TYPES.fetch(annotation_type) : annotation_type

        get_all_annotations.select { |a| a.type == type_int }
      end

      # Filter annotations by criteria
      # @param criteria [Hash] Filter criteria
      # @return [Array<Types::Annotation>] Filtered annotations
      def filter_annotations(criteria = {})
        check_document!
        annotations = get_all_annotations

        annotations.select do |annot|
          (criteria[:page].nil? || annot.page == criteria[:page]) &&
            (criteria[:type].nil? || annot.type == criteria[:type]) &&
            (criteria[:text].nil? || (annot.text && annot.text.include?(criteria[:text])))
        end
      end

      # Export annotations to JSON
      # @param output_path [String, nil] Path to save JSON or nil to return string
      # @return [String] JSON string
      def export_annotations(output_path = nil)
        check_document!
        annotations = get_all_annotations
        json_data = annotations.map(&:to_h).to_json

        if output_path
          File.write(output_path, json_data)
          output_path
        else
          json_data
        end
      end

      # Import annotations from JSON or XFDF
      #
      # Imports annotations from a JSON file/string or XFDF file.
      # JSON format should match the output of export_annotations.
      # XFDF files are detected automatically.
      #
      # @param input_path_or_json [String] File path (JSON or XFDF) or JSON string
      # @return [Integer] Number of annotations imported
      # @raise [ParseError] If the input cannot be parsed
      # @raise [ArgumentError] If the input is invalid
      # @example Import from JSON file
      #   count = manager.import_annotations('/path/to/annotations.json')
      # @example Import from JSON string
      #   count = manager.import_annotations('[{"type":"highlight","page":0,"bbox":{"x":100,"y":200,"width":50,"height":20}}]')
      # @example Import from XFDF file
      #   count = manager.import_annotations('/path/to/annotations.xfdf')
      def import_annotations(input_path_or_json)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Input cannot be nil or empty' if input_path_or_json.nil? || input_path_or_json.empty?

        # Determine if input is a file path or raw data
        data = if File.exist?(input_path_or_json)
                 File.read(input_path_or_json)
               else
                 input_path_or_json
               end

        # Detect format and parse accordingly
        annotations = if data.strip.start_with?('<?xml') || data.strip.start_with?('<xfdf')
                        parse_xfdf_annotations(data)
                      else
                        parse_json_annotations(data)
                      end

        import_count = 0

        annotations.each do |annot|
          begin
            case annot[:type]
            when :highlight, 'highlight', ANNOTATION_TYPE_HIGHLIGHT
              add_highlight(annot[:page], annot[:bbox][:x], annot[:bbox][:y],
                            annot[:bbox][:width], annot[:bbox][:height], annot[:options] || {})
              import_count += 1
            when :underline, 'underline', ANNOTATION_TYPE_UNDERLINE
              add_underline(annot[:page], annot[:bbox][:x], annot[:bbox][:y],
                            annot[:bbox][:width], annot[:bbox][:height], annot[:options] || {})
              import_count += 1
            when :strikeout, 'strikeout', ANNOTATION_TYPE_STRIKEOUT
              add_strikeout(annot[:page], annot[:bbox][:x], annot[:bbox][:y],
                            annot[:bbox][:width], annot[:bbox][:height], annot[:options] || {})
              import_count += 1
            when :text, 'text', ANNOTATION_TYPE_TEXT
              add_comment(annot[:page], annot[:bbox][:x], annot[:bbox][:y],
                          annot[:text] || '', annot[:options] || {})
              import_count += 1
            else
              # Skip unsupported annotation types silently
              next
            end
          rescue ::PdfOxide::Error => e
            # Log error but continue importing other annotations
            next
          end
        end

        import_count
      end

      # Import annotations from block (DSL style)
      #
      # Provides a DSL for importing annotations programmatically.
      #
      # @yield [builder] Block to define annotations
      # @return [Integer] Number of annotations imported
      # @example
      #   count = manager.import_annotations_dsl do |builder|
      #     builder.highlight(page: 0, x: 100, y: 200, width: 50, height: 20)
      #     builder.comment(page: 0, x: 150, y: 300, text: "Important!")
      #   end
      def import_annotations_dsl
        check_document!
        builder = AnnotationBuilder.new(self)
        yield builder
        builder.import_count
      end

      # Get annotation statistics
      # @return [Hash] Statistics about annotations
      def annotation_statistics
        check_document!
        all_annots = get_all_annotations

        stats = {
          total_annotations: all_annots.count,
          by_type: {},
          by_page: {}
        }

        all_annots.each do |annot|
          type_name = TYPE_NAMES[annot.type] || 'unknown'
          stats[:by_type][type_name] ||= 0
          stats[:by_type][type_name] += 1

          stats[:by_page][annot.page] ||= 0
          stats[:by_page][annot.page] += 1
        end

        stats
      end

      private

      # Parse JSON annotation data
      # @param json_data [String] JSON string
      # @return [Array<Hash>] Parsed annotations
      def parse_json_annotations(json_data)
        parsed = JSON.parse(json_data, symbolize_names: true)
        parsed = [parsed] unless parsed.is_a?(Array)

        parsed.map do |annot|
          normalize_annotation_data(annot)
        end
      rescue JSON::ParserError => e
        raise ::PdfOxide::ParseError.new("Invalid JSON format: #{e.message}")
      end

      # Parse XFDF annotation data
      # @param xfdf_data [String] XFDF XML string
      # @return [Array<Hash>] Parsed annotations
      def parse_xfdf_annotations(xfdf_data)
        annotations = []

        # Simple XML parsing without external dependencies
        # Extract highlight annotations
        xfdf_data.scan(/<highlight([^>]*)>(.*?)<\/highlight>/m) do |attrs, content|
          page = extract_xml_attr(attrs, 'page').to_i
          rect = extract_xml_attr(attrs, 'rect')&.split(',')&.map(&:to_f) || [0, 0, 0, 0]

          annotations << {
            type: :highlight,
            page: page,
            bbox: { x: rect[0], y: rect[1], width: rect[2] - rect[0], height: rect[3] - rect[1] },
            text: extract_xml_content(content, 'contents'),
            options: {}
          }
        end

        # Extract underline annotations
        xfdf_data.scan(/<underline([^>]*)>(.*?)<\/underline>/m) do |attrs, content|
          page = extract_xml_attr(attrs, 'page').to_i
          rect = extract_xml_attr(attrs, 'rect')&.split(',')&.map(&:to_f) || [0, 0, 0, 0]

          annotations << {
            type: :underline,
            page: page,
            bbox: { x: rect[0], y: rect[1], width: rect[2] - rect[0], height: rect[3] - rect[1] },
            text: extract_xml_content(content, 'contents'),
            options: {}
          }
        end

        # Extract strikeout annotations
        xfdf_data.scan(/<strikeout([^>]*)>(.*?)<\/strikeout>/m) do |attrs, content|
          page = extract_xml_attr(attrs, 'page').to_i
          rect = extract_xml_attr(attrs, 'rect')&.split(',')&.map(&:to_f) || [0, 0, 0, 0]

          annotations << {
            type: :strikeout,
            page: page,
            bbox: { x: rect[0], y: rect[1], width: rect[2] - rect[0], height: rect[3] - rect[1] },
            text: extract_xml_content(content, 'contents'),
            options: {}
          }
        end

        # Extract text annotations (comments/notes)
        xfdf_data.scan(/<text([^>]*)>(.*?)<\/text>/m) do |attrs, content|
          page = extract_xml_attr(attrs, 'page').to_i
          rect = extract_xml_attr(attrs, 'rect')&.split(',')&.map(&:to_f) || [0, 0, 0, 0]

          annotations << {
            type: :text,
            page: page,
            bbox: { x: rect[0], y: rect[1], width: rect[2] - rect[0], height: rect[3] - rect[1] },
            text: extract_xml_content(content, 'contents'),
            options: { subject: extract_xml_attr(attrs, 'subject') }
          }
        end

        annotations
      end

      # Extract XML attribute value
      # @param attrs [String] Attribute string
      # @param name [String] Attribute name
      # @return [String, nil] Attribute value
      def extract_xml_attr(attrs, name)
        match = attrs.match(/#{name}=["']([^"']*)["']/)
        match ? match[1] : nil
      end

      # Extract XML element content
      # @param xml [String] XML content
      # @param tag [String] Tag name
      # @return [String, nil] Element content
      def extract_xml_content(xml, tag)
        match = xml.match(/<#{tag}[^>]*>([^<]*)<\/#{tag}>/)
        match ? match[1] : nil
      end

      # Normalize annotation data to consistent format
      # @param annot [Hash] Raw annotation data
      # @return [Hash] Normalized annotation
      def normalize_annotation_data(annot)
        # Handle different bbox formats
        bbox = if annot[:bbox].is_a?(Hash)
                 annot[:bbox]
               elsif annot[:rect].is_a?(Array)
                 { x: annot[:rect][0], y: annot[:rect][1],
                   width: annot[:rect][2] - annot[:rect][0],
                   height: annot[:rect][3] - annot[:rect][1] }
               elsif annot[:x] && annot[:y] && annot[:width] && annot[:height]
                 { x: annot[:x], y: annot[:y], width: annot[:width], height: annot[:height] }
               else
                 { x: 0, y: 0, width: 0, height: 0 }
               end

        # Normalize type
        type = annot[:type]
        type = type.to_sym if type.is_a?(String)

        {
          type: type,
          page: annot[:page].to_i,
          bbox: bbox,
          text: annot[:text] || annot[:contents],
          options: annot[:options] || {}
        }
      end

      def parse_annotation_list(handle, page_index)
        return [] if handle.nil? || handle.null?

        begin
          count = FFI::Bindings.pdf_oxide_annotation_count(handle)

          annotations = count.times.map do |i|
            type_int = FFI::Bindings.pdf_oxide_annotation_get_type(handle, i)
            text_ptr = FFI::Bindings.pdf_oxide_annotation_get_text(handle, i)
            color = FFI::Bindings.pdf_oxide_annotation_get_color(handle, i)

            bbox_ptr = ::FFI::MemoryPointer.new(:float, 4)
            FFI::Bindings.pdf_oxide_annotation_get_bbox(handle, i, bbox_ptr)
            x, y, width, height = bbox_ptr.read_array_of_float(4)

            Types::Annotation.new(
              type: type_int,
              page: page_index,
              text: FFI::StringMarshaller.read_c_string(text_ptr),
              bbox: Types::BoundingBox.new(x: x, y: y, width: width, height: height),
              color: color
            )
          end

          annotations
        ensure
          FFI::Bindings.pdf_oxide_annotation_list_free(handle) unless handle.nil? || handle.null?
        end
      end
    end

    # Builder class for DSL-style annotation import
    class AnnotationBuilder
      attr_reader :import_count

      # Initialize builder
      # @param manager [Annotation] Annotation manager
      def initialize(manager)
        @manager = manager
        @import_count = 0
      end

      # Add highlight annotation
      # @param page [Integer] Page index
      # @param x [Float] X coordinate
      # @param y [Float] Y coordinate
      # @param width [Float] Width
      # @param height [Float] Height
      # @param options [Hash] Additional options
      # @return [self]
      def highlight(page:, x:, y:, width:, height:, **options)
        @manager.add_highlight(page, x, y, width, height, options)
        @import_count += 1
        self
      end

      # Add underline annotation
      # @param page [Integer] Page index
      # @param x [Float] X coordinate
      # @param y [Float] Y coordinate
      # @param width [Float] Width
      # @param height [Float] Height
      # @param options [Hash] Additional options
      # @return [self]
      def underline(page:, x:, y:, width:, height:, **options)
        @manager.add_underline(page, x, y, width, height, options)
        @import_count += 1
        self
      end

      # Add strikeout annotation
      # @param page [Integer] Page index
      # @param x [Float] X coordinate
      # @param y [Float] Y coordinate
      # @param width [Float] Width
      # @param height [Float] Height
      # @param options [Hash] Additional options
      # @return [self]
      def strikeout(page:, x:, y:, width:, height:, **options)
        @manager.add_strikeout(page, x, y, width, height, options)
        @import_count += 1
        self
      end

      # Add text/comment annotation
      # @param page [Integer] Page index
      # @param x [Float] X coordinate
      # @param y [Float] Y coordinate
      # @param text [String] Comment text
      # @param options [Hash] Additional options
      # @return [self]
      def comment(page:, x:, y:, text:, **options)
        @manager.add_comment(page, x, y, text, options)
        @import_count += 1
        self
      end

      alias_method :text, :comment
      alias_method :note, :comment
    end
  end
end

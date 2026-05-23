# frozen_string_literal: true

require_relative 'base'

module PdfOxide
  module Managers
    # Manager for document metadata operations
    # Provides access to and modification of PDF document properties
    class Metadata < Base
      # Get document title
      # @return [String, nil] Document title
      def title
        check_document!
        FFI::StringMarshaller.from_c_string(
          with_error_check('get_title') do |error_ptr|
            FFI::Bindings.pdf_document_get_title(@document.handle, error_ptr)
          end
        )
      end

      # Get document author
      # @return [String, nil] Document author
      def author
        check_document!
        FFI::StringMarshaller.from_c_string(
          with_error_check('get_author') do |error_ptr|
            FFI::Bindings.pdf_document_get_author(@document.handle, error_ptr)
          end
        )
      end

      # Get document subject
      # @return [String, nil] Document subject
      def subject
        check_document!
        FFI::StringMarshaller.from_c_string(
          with_error_check('get_subject') do |error_ptr|
            FFI::Bindings.pdf_document_get_subject(@document.handle, error_ptr)
          end
        )
      end

      # Get document keywords
      # @return [String, nil] Document keywords
      def keywords
        check_document!
        FFI::StringMarshaller.from_c_string(
          with_error_check('get_keywords') do |error_ptr|
            FFI::Bindings.pdf_document_get_keywords(@document.handle, error_ptr)
          end
        )
      end

      # Get document creator
      # @return [String, nil] Document creator
      def creator
        check_document!
        FFI::StringMarshaller.from_c_string(
          with_error_check('get_creator') do |error_ptr|
            FFI::Bindings.pdf_document_get_creator(@document.handle, error_ptr)
          end
        )
      end

      # Get document producer
      # @return [String, nil] Document producer
      def producer
        check_document!
        FFI::StringMarshaller.from_c_string(
          with_error_check('get_producer') do |error_ptr|
            FFI::Bindings.pdf_document_get_producer(@document.handle, error_ptr)
          end
        )
      end

      # Get document creation date
      # @return [String, nil] Creation date as ISO 8601 string
      def creation_date
        check_document!
        FFI::StringMarshaller.from_c_string(
          with_error_check('get_creation_date') do |error_ptr|
            FFI::Bindings.pdf_document_get_creation_date(@document.handle, error_ptr)
          end
        )
      end

      # Get document modification date
      # @return [String, nil] Modification date as ISO 8601 string
      def modification_date
        check_document!
        FFI::StringMarshaller.from_c_string(
          with_error_check('get_modification_date') do |error_ptr|
            FFI::Bindings.pdf_document_get_modification_date(@document.handle, error_ptr)
          end
        )
      end

      # Set document title
      # @param title [String] New title
      # @return [Boolean] Whether operation succeeded
      def set_title(title)
        check_document!
        title_utf8 = FFI::StringMarshaller.to_utf8(title)
        with_error_check('set_title', title: title) do |error_ptr|
          FFI::Bindings.pdf_document_set_title(@document.handle, title_utf8, error_ptr)
        end
        true
      end

      # Set document author
      # @param author [String] New author
      # @return [Boolean] Whether operation succeeded
      def set_author(author)
        check_document!
        author_utf8 = FFI::StringMarshaller.to_utf8(author)
        with_error_check('set_author', author: author) do |error_ptr|
          FFI::Bindings.pdf_document_set_author(@document.handle, author_utf8, error_ptr)
        end
        true
      end

      # Set document subject
      # @param subject [String] New subject
      # @return [Boolean] Whether operation succeeded
      def set_subject(subject)
        check_document!
        subject_utf8 = FFI::StringMarshaller.to_utf8(subject)
        with_error_check('set_subject', subject: subject) do |error_ptr|
          FFI::Bindings.pdf_document_set_subject(@document.handle, subject_utf8, error_ptr)
        end
        true
      end

      # Set document keywords
      # @param keywords [String] New keywords
      # @return [Boolean] Whether operation succeeded
      def set_keywords(keywords)
        check_document!
        keywords_utf8 = FFI::StringMarshaller.to_utf8(keywords)
        with_error_check('set_keywords', keywords: keywords) do |error_ptr|
          FFI::Bindings.pdf_document_set_keywords(@document.handle, keywords_utf8, error_ptr)
        end
        true
      end

      # Get all metadata as hash
      # @return [Hash] All metadata properties
      def all
        check_document!
        {
          title: title,
          author: author,
          subject: subject,
          keywords: keywords,
          creator: creator,
          producer: producer,
          creation_date: creation_date,
          modification_date: modification_date
        }
      end

      # Set multiple metadata properties at once
      # @param metadata [Hash] Hash of metadata properties
      # @return [Boolean] Whether all operations succeeded
      def set_all(metadata)
        check_document!
        metadata.each do |key, value|
          case key
          when :title
            set_title(value)
          when :author
            set_author(value)
          when :subject
            set_subject(value)
          when :keywords
            set_keywords(value)
          end
        end
        true
      end

      # Check if any metadata is present
      # @return [Boolean] Whether document has metadata
      def empty?
        check_document!
        all.values.all?(&:nil?)
      end

      # Get metadata as JSON-compatible hash
      # @return [Hash] Metadata as JSON
      def to_h
        all
      end

      # Get metadata as JSON string
      # @return [String] Metadata as JSON
      def to_json(*args)
        require 'json'
        to_h.to_json(*args)
      end
    end
  end
end

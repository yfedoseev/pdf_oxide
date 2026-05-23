# frozen_string_literal: true

module PdfOxide
  module Types
    # Options for text search operations
    class SearchOptions
      attr_accessor :case_sensitive, :whole_words_only, :regex, :include_annotations, :ignore_accents

      # Initialize search options
      # @param case_sensitive [Boolean] Case sensitive search
      # @param whole_words_only [Boolean] Match whole words only
      # @param regex [Boolean] Use regex pattern
      # @param include_annotations [Boolean] Include annotations in search
      # @param ignore_accents [Boolean] Ignore accents in search
      def initialize(
        case_sensitive: false,
        whole_words_only: false,
        regex: false,
        include_annotations: true,
        ignore_accents: false
      )
        @case_sensitive = case_sensitive
        @whole_words_only = whole_words_only
        @regex = regex
        @include_annotations = include_annotations
        @ignore_accents = ignore_accents
      end

      # Create case-insensitive search options
      # @return [SearchOptions] New instance
      def self.case_insensitive
        new(case_sensitive: false)
      end

      # Create case-sensitive search options
      # @return [SearchOptions] New instance
      def self.case_sensitive
        new(case_sensitive: true)
      end

      # Create regex search options
      # @return [SearchOptions] New instance
      def self.regex
        new(regex: true)
      end

      # Convert to hash
      # @return [Hash] Hash representation
      def to_h
        {
          case_sensitive: @case_sensitive,
          whole_words_only: @whole_words_only,
          regex: @regex,
          include_annotations: @include_annotations,
          ignore_accents: @ignore_accents
        }
      end

      # Convert to string
      # @return [String] String representation
      def to_s
        flags = []
        flags << 'case-sensitive' if @case_sensitive
        flags << 'whole-words' if @whole_words_only
        flags << 'regex' if @regex
        "SearchOptions(#{flags.join(', ')})"
      end
    end
  end
end

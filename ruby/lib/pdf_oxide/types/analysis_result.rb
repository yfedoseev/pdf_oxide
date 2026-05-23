# frozen_string_literal: true

module PdfOxide
  module Types
    # Represents page analysis results
    class AnalysisResult
      attr_reader :page, :complexity_level, :complexity_score, :content_type, :text_density, :image_density

      def initialize(page: 0, complexity_level: :simple, complexity_score: 0.0, content_type: :mixed,
                     text_density: 0.0, image_density: 0.0)
        @page = page
        @complexity_level = complexity_level
        @complexity_score = complexity_score
        @content_type = content_type
        @text_density = text_density
        @image_density = image_density
      end

      def to_h
        {
          page: @page,
          complexity_level: @complexity_level,
          complexity_score: @complexity_score,
          content_type: @content_type,
          text_density: @text_density,
          image_density: @image_density
        }
      end

      def to_s
        "AnalysisResult(page=#{@page}, complexity=#{@complexity_level}, score=#{@complexity_score})"
      end

      def inspect
        to_s
      end

      def text_heavy?
        @text_density > 0.7
      end

      def image_heavy?
        @image_density > 0.7
      end

      def simple?
        @complexity_level == :simple
      end

      def complex?
        @complexity_level == :complex || @complexity_level == :very_complex
      end

      def ==(other)
        other.is_a?(AnalysisResult) && page == other.page && complexity_score == other.complexity_score
      end

      def hash
        [page, complexity_score].hash
      end
    end
  end
end

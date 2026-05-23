# frozen_string_literal: true

module PdfOxide
  module Types
    # Represents OCR results for a page or document
    class OcrResult
      attr_reader :text, :confidence, :language, :page

      def initialize(text:, confidence: 0.0, language: 'en', page: 0)
        @text = text
        @confidence = confidence
        @language = language
        @page = page
      end

      def to_h
        {
          text: @text,
          confidence: @confidence,
          language: @language,
          page: @page
        }
      end

      def to_s
        "OcrResult(page=#{@page}, confidence=#{@confidence}, text_length=#{@text.length})"
      end

      def inspect
        to_s
      end

      def ==(other)
        other.is_a?(OcrResult) && text == other.text && page == other.page
      end

      def hash
        [text, page].hash
      end
    end
  end
end

# frozen_string_literal: true

module PdfOxide
  module Types
    # Represents PDF compliance validation results
    class ComplianceResult
      attr_reader :type, :compliant, :error_count, :warning_count, :errors, :warnings, :report

      def initialize(type:, compliant:, error_count: 0, warning_count: 0, errors: [], warnings: [], report: '')
        @type = type
        @compliant = compliant
        @error_count = error_count
        @warning_count = warning_count
        @errors = errors
        @warnings = warnings
        @report = report
      end

      def to_h
        {
          type: @type,
          compliant: @compliant,
          error_count: @error_count,
          warning_count: @warning_count,
          errors: @errors,
          warnings: @warnings,
          report: @report
        }
      end

      def to_s
        "ComplianceResult(type=#{@type}, compliant=#{@compliant}, errors=#{@error_count}, warnings=#{@warning_count})"
      end

      def inspect
        to_s
      end

      def has_errors?
        @error_count.positive?
      end

      def has_warnings?
        @warning_count.positive?
      end

      def ==(other)
        other.is_a?(ComplianceResult) && type == other.type && compliant == other.compliant
      end

      def hash
        [type, compliant].hash
      end
    end
  end
end

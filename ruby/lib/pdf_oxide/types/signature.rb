# frozen_string_literal: true

module PdfOxide
  module Types
    # Represents a digital signature on a PDF
    class Signature
      attr_reader :signer, :timestamp, :status, :reason, :location, :certificate

      def initialize(signer:, timestamp: nil, status: :unknown, reason: '', location: '', certificate: nil)
        @signer = signer
        @timestamp = timestamp
        @status = status
        @reason = reason
        @location = location
        @certificate = certificate
      end

      def to_h
        {
          signer: @signer,
          timestamp: @timestamp,
          status: @status,
          reason: @reason,
          location: @location,
          certificate: @certificate
        }
      end

      def to_s
        "Signature(signer=#{@signer}, status=#{@status})"
      end

      def inspect
        to_s
      end

      def valid?
        @status == :valid
      end

      def trusted?
        @status == :valid || @status == :self_signed
      end

      def signed_at
        @timestamp ? Time.at(@timestamp) : nil
      end

      def ==(other)
        other.is_a?(Signature) && signer == other.signer && timestamp == other.timestamp
      end

      def hash
        [signer, timestamp].hash
      end
    end
  end
end

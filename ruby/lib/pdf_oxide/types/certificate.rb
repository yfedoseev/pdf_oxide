# frozen_string_literal: true

module PdfOxide
  module Types
    # Represents an X.509 certificate used for PDF signatures
    class Certificate
      attr_reader :subject, :issuer, :valid_from, :valid_to, :serial_number, :thumbprint

      def initialize(subject:, issuer: '', valid_from: nil, valid_to: nil, serial_number: '', thumbprint: '')
        @subject = subject
        @issuer = issuer
        @valid_from = valid_from
        @valid_to = valid_to
        @serial_number = serial_number
        @thumbprint = thumbprint
      end

      def to_h
        {
          subject: @subject,
          issuer: @issuer,
          valid_from: @valid_from,
          valid_to: @valid_to,
          serial_number: @serial_number,
          thumbprint: @thumbprint
        }
      end

      def to_s
        "Certificate(subject=#{@subject}, issuer=#{@issuer})"
      end

      def inspect
        to_s
      end

      def valid?
        now = Time.now
        (@valid_from.nil? || now >= Time.at(@valid_from)) &&
          (@valid_to.nil? || now <= Time.at(@valid_to))
      end

      def valid_from_time
        @valid_from ? Time.at(@valid_from) : nil
      end

      def valid_to_time
        @valid_to ? Time.at(@valid_to) : nil
      end

      def ==(other)
        other.is_a?(Certificate) && subject == other.subject && serial_number == other.serial_number
      end

      def hash
        [subject, serial_number].hash
      end
    end
  end
end

#!/usr/bin/env ruby
# frozen_string_literal: true

require 'pdf_oxide'

# Basic text extraction example
# Usage: ruby examples/basic_extraction.rb document.pdf

if ARGV.empty?
  puts "Usage: #{$PROGRAM_NAME} <pdf_file>"
  exit 1
end

pdf_file = ARGV[0]

unless File.exist?(pdf_file)
  puts "Error: File not found: #{pdf_file}"
  exit 1
end

PdfOxide::Document.open(pdf_file) do |doc|
  puts "=" * 70
  puts "PDF TEXT EXTRACTION"
  puts "=" * 70
  puts "File: #{pdf_file}"
  puts "Pages: #{doc.page_count}"
  puts "Version: #{doc.version}"
  puts "Encrypted: #{doc.encrypted?}"
  puts "=" * 70
  puts

  # Extract text from each page
  doc.page_count.times do |page_index|
    puts "Page #{page_index + 1}:"
    puts "-" * 70

    text = doc.extraction.extract_text(page_index)
    puts text
    puts
  end
end

puts "Extraction complete!"

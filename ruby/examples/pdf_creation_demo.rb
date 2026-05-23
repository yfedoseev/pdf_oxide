#!/usr/bin/env ruby
# frozen_string_literal: true

require 'pdf_oxide'

# PDF creation example
# Usage: ruby examples/pdf_creation_demo.rb [output_file]

output_file = ARGV[0] || 'created_document.pdf'

puts "=" * 70
puts "PDF CREATION"
puts "=" * 70
puts "Output: #{output_file}"
puts "=" * 70
puts

# Create PDF from Markdown
markdown_content = <<~MARKDOWN
  # PDF Oxide Ruby Binding

  ## Features

  - Complete API coverage
  - High performance
  - Idiomatic Ruby

  ## Supported Operations

  1. Text extraction
  2. Page rendering
  3. Search and replace
  4. OCR processing
  5. Digital signatures
  6. PDF compliance
MARKDOWN

puts "Creating PDF from Markdown..."

creator = PdfOxide::Creator.from_markdown(markdown_content)
creator.title('PDF Oxide Ruby Example')
creator.author('Ruby Binding')
creator.subject('Demonstration Document')

creator.save(output_file)

puts "✓ PDF created successfully!"
puts "Output file: #{File.absolute_path(output_file)}"
puts "File size: #{File.size(output_file)} bytes"

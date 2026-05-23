#!/usr/bin/env ruby
# frozen_string_literal: true

require 'pdf_oxide'

# Text search example
# Usage: ruby examples/search_demo.rb document.pdf [query]

if ARGV.empty?
  puts "Usage: #{$PROGRAM_NAME} <pdf_file> [search_query]"
  exit 1
end

pdf_file = ARGV[0]
query = ARGV[1] || 'the'

unless File.exist?(pdf_file)
  puts "Error: File not found: #{pdf_file}"
  exit 1
end

PdfOxide::Document.open(pdf_file) do |doc|
  puts "=" * 70
  puts "SEARCH RESULTS"
  puts "=" * 70
  puts "File: #{pdf_file}"
  puts "Query: \"#{query}\""
  puts "=" * 70
  puts

  # Search all pages
  results = doc.search.search_all(query, case_sensitive: false)

  if results.empty?
    puts "No matches found."
  else
    puts "Found #{results.count} matches:\n\n"

    results.each_with_index do |result, index|
      puts "#{index + 1}. Page #{result.page_number}"
      puts "   Text: #{result.text}"
      puts "   Position: (#{result.bbox.x.round(1)}, #{result.bbox.y.round(1)})"
      puts "   Size: #{result.bbox.width.round(1)}x#{result.bbox.height.round(1)}"
      puts
    end
  end
end

# frozen_string_literal: true

# 01 — Extract text (Ruby)
#
# Opens a PDF, prints the page count, then the text of each page.
#
#   ruby main.rb ../../../tests/fixtures/simple.pdf

require 'pdf_oxide'

path = ARGV[0] or abort 'usage: ruby main.rb <pdf>'

PdfOxide.open(path) do |doc|
  puts "Pages: #{doc.page_count}"
  doc.page_count.times do |i|
    puts "--- Page #{i + 1} ---"
    puts doc.extract_text(i)
  end
end

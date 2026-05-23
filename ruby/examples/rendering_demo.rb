#!/usr/bin/env ruby
# frozen_string_literal: true

require 'pdf_oxide'
require 'fileutils'

# Page rendering example
# Usage: ruby examples/rendering_demo.rb document.pdf [output_dir]

if ARGV.empty?
  puts "Usage: #{$PROGRAM_NAME} <pdf_file> [output_dir]"
  exit 1
end

pdf_file = ARGV[0]
output_dir = ARGV[1] || 'rendered_pages'

unless File.exist?(pdf_file)
  puts "Error: File not found: #{pdf_file}"
  exit 1
end

FileUtils.mkdir_p(output_dir)

PdfOxide::Document.open(pdf_file) do |doc|
  puts "=" * 70
  puts "PAGE RENDERING"
  puts "=" * 70
  puts "File: #{pdf_file}"
  puts "Output: #{output_dir}"
  puts "Pages to render: #{doc.page_count}"
  puts "=" * 70
  puts

  # Render all pages with different quality presets
  [
    { preset: 'draft', options: PdfOxide::Types::RenderOptions.draft },
    { preset: 'normal', options: PdfOxide::Types::RenderOptions.normal },
    { preset: 'high', options: PdfOxide::Types::RenderOptions.high }
  ].each do |config|
    preset_dir = File.join(output_dir, config[:preset])
    FileUtils.mkdir_p(preset_dir)

    puts "Rendering at #{config[:preset]} quality..."

    doc.page_count.times do |page_index|
      output_path = File.join(preset_dir, "page_#{page_index + 1}.png")
      doc.rendering.render_page_to_file(page_index, output_path, config[:options])
      puts "  ✓ Rendered page #{page_index + 1}"
    end

    puts
  end
end

puts "Rendering complete!"
puts "Output saved to: #{File.absolute_path(output_dir)}"

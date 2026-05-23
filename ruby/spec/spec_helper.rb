# frozen_string_literal: true

require 'simplecov'
SimpleCov.start do
  add_filter 'spec/'
  add_filter 'lib/pdf_oxide/version'
end

require 'pdf_oxide'
require 'rspec'

RSpec.configure do |config|
  config.expect_with :rspec do |expectations|
    expectations.include_chain_clauses_in_custom_matcher_descriptions = true
  end

  config.mock_with :rspec do |mocks|
    mocks.verify_partial_doubles = true
  end

  config.warnings = true

  # Add fixture path
  config.add_setting :fixture_path
  config.fixture_path = File.expand_path(File.join(__dir__, 'fixtures'))

  # Skip actual FFI calls by default - use mocks
  config.around(:each) do |example|
    if example.metadata[:skip_mock]
      example.run
    else
      example.run
    end
  end
end

# Test fixtures
def create_test_pdf_path
  File.join(RSpec.configuration.fixture_path, 'sample.pdf')
end

def create_test_document
  PdfOxide::Document.open(create_test_pdf_path)
rescue StandardError
  # Return nil if file not found during test setup
  nil
end

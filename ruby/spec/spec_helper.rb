# frozen_string_literal: true

require 'simplecov'

# Optional LCOV emit for Codecov; gated by env so dev/local runs stay
# fast and we don't hard-require simplecov-lcov in the dev Gemfile.
if ENV['COVERAGE_LCOV'] == '1'
  require 'simplecov-lcov'
  SimpleCov::Formatter::LcovFormatter.config do |c|
    c.report_with_single_file = true
    c.single_report_path = 'coverage/lcov.info'
  end
  SimpleCov.formatters = SimpleCov::Formatter::MultiFormatter.new(
    [
      SimpleCov::Formatter::HTMLFormatter,
      SimpleCov::Formatter::LcovFormatter
    ]
  )
end

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

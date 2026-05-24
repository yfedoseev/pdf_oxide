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

# Resolve the shared fixture set used by every binding's smoke tests.
# Tests skip silently when the directory isn't reachable (out-of-tree
# consumers won't have it).
PDF_OXIDE_FIXTURE_ROOT = File.expand_path('../../tests/fixtures', __dir__).freeze

RSpec.configure do |config|
  config.expect_with :rspec do |expectations|
    expectations.include_chain_clauses_in_custom_matcher_descriptions = true
  end

  config.mock_with :rspec do |mocks|
    mocks.verify_partial_doubles = true
  end
end

def fixture(name)
  File.join(PDF_OXIDE_FIXTURE_ROOT, name)
end

# Skip a whole example group when the fixture set is absent.
RSpec.shared_context 'fixtures-present' do
  before(:all) do
    skip "fixtures dir not present: #{PDF_OXIDE_FIXTURE_ROOT}" unless Dir.exist?(PDF_OXIDE_FIXTURE_ROOT)
  end
end

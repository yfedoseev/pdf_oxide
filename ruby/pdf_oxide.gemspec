# frozen_string_literal: true

require_relative 'lib/pdf_oxide/version'

Gem::Specification.new do |spec|
  spec.name = 'pdf_oxide'
  spec.version = PdfOxide::VERSION
  spec.authors = ['PDF Oxide Contributors']
  spec.email = ['support@pdf-oxide.dev']

  spec.summary = 'Ruby bindings for PDF Oxide - high-performance PDF processing'
  spec.description = 'Idiomatic Ruby bindings for PDF Oxide. Process, analyze, ' \
                     'and generate PDFs through the libpdf_oxide cdylib used by ' \
                     'the Python, Java, Node, Go, and C# bindings.'
  spec.homepage = 'https://github.com/fyi-oxide/pdf_oxide'
  spec.license = 'Apache-2.0'
  spec.required_ruby_version = '>= 2.7.0'

  spec.metadata = {
    'homepage_uri' => spec.homepage,
    'source_code_uri' => 'https://github.com/fyi-oxide/pdf_oxide',
    'bug_tracker_uri' => 'https://github.com/fyi-oxide/pdf_oxide/issues',
    'documentation_uri' => 'https://rubydoc.info/gems/pdf_oxide',
    'changelog_uri' => 'https://github.com/fyi-oxide/pdf_oxide/blob/main/CHANGELOG.md'
  }

  # ship only library code, the LICENSE, the README, and the Gemfile.
  # Promotional PHASE*/IMPLEMENTATION_*/RUBY_*.md status files live alongside
  # the gem on disk but are deliberately omitted from `spec.files` so they
  # do not appear on RubyGems.
  spec.files = Dir.glob('lib/**/*.rb') + Dir.glob('ext/**/*.{rb,c,h}') +
               %w[README.md LICENSE Gemfile]
  spec.require_paths = ['lib']

  # Runtime dependency
  spec.add_dependency 'ffi', '~> 1.16'

  # Development dependencies
  spec.add_development_dependency 'bundler', '>= 2.0'
  spec.add_development_dependency 'rake', '~> 13.0'
  spec.add_development_dependency 'rspec', '~> 3.12'
  spec.add_development_dependency 'rubocop', '~> 1.50'
  spec.add_development_dependency 'rubocop-rspec', '~> 2.20'
  spec.add_development_dependency 'yard', '~> 0.9'
  spec.add_development_dependency 'simplecov', '~> 0.22'
end

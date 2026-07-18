# frozen_string_literal: true

require_relative 'lib/pdf_oxide/version'

Gem::Specification.new do |spec|
  spec.name = 'pdf_oxide'
  spec.version = PdfOxide::VERSION
  spec.authors = ['PDF Oxide Contributors']
  spec.email = ['support@pdf-oxide.dev']

  # Native-gem build matrix.  When PDF_OXIDE_GEM_PLATFORM is set the
  # gemspec emits a platform-tagged gem; otherwise we ship a source gem
  # (no bundled cdylib — user installs Rust + cargo build).
  if (gem_plat = ENV['PDF_OXIDE_GEM_PLATFORM']) && !gem_plat.empty?
    spec.platform = Gem::Platform.new(gem_plat)
  end

  spec.summary = 'The fastest Ruby PDF library — 5× faster than the industry leaders, 100% pass rate on 3,830 real-world PDFs'
  spec.description = 'The fastest Ruby PDF library: 0.8ms mean, 5× faster than the ' \
                     'industry leaders, 100% pass rate on 3,830 real-world PDFs. ' \
                     'Idiomatic Ruby bindings for text extraction, Markdown/HTML ' \
                     'conversion, and PDF creation and editing over the pdf_oxide ' \
                     'Rust core (the libpdf_oxide cdylib shared by the Python, ' \
                     'Java, Node, Go, and C# bindings).'
  spec.homepage = 'https://github.com/yfedoseev/pdf_oxide'
  # Dual-licensed at the repo root (MIT OR Apache-2.0); mirror that here.
  spec.licenses = ['MIT', 'Apache-2.0']
  spec.required_ruby_version = '>= 3.1.0'

  spec.metadata = {
    'homepage_uri' => spec.homepage,
    'source_code_uri' => 'https://github.com/yfedoseev/pdf_oxide',
    'bug_tracker_uri' => 'https://github.com/yfedoseev/pdf_oxide/issues',
    'documentation_uri' => 'https://rubydoc.info/gems/pdf_oxide',
    'changelog_uri' => 'https://github.com/yfedoseev/pdf_oxide/blob/main/CHANGELOG.md'
  }

  # ship only library code, the LICENSE, the README, and the Gemfile.
  # Promotional PHASE*/IMPLEMENTATION_*/RUBY_*.md status files live alongside
  # the gem on disk but are deliberately omitted from `spec.files` so they
  # do not appear on RubyGems.
  #
  # For platform-tagged gems (built with `gem build --platform <plat>`),
  # the CI staging step copies the per-target cdylib into ext/pdf_oxide/
  # so the binary-glob below packs the right libpdf_oxide.{so,dylib,dll}
  # into the gem.  The plain `gem build pdf_oxide.gemspec` (source gem)
  # picks up whatever happens to be in ext/pdf_oxide/ — typically nothing,
  # because users install Rust + `cargo build --release` themselves.
  spec.files = Dir.glob('lib/**/*.rb') +
               Dir.glob('ext/pdf_oxide/*.{so,dylib,dll}') +
               Dir.glob('ext/pdf_oxide/*.{rb,c,h}') +
               %w[README.md LICENSE LICENSE-MIT LICENSE-APACHE Gemfile]
  spec.require_paths = ['lib']

  # Runtime dependency
  spec.add_dependency 'ffi', '~> 1.16'

  # Development dependencies
  spec.add_development_dependency 'bundler', '>= 2.0'
  spec.add_development_dependency 'rake', '~> 13.0'
  spec.add_development_dependency 'rspec', '~> 3.12'
  spec.add_development_dependency 'rubocop', '~> 1.86'
  spec.add_development_dependency 'rubocop-rspec', '~> 3.9'
  spec.add_development_dependency 'simplecov-lcov', '~> 0.8'
  spec.add_development_dependency 'yard', '~> 0.9'
  spec.add_development_dependency 'simplecov', '~> 0.22'
end

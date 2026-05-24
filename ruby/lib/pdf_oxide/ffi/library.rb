# frozen_string_literal: true

require 'ffi'
require 'rbconfig'

module PdfOxide
  module FFI
    # Loads the native PDF Oxide library with cross-platform support
    module Library
      # Finds library for current platform
      # @return [Array<String>] Library names to try loading
      def self.find_library
        case RbConfig::CONFIG['host_os']
        when /darwin/
          %w[libpdf_oxide.dylib libpdf_oxide.0.dylib]
        when /linux/
          %w[libpdf_oxide.so libpdf_oxide.so.0]
        when /mswin|mingw/
          %w[pdf_oxide.dll libpdf_oxide.dll]
        else
          raise UnsupportedPlatformError, "Unsupported OS: #{RbConfig::CONFIG['host_os']}"
        end
      end

      # @return [String] Path to native library
      def self.library_path
        @library_path ||= find_library_path
      end

      def self.find_library_path
        # Try to find in standard locations
        find_library.each do |lib_name|
          # Native-gem layout: cdylib staged inside the gem at
          # ext/pdf_oxide/ during platform-specific gem packaging.  This is
          # the path bundled into platform-tagged gems and is the first
          # thing the loader should try when installed from a native gem.
          gem_native = File.expand_path("../../../ext/pdf_oxide/#{lib_name}", __dir__)
          return gem_native if File.exist?(gem_native)

          # Try system paths
          result = system_find_library(lib_name)
          return result if result

          # Try relative to gem (dev-checkout layouts)
          relative_paths = [
            File.expand_path("../../target/release/#{lib_name}", __dir__),
            File.expand_path("../../target/debug/#{lib_name}", __dir__),
            File.expand_path("../../../target/release/#{lib_name}", __dir__),
            File.expand_path("../../../target/debug/#{lib_name}", __dir__),
            lib_name
          ]

          relative_paths.each do |path|
            return path if File.exist?(path)
          end
        end

        # Fallback to library name (system will search)
        find_library.first
      end

      def self.system_find_library(lib_name)
        case RbConfig::CONFIG['host_os']
        when /darwin/
          ldconfig_search(lib_name) || homebrew_find(lib_name)
        when /linux/
          ldconfig_search(lib_name)
        when /mswin|mingw/
          windows_find(lib_name)
        end
      end

      def self.ldconfig_search(lib_name)
        output = `ldconfig -p 2>/dev/null | grep #{lib_name}`.strip
        return nil if output.empty?

        output.split("\n").first&.split('=>')&.last&.strip
      rescue StandardError
        nil
      end

      def self.homebrew_find(lib_name)
        output = `brew --prefix 2>/dev/null`.strip
        return nil if output.empty?

        path = File.join(output, 'lib', lib_name)
        File.exist?(path) ? path : nil
      rescue StandardError
        nil
      end

      def self.windows_find(_lib_name)
        # Windows DLL search path is handled by system
        nil
      end
    end
  end
end

# frozen_string_literal: true

require 'json'
require_relative 'base'

module PdfOxide
  module Managers
    # Manager for cache operations and optimization
    # Controls page rendering cache and performance tuning
    class Cache < Base
      # Clear entire cache
      # @return [void]
      def clear
        check_document!
        clear_cache
      end

      # Invalidate cache for specific page
      # @param page_index [Integer] Page index (0-indexed)
      # @return [void]
      def invalidate_page(page_index)
        check_document!
        validate_page_index!(page_index)
        super(page_index)
      end

      # Get cache statistics
      # @return [Hash] Cache statistics
      def statistics
        check_document!
        cache_statistics
      end

      # Set maximum cache size in bytes
      # @param size_bytes [Integer] Maximum size in bytes
      # @return [void]
      def set_max_size(size_bytes)
        check_document!
        raise ::PdfOxide::ArgumentError, 'Size must be positive' if size_bytes <= 0

        FFI::Bindings.pdf_cache_set_max_size(@document.handle, size_bytes)
      end

      # Get current cache size in bytes
      # @return [Integer] Current cache size
      def get_current_size
        check_document!
        stats = statistics
        stats[:memory_used_bytes] || 0
      end

      # Get cache hit rate
      # @return [Float] Hit rate (0.0 to 1.0)
      def get_hit_rate
        check_document!
        stats = statistics
        hits = stats[:cache_hits] || 0
        misses = stats[:cache_misses] || 0
        total = hits + misses

        return 0.0 if total.zero?

        hits.to_f / total.to_f
      end

      # Get cache miss rate
      # @return [Float] Miss rate (0.0 to 1.0)
      def get_miss_rate
        check_document!
        1.0 - get_hit_rate
      end

      # Get number of cached pages
      # @return [Integer] Number of pages in cache
      def get_cached_pages_count
        check_document!
        stats = statistics
        stats[:total_cached_pages] || 0
      end

      # Get number of cache hits
      # @return [Integer] Total cache hits
      def get_hit_count
        check_document!
        stats = statistics
        stats[:cache_hits] || 0
      end

      # Get number of cache misses
      # @return [Integer] Total cache misses
      def get_miss_count
        check_document!
        stats = statistics
        stats[:cache_misses] || 0
      end

      # Get total cache accesses
      # @return [Integer] Total hits + misses
      def get_total_accesses
        check_document!
        get_hit_count + get_miss_count
      end

      # Get cache information
      # @return [Hash] Comprehensive cache information
      def info
        check_document!
        stats = statistics

        {
          cached_pages: stats[:total_cached_pages] || 0,
          memory_used: stats[:memory_used_bytes] || 0,
          hits: stats[:cache_hits] || 0,
          misses: stats[:cache_misses] || 0,
          total_accesses: get_total_accesses,
          hit_rate: get_hit_rate,
          miss_rate: get_miss_rate
        }
      end

      # Get cache information as formatted string
      # @return [String] Formatted cache info
      def info_string
        check_document!
        info_hash = info

        output = "Cache Statistics\n"
        output += "=" * 40 + "\n\n"
        output += "Cached Pages: #{info_hash[:cached_pages]}\n"
        output += "Memory Used: #{format_bytes(info_hash[:memory_used])}\n"
        output += "Cache Hits: #{info_hash[:hits]}\n"
        output += "Cache Misses: #{info_hash[:misses]}\n"
        output += "Total Accesses: #{info_hash[:total_accesses]}\n"
        output += "Hit Rate: #{(info_hash[:hit_rate] * 100).round(2)}%\n"
        output += "Miss Rate: #{(info_hash[:miss_rate] * 100).round(2)}%\n"

        output
      end

      # Reset cache statistics
      # @return [void]
      def reset_statistics
        check_document!
        clear
      end

      # Enable caching (clear and initialize)
      # @return [void]
      def enable
        check_document!
        clear
      end

      # Disable caching (clear cache)
      # @return [void]
      def disable
        check_document!
        clear
      end

      # Check if cache is enabled
      # @return [Boolean] Whether caching is enabled
      def enabled?
        check_document!
        true # Caching is generally always available
      end

      # Export cache statistics as JSON
      #
      # Exports comprehensive cache statistics as a JSON-formatted string.
      # Cache hit and miss rates are converted to percentages (0-100) and rounded
      # to two decimal places. A timestamp of the export time is included.
      #
      # This is useful for:
      # - Logging cache performance metrics
      # - Monitoring cache efficiency over time
      # - Exporting to external monitoring systems
      # - Creating performance reports
      #
      # @param *args Additional arguments passed to Hash#to_json
      # @return [String] JSON string with keys:
      #   - cached_pages [Integer] Number of pages in cache
      #   - memory_used_bytes [Integer] Memory used by cache in bytes
      #   - cache_hits [Integer] Total number of cache hits
      #   - cache_misses [Integer] Total number of cache misses
      #   - total_accesses [Integer] Sum of hits and misses
      #   - hit_rate [Float] Hit rate as percentage (0-100)
      #   - miss_rate [Float] Miss rate as percentage (0-100)
      #   - timestamp [Integer] Unix timestamp of export
      # @raise [PdfException] if document is invalid or closed
      # @example
      #   manager = cache_manager
      #   json = manager.to_json
      #   parsed = JSON.parse(json)
      #   puts "Hit rate: #{parsed['hit_rate']}%"
      def to_json(*args)
        check_document!
        cache_info = info

        json_data = {
          cached_pages: cache_info[:cached_pages],
          memory_used_bytes: cache_info[:memory_used],
          cache_hits: cache_info[:hits],
          cache_misses: cache_info[:misses],
          total_accesses: cache_info[:total_accesses],
          hit_rate: (cache_info[:hit_rate] * 100).round(2),
          miss_rate: (cache_info[:miss_rate] * 100).round(2),
          timestamp: Time.now.to_i
        }

        json_data.to_json(*args)
      end

      # Export cache statistics as hash (JSON compatible)
      #
      # Exports comprehensive cache statistics as a Ruby hash.
      # Cache hit and miss rates are converted to percentages for easier
      # interpretation. The hash includes all statistics and a timestamp.
      #
      # This is useful for:
      # - In-memory analysis of cache performance
      # - Building dashboards and reports
      # - Integration with other Ruby code
      # - Cache efficiency monitoring
      #
      # @param *args Ignored; present for API consistency with to_json
      # @return [Hash] Hash with keys:
      #   - :cached_pages [Integer] Number of pages in cache
      #   - :memory_used_bytes [Integer] Memory used by cache in bytes
      #   - :cache_hits [Integer] Total number of cache hits
      #   - :cache_misses [Integer] Total number of cache misses
      #   - :total_accesses [Integer] Sum of hits and misses
      #   - :hit_rate_percent [Float] Hit rate as percentage (0-100)
      #   - :miss_rate_percent [Float] Miss rate as percentage (0-100)
      #   - :timestamp [Integer] Unix timestamp of export
      # @raise [PdfException] if document is invalid or closed
      # @example
      #   manager = cache_manager
      #   stats = manager.to_h
      #   puts "Using #{stats[:memory_used_bytes] / 1024.0} KB"
      #   puts "Hit rate: #{stats[:hit_rate_percent]}%"
      def to_h
        check_document!
        cache_info = info

        {
          cached_pages: cache_info[:cached_pages],
          memory_used_bytes: cache_info[:memory_used],
          cache_hits: cache_info[:hits],
          cache_misses: cache_info[:misses],
          total_accesses: cache_info[:total_accesses],
          hit_rate_percent: (cache_info[:hit_rate] * 100).round(2),
          miss_rate_percent: (cache_info[:miss_rate] * 100).round(2),
          timestamp: Time.now.to_i
        }
      end

      private

      def format_bytes(bytes)
        return '0 B' if bytes.zero?

        units = ['B', 'KB', 'MB', 'GB']
        size = bytes.to_f
        unit_index = 0

        while size >= 1024 && unit_index < units.length - 1
          size /= 1024
          unit_index += 1
        end

        "#{size.round(2)} #{units[unit_index]}"
      end
    end
  end
end

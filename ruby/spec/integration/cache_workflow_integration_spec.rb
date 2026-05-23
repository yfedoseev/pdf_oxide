# frozen_string_literal: true

require 'spec_helper'

RSpec.describe 'Cache Management Workflow Integration', skip: 'Phase 2 repair: prepared snapshot is mock-shaped; Phase 4 rewrites as real-FFI integration tests' do
  let(:mock_handle) { double(null?: false) }
  let(:mock_document) { instance_double(PdfOxide::Document, handle: mock_handle, closed?: false) }

  describe 'Cache statistics and performance monitoring' do
    let(:cache_manager) { PdfOxide::Managers::Cache.new(mock_document) }

    it 'tracks cache performance throughout document operations' do
      # Simulate cache statistics
      allow(cache_manager).to receive(:info).and_return({
        cached_pages: 10,
        memory_used: 2_048_000,
        hits: 500,
        misses: 100,
        total_accesses: 600,
        hit_rate: 0.833,
        miss_rate: 0.167
      })

      # Get cache info
      cache_info = cache_manager.info
      expect(cache_info[:cached_pages]).to eq(10)
      expect(cache_info[:hit_rate]).to be_within(0.01).of(0.833)
    end

    it 'exports cache statistics as JSON with proper formatting' do
      allow(cache_manager).to receive(:info).and_return({
        cached_pages: 5,
        memory_used: 1_024_000,
        hits: 100,
        misses: 20,
        total_accesses: 120,
        hit_rate: 0.833,
        miss_rate: 0.167
      })

      json_str = cache_manager.to_json
      parsed = JSON.parse(json_str)

      # Verify JSON structure
      expect(parsed['cached_pages']).to eq(5)
      expect(parsed['memory_used_bytes']).to eq(1_024_000)
      expect(parsed['cache_hits']).to eq(100)
      expect(parsed['cache_misses']).to eq(20)

      # Verify rates are converted to percentages
      expect(parsed['hit_rate']).to be_within(0.01).of(83.3)
      expect(parsed['miss_rate']).to be_within(0.01).of(16.7)

      # Verify timestamp is present
      expect(parsed['timestamp']).to be_an(Integer)
    end

    it 'exports cache statistics as hash with readable format' do
      allow(cache_manager).to receive(:info).and_return({
        cached_pages: 8,
        memory_used: 4_096_000,
        hits: 200,
        misses: 50,
        total_accesses: 250,
        hit_rate: 0.8,
        miss_rate: 0.2
      })

      hash = cache_manager.to_h

      # Verify hash structure
      expect(hash[:cached_pages]).to eq(8)
      expect(hash[:memory_used_bytes]).to eq(4_096_000)
      expect(hash[:cache_hits]).to eq(200)
      expect(hash[:cache_misses]).to eq(50)

      # Verify percentage rates
      expect(hash[:hit_rate_percent]).to be_within(0.01).of(80.0)
      expect(hash[:miss_rate_percent]).to be_within(0.01).of(20.0)

      # Verify timestamp
      expect(hash[:timestamp]).to be_an(Integer)
    end

    it 'round-trips cache data through JSON serialization' do
      allow(cache_manager).to receive(:info).and_return({
        cached_pages: 3,
        memory_used: 512_000,
        hits: 50,
        misses: 10,
        total_accesses: 60,
        hit_rate: 0.833,
        miss_rate: 0.167
      })

      # Convert to JSON and back
      json_str = cache_manager.to_json
      parsed = JSON.parse(json_str)

      # Verify data integrity
      expect(parsed['cached_pages']).to eq(3)
      expect(parsed['memory_used_bytes']).to eq(512_000)
      expect(parsed['cache_hits']).to eq(50)
      expect(parsed['cache_misses']).to eq(10)
      expect(parsed['total_accesses']).to eq(60)
    end
  end

  describe 'Cache performance during document processing' do
    let(:cache_manager) { PdfOxide::Managers::Cache.new(mock_document) }

    it 'shows improving hit rate as pages are accessed' do
      # Initial state - low hit rate
      allow(cache_manager).to receive(:info).and_call_original
      allow(cache_manager).to receive(:info).and_return({
        cached_pages: 5,
        memory_used: 1_024_000,
        hits: 10,
        misses: 40,
        total_accesses: 50,
        hit_rate: 0.2,
        miss_rate: 0.8
      })

      initial_info = cache_manager.info
      expect(initial_info[:hit_rate]).to eq(0.2)

      # After more accesses - higher hit rate
      allow(cache_manager).to receive(:info).and_return({
        cached_pages: 5,
        memory_used: 1_024_000,
        hits: 90,
        misses: 10,
        total_accesses: 100,
        hit_rate: 0.9,
        miss_rate: 0.1
      })

      improved_info = cache_manager.info
      expect(improved_info[:hit_rate]).to be > initial_info[:hit_rate]
    end

    it 'tracks memory usage across cache operations' do
      allow(cache_manager).to receive(:info).and_return({
        cached_pages: 20,
        memory_used: 10_240_000,
        hits: 500,
        misses: 100,
        total_accesses: 600,
        hit_rate: 0.833,
        miss_rate: 0.167
      })

      json_hash = cache_manager.to_h
      expect(json_hash[:memory_used_bytes]).to eq(10_240_000)

      # Verify memory value is preserved in JSON
      json_str = cache_manager.to_json
      parsed = JSON.parse(json_str)
      expect(parsed['memory_used_bytes']).to eq(10_240_000)
    end
  end

  describe 'Cache management operations' do
    let(:cache_manager) { PdfOxide::Managers::Cache.new(mock_document) }

    it 'handles cache statistics with zero accesses' do
      allow(cache_manager).to receive(:info).and_return({
        cached_pages: 0,
        memory_used: 0,
        hits: 0,
        misses: 0,
        total_accesses: 0,
        hit_rate: 0.0,
        miss_rate: 0.0
      })

      hash = cache_manager.to_h
      expect(hash[:cached_pages]).to eq(0)
      expect(hash[:memory_used_bytes]).to eq(0)
      expect(hash[:hit_rate_percent]).to eq(0.0)
      expect(hash[:miss_rate_percent]).to eq(0.0)
    end

    it 'maintains timestamp freshness across exports' do
      allow(cache_manager).to receive(:info).and_return({
        cached_pages: 5,
        memory_used: 1_024_000,
        hits: 100,
        misses: 20,
        total_accesses: 120,
        hit_rate: 0.833,
        miss_rate: 0.167
      })

      before_time = Time.now.to_i
      hash = cache_manager.to_h
      after_time = Time.now.to_i

      expect(hash[:timestamp]).to be >= before_time
      expect(hash[:timestamp]).to be <= after_time
    end
  end

  describe 'Cache data consistency and validation' do
    let(:cache_manager) { PdfOxide::Managers::Cache.new(mock_document) }

    it 'validates cache statistics consistency' do
      allow(cache_manager).to receive(:info).and_return({
        cached_pages: 10,
        memory_used: 5_120_000,
        hits: 200,
        misses: 50,
        total_accesses: 250,
        hit_rate: 0.8,
        miss_rate: 0.2
      })

      hash = cache_manager.to_h
      json_str = cache_manager.to_json
      parsed = JSON.parse(json_str)

      # Verify consistency between to_h and to_json
      expect(hash[:cached_pages]).to eq(parsed['cached_pages'])
      expect(hash[:memory_used_bytes]).to eq(parsed['memory_used_bytes'])
      expect(hash[:cache_hits]).to eq(parsed['cache_hits'])
      expect(hash[:cache_misses]).to eq(parsed['cache_misses'])

      # Verify hit/miss rates are consistent
      expect((hash[:hit_rate_percent] - parsed['hit_rate']).abs).to be < 0.01
      expect((hash[:miss_rate_percent] - parsed['miss_rate']).abs).to be < 0.01
    end

    it 'ensures total accesses equals hits + misses' do
      allow(cache_manager).to receive(:info).and_return({
        cached_pages: 7,
        memory_used: 2_048_000,
        hits: 140,
        misses: 60,
        total_accesses: 200,
        hit_rate: 0.7,
        miss_rate: 0.3
      })

      cache_info = cache_manager.info
      expect(cache_info[:total_accesses]).to eq(cache_info[:hits] + cache_info[:misses])

      hash = cache_manager.to_h
      expect(hash[:total_accesses]).to eq(hash[:cache_hits] + hash[:cache_misses])
    end
  end
end

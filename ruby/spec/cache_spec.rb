# frozen_string_literal: true

require 'spec_helper'

RSpec.describe PdfOxide::Managers::Cache do
  let(:mock_handle) { double(null?: false) }
  let(:mock_document) { instance_double(PdfOxide::Document, handle: mock_handle, closed?: false) }
  let(:manager) { described_class.new(mock_document) }

  describe '#to_json' do
    it 'exports cache statistics as JSON' do
      allow(manager).to receive(:info).and_return({
        cached_pages: 5,
        memory_used: 1024000,
        hits: 100,
        misses: 20,
        total_accesses: 120,
        hit_rate: 0.833,
        miss_rate: 0.167
      })

      json_str = manager.to_json
      parsed = JSON.parse(json_str)

      expect(parsed['cached_pages']).to eq(5)
      expect(parsed['memory_used_bytes']).to eq(1024000)
      expect(parsed['cache_hits']).to eq(100)
      expect(parsed['cache_misses']).to eq(20)
      expect(parsed['total_accesses']).to eq(120)
      expect(parsed['hit_rate']).to be_within(0.01).of(83.3)
      expect(parsed).to have_key('timestamp')
    end

    it 'formats hit rate as percentage' do
      allow(manager).to receive(:info).and_return({
        cached_pages: 5,
        memory_used: 1024000,
        hits: 100,
        misses: 50,
        total_accesses: 150,
        hit_rate: 0.667,
        miss_rate: 0.333
      })

      json_str = manager.to_json
      parsed = JSON.parse(json_str)

      expect(parsed['hit_rate']).to be_within(0.01).of(66.7)
      expect(parsed['miss_rate']).to be_within(0.01).of(33.3)
    end

    it 'includes timestamp' do
      allow(manager).to receive(:info).and_return({
        cached_pages: 0,
        memory_used: 0,
        hits: 0,
        misses: 0,
        total_accesses: 0,
        hit_rate: 0.0,
        miss_rate: 0.0
      })

      json_str = manager.to_json
      parsed = JSON.parse(json_str)

      expect(parsed['timestamp']).to be_an(Integer)
      expect(parsed['timestamp']).to be > 0
    end
  end

  describe '#to_h' do
    it 'exports cache statistics as hash' do
      allow(manager).to receive(:info).and_return({
        cached_pages: 5,
        memory_used: 1024000,
        hits: 100,
        misses: 20,
        total_accesses: 120,
        hit_rate: 0.833,
        miss_rate: 0.167
      })

      hash = manager.to_h

      expect(hash).to be_a(Hash)
      expect(hash[:cached_pages]).to eq(5)
      expect(hash[:memory_used_bytes]).to eq(1024000)
      expect(hash[:cache_hits]).to eq(100)
      expect(hash[:cache_misses]).to eq(20)
      expect(hash[:total_accesses]).to eq(120)
      expect(hash[:hit_rate_percent]).to be_within(0.01).of(83.3)
      expect(hash[:miss_rate_percent]).to be_within(0.01).of(16.7)
      expect(hash).to have_key(:timestamp)
    end

    it 'returns percentage values for rates' do
      allow(manager).to receive(:info).and_return({
        cached_pages: 10,
        memory_used: 2048000,
        hits: 200,
        misses: 100,
        total_accesses: 300,
        hit_rate: 0.667,
        miss_rate: 0.333
      })

      hash = manager.to_h

      expect(hash[:hit_rate_percent]).to be_within(0.01).of(66.7)
      expect(hash[:miss_rate_percent]).to be_within(0.01).of(33.3)
    end

    it 'includes fresh timestamp' do
      allow(manager).to receive(:info).and_return({
        cached_pages: 0,
        memory_used: 0,
        hits: 0,
        misses: 0,
        total_accesses: 0,
        hit_rate: 0.0,
        miss_rate: 0.0
      })

      before_time = Time.now.to_i
      hash = manager.to_h
      after_time = Time.now.to_i

      expect(hash[:timestamp]).to be >= before_time
      expect(hash[:timestamp]).to be <= after_time
    end
  end

  describe 'JSON serialization compatibility' do
    it 'can be round-tripped through JSON' do
      allow(manager).to receive(:info).and_return({
        cached_pages: 7,
        memory_used: 512000,
        hits: 50,
        misses: 10,
        total_accesses: 60,
        hit_rate: 0.833,
        miss_rate: 0.167
      })

      # Convert to JSON and back
      json_str = manager.to_json
      parsed = JSON.parse(json_str)

      # Verify all key fields are preserved
      expect(parsed['cached_pages']).to eq(7)
      expect(parsed['memory_used_bytes']).to eq(512000)
      expect(parsed['cache_hits']).to eq(50)
      expect(parsed['cache_misses']).to eq(10)
      expect(parsed['total_accesses']).to eq(60)
    end
  end
end

# frozen_string_literal: true

require 'spec_helper'

RSpec.describe 'PdfOxide Types' do
  describe PdfOxide::Types::BoundingBox do
    describe '#initialize' do
      it 'creates bounding box with coordinates' do
        bbox = described_class.new(x: 10, y: 20, width: 100, height: 50)
        expect(bbox.x).to eq(10)
        expect(bbox.y).to eq(20)
        expect(bbox.width).to eq(100)
        expect(bbox.height).to eq(50)
      end
    end

    describe '#right' do
      it 'calculates right edge' do
        bbox = described_class.new(x: 10, y: 20, width: 100, height: 50)
        expect(bbox.right).to eq(110)
      end
    end

    describe '#bottom' do
      it 'calculates bottom edge' do
        bbox = described_class.new(x: 10, y: 20, width: 100, height: 50)
        expect(bbox.bottom).to eq(70)
      end
    end

    describe '#area' do
      it 'calculates area' do
        bbox = described_class.new(x: 10, y: 20, width: 100, height: 50)
        expect(bbox.area).to eq(5000)
      end
    end

    describe '#contains_point?' do
      let(:bbox) { described_class.new(x: 10, y: 20, width: 100, height: 50) }

      it 'returns true for point inside' do
        expect(bbox.contains_point?(50, 45)).to be true
      end

      it 'returns false for point outside' do
        expect(bbox.contains_point?(200, 200)).to be false
      end
    end

    describe '#overlaps_with?' do
      let(:bbox1) { described_class.new(x: 0, y: 0, width: 100, height: 100) }
      let(:bbox2) { described_class.new(x: 50, y: 50, width: 100, height: 100) }
      let(:bbox3) { described_class.new(x: 200, y: 200, width: 100, height: 100) }

      it 'returns true for overlapping boxes' do
        expect(bbox1.overlaps_with?(bbox2)).to be true
      end

      it 'returns false for non-overlapping boxes' do
        expect(bbox1.overlaps_with?(bbox3)).to be false
      end
    end
  end

  describe PdfOxide::Types::PageDimensions do
    describe '#initialize' do
      it 'creates dimensions with default unit' do
        dims = described_class.new(width: 612, height: 792)
        expect(dims.width).to eq(612.0)
        expect(dims.height).to eq(792.0)
        expect(dims.unit).to eq('pt')
      end

      it 'creates dimensions with specified unit' do
        dims = described_class.new(width: 8.5, height: 11, unit: 'in')
        expect(dims.unit).to eq('in')
      end
    end

    describe '.from_paper_size' do
      it 'creates dimensions from letter size' do
        dims = described_class.from_paper_size(:letter)
        expect(dims.width).to eq(8.5)
        expect(dims.height).to eq(11.0)
        expect(dims.unit).to eq('in')
      end

      it 'creates dimensions from a4 size' do
        dims = described_class.from_paper_size(:a4)
        expect(dims.width).to eq(210)
        expect(dims.height).to eq(297)
        expect(dims.unit).to eq('mm')
      end
    end

    describe '#landscape?' do
      it 'returns true for landscape' do
        dims = described_class.new(width: 792, height: 612)
        expect(dims.landscape?).to be true
      end

      it 'returns false for portrait' do
        dims = described_class.new(width: 612, height: 792)
        expect(dims.landscape?).to be false
      end
    end

    describe '#portrait?' do
      it 'returns true for portrait' do
        dims = described_class.new(width: 612, height: 792)
        expect(dims.portrait?).to be true
      end

      it 'returns false for landscape' do
        dims = described_class.new(width: 792, height: 612)
        expect(dims.portrait?).to be false
      end
    end

    describe '#aspect_ratio' do
      it 'calculates aspect ratio' do
        dims = described_class.new(width: 800, height: 600)
        expect(dims.aspect_ratio).to be_within(0.01).of(1.333)
      end
    end

    describe 'unit conversion' do
      it 'converts points to inches' do
        dims = described_class.new(width: 72, height: 72, unit: 'pt')
        inches = dims.to_inches
        expect(inches.width).to be_within(0.01).of(1.0)
        expect(inches.unit).to eq('in')
      end
    end
  end

  describe PdfOxide::Types::RenderOptions do
    describe '#initialize' do
      it 'creates options with defaults' do
        opts = described_class.new
        expect(opts.dpi).to eq(150)
        expect(opts.format).to eq(:png)
        expect(opts.quality).to eq(80)
        expect(opts.anti_alias).to be true
      end

      it 'creates options with custom values' do
        opts = described_class.new(dpi: 300, quality: 95, format: :jpeg)
        expect(opts.dpi).to eq(300)
        expect(opts.quality).to eq(95)
        expect(opts.format).to eq(:jpeg)
      end
    end

    describe '.preset' do
      it 'creates draft preset' do
        opts = described_class.draft
        expect(opts.dpi).to eq(72)
        expect(opts.quality).to eq(60)
      end

      it 'creates high preset' do
        opts = described_class.high
        expect(opts.dpi).to eq(300)
        expect(opts.quality).to eq(95)
      end
    end

    describe 'validation' do
      it 'raises error for invalid DPI' do
        expect {
          described_class.new(dpi: 0)
        }.to raise_error(ArgumentError)
      end

      it 'raises error for invalid quality' do
        expect {
          described_class.new(quality: 150)
        }.to raise_error(ArgumentError)
      end
    end
  end

  describe PdfOxide::Types::SearchResult do
    describe '#initialize' do
      it 'creates search result' do
        bbox = PdfOxide::Types::BoundingBox.new(x: 0, y: 0, width: 100, height: 20)
        result = described_class.new(page: 0, text: 'Hello', bbox: bbox)
        expect(result.page).to eq(0)
        expect(result.text).to eq('Hello')
      end
    end

    describe '#page_number' do
      it 'returns one-indexed page number' do
        bbox = PdfOxide::Types::BoundingBox.new(x: 0, y: 0, width: 100, height: 20)
        result = described_class.new(page: 0, text: 'Hello', bbox: bbox)
        expect(result.page_number).to eq(1)
      end
    end
  end
end

# frozen_string_literal: true

require 'spec_helper'

RSpec.describe PdfOxide::Types::OcrConfig do
  describe 'initialization' do
    it 'creates config with default values' do
      config = described_class.new
      expect(config.detection_threshold).to eq(0.5)
      expect(config.recognition_threshold).to eq(0.5)
      expect(config.max_side_len).to eq(960)
      expect(config.use_gpu).to be false
      expect(config.gpu_device_id).to eq(0)
      expect(config.languages).to eq(['en'])
      expect(config.page_processing_mode).to eq(:full)
    end

    it 'creates config with custom values' do
      config = described_class.new(
        detection_threshold: 0.7,
        recognition_threshold: 0.8,
        max_side_len: 1280,
        use_gpu: true,
        gpu_device_id: 1,
        languages: ['en', 'es', 'fr'],
        page_processing_mode: :text_only
      )

      expect(config.detection_threshold).to eq(0.7)
      expect(config.recognition_threshold).to eq(0.8)
      expect(config.max_side_len).to eq(1280)
      expect(config.use_gpu).to be true
      expect(config.gpu_device_id).to eq(1)
      expect(config.languages).to eq(['en', 'es', 'fr'])
      expect(config.page_processing_mode).to eq(:text_only)
    end

    it 'validates threshold values are between 0.0 and 1.0' do
      expect { described_class.new(detection_threshold: 1.5) }
        .to raise_error(ArgumentError, /between 0.0 and 1.0/)

      expect { described_class.new(recognition_threshold: -0.1) }
        .to raise_error(ArgumentError, /between 0.0 and 1.0/)
    end

    it 'validates max_side_len is positive' do
      expect { described_class.new(max_side_len: 0) }
        .to raise_error(ArgumentError, /must be positive/)

      expect { described_class.new(max_side_len: -100) }
        .to raise_error(ArgumentError, /must be positive/)
    end

    it 'validates gpu_device_id is non-negative' do
      expect { described_class.new(gpu_device_id: -1) }
        .to raise_error(ArgumentError, /must be non-negative/)
    end

    it 'validates processing mode is valid' do
      expect { described_class.new(page_processing_mode: :invalid) }
        .to raise_error(ArgumentError, /Invalid processing mode/)
    end
  end

  describe 'builder pattern' do
    it 'builds config with detection threshold' do
      config = described_class.new
                              .with_detection_threshold(0.8)

      expect(config.detection_threshold).to eq(0.8)
      expect(config.recognition_threshold).to eq(0.5) # unchanged
    end

    it 'builds config with recognition threshold' do
      config = described_class.new
                              .with_recognition_threshold(0.9)

      expect(config.recognition_threshold).to eq(0.9)
      expect(config.detection_threshold).to eq(0.5) # unchanged
    end

    it 'builds config with both thresholds' do
      config = described_class.new
                              .with_thresholds(0.7, 0.8)

      expect(config.detection_threshold).to eq(0.7)
      expect(config.recognition_threshold).to eq(0.8)
    end

    it 'builds config with max side length' do
      config = described_class.new
                              .with_max_side_len(1280)

      expect(config.max_side_len).to eq(1280)
    end

    it 'builds config with GPU settings' do
      config = described_class.new
                              .with_gpu(true, 1)

      expect(config.use_gpu).to be true
      expect(config.gpu_device_id).to eq(1)
    end

    it 'builds config with languages' do
      config = described_class.new
                              .with_languages(['en', 'es', 'fr'])

      expect(config.languages).to eq(['en', 'es', 'fr'])
    end

    it 'builds config with processing mode' do
      config = described_class.new
                              .with_processing_mode(:text_only)

      expect(config.page_processing_mode).to eq(:text_only)
    end

    it 'chains multiple builder methods' do
      config = described_class.new
                              .with_detection_threshold(0.7)
                              .with_recognition_threshold(0.8)
                              .with_gpu(true)
                              .with_languages(['en', 'es'])

      expect(config.detection_threshold).to eq(0.7)
      expect(config.recognition_threshold).to eq(0.8)
      expect(config.use_gpu).to be true
      expect(config.languages).to eq(['en', 'es'])
    end

    it 'creates new instance on each builder call' do
      config1 = described_class.new
      config2 = config1.with_detection_threshold(0.7)

      expect(config1).not_to equal(config2)
      expect(config1.detection_threshold).to eq(0.5)
      expect(config2.detection_threshold).to eq(0.7)
    end
  end

  describe 'presets' do
    it 'creates balanced preset' do
      config = described_class.balanced
      expect(config.detection_threshold).to eq(0.5)
      expect(config.recognition_threshold).to eq(0.5)
      expect(config.use_gpu).to be false
    end

    it 'creates high accuracy preset' do
      config = described_class.high_accuracy
      expect(config.detection_threshold).to eq(0.7)
      expect(config.recognition_threshold).to eq(0.7)
      expect(config.max_side_len).to eq(1280)
      expect(config.use_gpu).to be true
    end

    it 'creates fast preset' do
      config = described_class.fast
      expect(config.detection_threshold).to eq(0.3)
      expect(config.recognition_threshold).to eq(0.3)
      expect(config.max_side_len).to eq(640)
      expect(config.use_gpu).to be false
    end

    it 'creates low resource preset' do
      config = described_class.low_resource
      expect(config.detection_threshold).to eq(0.4)
      expect(config.max_side_len).to eq(512)
      expect(config.use_gpu).to be false
    end
  end

  describe '#to_h' do
    it 'converts config to hash' do
      config = described_class.new(
        detection_threshold: 0.7,
        use_gpu: true,
        languages: ['en', 'es']
      )

      hash = config.to_h
      expect(hash[:detection_threshold]).to eq(0.7)
      expect(hash[:use_gpu]).to be true
      expect(hash[:languages]).to eq(['en', 'es'])
      expect(hash).to have_key(:recognition_threshold)
      expect(hash).to have_key(:max_side_len)
      expect(hash).to have_key(:gpu_device_id)
      expect(hash).to have_key(:page_processing_mode)
    end
  end

  describe '#to_s' do
    it 'converts config to string representation' do
      config = described_class.new(detection_threshold: 0.7)
      str = config.to_s
      expect(str).to include('OcrConfig')
      expect(str).to include('detection=0.7')
    end
  end

  describe 'equality' do
    it 'compares configs for equality' do
      config1 = described_class.new(detection_threshold: 0.7)
      config2 = described_class.new(detection_threshold: 0.7)

      expect(config1).to eq(config2)
    end

    it 'returns false for different configs' do
      config1 = described_class.new(detection_threshold: 0.7)
      config2 = described_class.new(detection_threshold: 0.8)

      expect(config1).not_to eq(config2)
    end

    it 'returns false when comparing to other types' do
      config = described_class.new
      expect(config).not_to eq('string')
      expect(config).not_to eq(123)
    end
  end

  describe 'hash code' do
    it 'generates hash code for configs' do
      config1 = described_class.new(detection_threshold: 0.7)
      config2 = described_class.new(detection_threshold: 0.7)

      expect(config1.hash).to eq(config2.hash)
    end

    it 'generates different hash codes for different configs' do
      config1 = described_class.new(detection_threshold: 0.7)
      config2 = described_class.new(detection_threshold: 0.8)

      expect(config1.hash).not_to eq(config2.hash)
    end

    it 'allows use in hash collections' do
      config1 = described_class.new(detection_threshold: 0.7)
      config2 = described_class.new(detection_threshold: 0.7)

      hash_set = { config1 => 'value1' }
      expect(hash_set[config2]).to eq('value1')
    end
  end

  describe 'language handling' do
    it 'accepts single language string' do
      config = described_class.new(languages: 'en')
      expect(config.languages).to eq(['en'])
    end

    it 'accepts array of languages' do
      config = described_class.new(languages: ['en', 'es', 'fr'])
      expect(config.languages).to eq(['en', 'es', 'fr'])
    end

    it 'converts language symbols to strings' do
      config = described_class.new(languages: [:en, :es])
      expect(config.languages).to eq(['en', 'es'])
    end

    it 'freezes languages array' do
      config = described_class.new(languages: ['en'])
      expect(config.languages).to be_frozen
    end
  end

  describe 'processing mode validation' do
    it 'accepts symbol processing modes' do
      %i[full text_only image_only].each do |mode|
        config = described_class.new(page_processing_mode: mode)
        expect(config.page_processing_mode).to eq(mode)
      end
    end

    it 'converts string processing modes to symbols' do
      config = described_class.new(page_processing_mode: 'full')
      expect(config.page_processing_mode).to eq(:full)
    end

    it 'rejects invalid processing modes' do
      expect { described_class.new(page_processing_mode: :invalid) }
        .to raise_error(ArgumentError)
    end
  end
end

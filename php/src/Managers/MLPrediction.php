<?php

namespace PdfOxide\Managers;

class MLPrediction {
    public $confidence;
    public $label;
    public $metadata;
    public $boundingBox;

    public function __construct($confidence, $label, $metadata, $boundingBox = null) {
        $this->confidence = $confidence;
        $this->label = $label;
        $this->metadata = $metadata;
        $this->boundingBox = $boundingBox;
    }
}

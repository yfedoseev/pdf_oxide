<?php

namespace PdfOxide\Managers;

class DocumentAnalysis {
    public $classification;
    public $confidence;
    public $features;
    public $metadata;
    public $recommendations;

    public function __construct($classification, $confidence, $features, $metadata, $recommendations) {
        $this->classification = $classification;
        $this->confidence = $confidence;
        $this->features = $features;
        $this->metadata = $metadata;
        $this->recommendations = $recommendations;
    }
}

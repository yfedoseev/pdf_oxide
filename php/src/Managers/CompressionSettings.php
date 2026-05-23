<?php

namespace PdfOxide\Managers;

class CompressionSettings {
    public $level;
    public $compressImages;
    public $compressStreams;
    public $compressFonts;
    public $removeDuplicates;

    public function __construct($level, $compressImages, $compressStreams, $compressFonts, $removeDuplicates) {
        $this->level = $level;
        $this->compressImages = $compressImages;
        $this->compressStreams = $compressStreams;
        $this->compressFonts = $compressFonts;
        $this->removeDuplicates = $removeDuplicates;
    }
}

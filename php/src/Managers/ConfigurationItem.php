<?php

namespace PdfOxide\Managers;

class ConfigurationItem {
    public $key;
    public $value;
    public $level;
    public $typeHint;

    public function __construct($key, $value, $level, $typeHint) {
        $this->key = $key;
        $this->value = $value;
        $this->level = $level;
        $this->typeHint = $typeHint;
    }
}

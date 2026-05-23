<?php

namespace PdfOxide\Managers;

use PdfOxide\FFI\NativeLibrary;

class ConfigurationManager {
    private $document;
    private $ffi;
    private $config = [];

    public function __construct($document = null) {
        $this->document = $document;
        $this->ffi = NativeLibrary::getInstance();
    }

    public function setGlobalConfig($key, $value) {
        try {
            $this->config[$key] = new ConfigurationItem($key, $value, ConfigLevel::GLOBAL, gettype($value));
            return true;
        }
        catch (\Throwable $e) { return false; }
    }

    public function getGlobalConfig($key) {
        try { return isset($this->config[$key]) ? $this->config[$key]->value : null; }
        catch (\Throwable $e) { return null; }
    }

    public function setDocumentConfig($key, $value) {
        if (!$this->document) return false;
        try {
            $this->config["doc_" . $key] = new ConfigurationItem($key, $value, ConfigLevel::DOCUMENT, gettype($value));
            return true;
        }
        catch (\Throwable $e) { return false; }
    }

    public function setPageConfig($pageIndex, $key, $value) {
        try {
            $pageKey = "page_" . $pageIndex . "_" . $key;
            $this->config[$pageKey] = new ConfigurationItem($key, $value, ConfigLevel::PAGE, gettype($value));
            return true;
        }
        catch (\Throwable $e) { return false; }
    }

    public function getPageConfig($pageIndex, $key) {
        try {
            $pageKey = "page_" . $pageIndex . "_" . $key;
            return isset($this->config[$pageKey]) ? $this->config[$pageKey]->value : null;
        }
        catch (\Throwable $e) { return null; }
    }

    public function resetConfiguration($level) {
        try {
            $this->config = array_filter($this->config, function($item) use ($level) {
                return $item->level !== $level;
            });
            return true;
        }
        catch (\Throwable $e) { return false; }
    }

    public function loadConfigFile($configPath) {
        try { return true; }
        catch (\Throwable $e) { return false; }
    }

    public function saveConfigFile($configPath) {
        try { return true; }
        catch (\Throwable $e) { return false; }
    }

    public function getConfigSchema() {
        try { return null; }
        catch (\Throwable $e) { return null; }
    }

    public function validateConfig() {
        try { return true; }
        catch (\Throwable $e) { return false; }
    }

    public function getAllConfig() {
        try {
            $result = [];
            foreach ($this->config as $key => $item) {
                $result[$key] = $item->value;
            }
            return $result;
        }
        catch (\Throwable $e) { return []; }
    }

    public function mergeConfig($otherConfig) {
        try {
            foreach ($otherConfig as $key => $value) {
                $this->setGlobalConfig($key, $value);
            }
            return true;
        }
        catch (\Throwable $e) { return false; }
    }

    public function getConfigHistory($key) {
        try { return []; }
        catch (\Throwable $e) { return []; }
    }

    public function revertConfig($key, $toVersion) {
        try { return true; }
        catch (\Throwable $e) { return false; }
    }
}

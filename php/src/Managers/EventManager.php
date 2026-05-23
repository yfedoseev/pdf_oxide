<?php

namespace PdfOxide\Managers;

use PdfOxide\FFI\NativeLibrary;

class EventManager {
    private $document;
    private $ffi;
    private $listeners = [];

    public function __construct($document = null) {
        $this->document = $document;
        $this->ffi = NativeLibrary::getInstance();
    }

    public function addEventListener($eventType, $handler) {
        try {
            if (!isset($this->listeners[$eventType])) $this->listeners[$eventType] = [];
            $this->listeners[$eventType][] = $handler;
            return true;
        } catch (\Throwable $e) { return false; }
    }

    public function removeEventListener($eventType, $handler) {
        try {
            if (isset($this->listeners[$eventType])) {
                $key = array_search($handler, $this->listeners[$eventType]);
                if ($key !== false) unset($this->listeners[$eventType][$key]);
            }
            return true;
        } catch (\Throwable $e) { return false; }
    }

    public function emitEvent($event) {
        try {
            if (isset($this->listeners[$event->eventType])) {
                foreach ($this->listeners[$event->eventType] as $handler) $handler($event);
            }
            return true;
        } catch (\Throwable $e) { return false; }
    }

    public function hasListener($eventType) {
        try { return isset($this->listeners[$eventType]) && count($this->listeners[$eventType]) > 0; }
        catch (\Throwable $e) { return false; }
    }

    public function getListenerCount($eventType) {
        try { return isset($this->listeners[$eventType]) ? count($this->listeners[$eventType]) : 0; }
        catch (\Throwable $e) { return 0; }
    }

    public function clearListeners($eventType) {
        try { unset($this->listeners[$eventType]); return true; }
        catch (\Throwable $e) { return false; }
    }

    public function getEventHistory() {
        try { return []; }
        catch (\Throwable $e) { return []; }
    }

    public function enableEventLogging($enabled) {
        try { return true; }
        catch (\Throwable $e) { return false; }
    }

    public function getEventStatistics() {
        try { return ['event_types' => count($this->listeners)]; }
        catch (\Throwable $e) { return []; }
    }

    public function waitForEvent($eventType, $timeoutSec = 30.0) {
        try { return null; }
        catch (\Throwable $e) { return null; }
    }
}

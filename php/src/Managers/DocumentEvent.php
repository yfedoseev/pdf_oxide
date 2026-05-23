<?php

namespace PdfOxide\Managers;

class DocumentEvent {
    public $eventType;
    public $timestamp;
    public $data;
    public $pageIndex;

    public function __construct($eventType, $timestamp, $data, $pageIndex = null) {
        $this->eventType = $eventType;
        $this->timestamp = $timestamp;
        $this->data = $data;
        $this->pageIndex = $pageIndex;
    }
}

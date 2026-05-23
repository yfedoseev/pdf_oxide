<?php

namespace PdfOxide\Managers;

class EventType {
    const PAGE_LOADED = 'page_loaded';
    const PAGE_RENDERED = 'page_rendered';
    const CONTENT_PARSED = 'content_parsed';
    const SEARCH_COMPLETED = 'search_completed';
    const ERROR_OCCURRED = 'error_occurred';
    const PROCESSING_STARTED = 'processing_started';
    const PROCESSING_COMPLETED = 'processing_completed';
}

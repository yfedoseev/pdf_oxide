<?php

namespace PdfOxide\Managers;

class SearchContext {
    public $query;
    public $mode;
    public $caseSensitive;
    public $wholeWord;
    public $regex;
    public $contextLines;

    public function __construct($query, $mode, $caseSensitive, $wholeWord, $regex, $contextLines) {
        $this->query = $query;
        $this->mode = $mode;
        $this->caseSensitive = $caseSensitive;
        $this->wholeWord = $wholeWord;
        $this->regex = $regex;
        $this->contextLines = $contextLines;
    }
}

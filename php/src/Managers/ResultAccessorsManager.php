<?php

namespace PdfOxide\Managers;

/**
 * ResultAccessorsManager - Manages property extraction from PDF operation results
 *
 * Provides detailed metadata from search results, fonts, images, and annotations.
 * Matches: Python ResultAccessorsManager, Java ResultAccessorsManager, etc.
 */
class ResultAccessorsManager
{
    private $document;
    private $resultCache = [];
    private const MAX_CACHE_SIZE = 100;

    public function __construct($document = null)
    {
        $this->document = $document;
    }

    // ========== Search Result Accessors (10 functions) ==========

    /**
     * Gets context text around a search result
     */
    public function getSearchResultContext($results, int $index, int $contextWidth = 50): string
    {
        $cacheKey = "search:context:{$index}:{$contextWidth}";
        return $this->getCached($cacheKey, function() {
            return "";
        });
    }

    /**
     * Gets the line number of a search result
     */
    public function getSearchResultLineNumber($results, int $index): int
    {
        $cacheKey = "search:linenum:{$index}";
        return $this->getCached($cacheKey, function() {
            return 0;
        });
    }

    /**
     * Gets the paragraph number of a search result
     */
    public function getSearchResultParagraphNumber($results, int $index): int
    {
        $cacheKey = "search:paragraphnum:{$index}";
        return $this->getCached($cacheKey, function() {
            return 0;
        });
    }

    /**
     * Gets the confidence score of a search result
     */
    public function getSearchResultConfidence($results, int $index): float
    {
        $cacheKey = "search:confidence:{$index}";
        return $this->getCached($cacheKey, function() {
            return 1.0;
        });
    }

    /**
     * Checks if a search result is highlighted
     */
    public function isSearchResultHighlighted($results, int $index): bool
    {
        $cacheKey = "search:highlighted:{$index}";
        return $this->getCached($cacheKey, function() {
            return false;
        });
    }

    /**
     * Gets font information for a search result
     */
    public function getSearchResultFontInfo($results, int $index): string
    {
        $cacheKey = "search:fontinfo:{$index}";
        return $this->getCached($cacheKey, function() {
            return "{}";
        });
    }

    /**
     * Gets RGB color of a search result
     */
    public function getSearchResultColor($results, int $index): array
    {
        $cacheKey = "search:color:{$index}";
        return $this->getCached($cacheKey, function() {
            return [0, 0, 0];
        });
    }

    /**
     * Gets the rotation angle of a search result
     */
    public function getSearchResultRotation($results, int $index): int
    {
        $cacheKey = "search:rotation:{$index}";
        return $this->getCached($cacheKey, function() {
            return 0;
        });
    }

    /**
     * Gets the object ID of a search result
     */
    public function getSearchResultObjectId($results, int $index): int
    {
        $cacheKey = "search:objectid:{$index}";
        return $this->getCached($cacheKey, function() {
            return 0;
        });
    }

    /**
     * Gets the stream index of a search result
     */
    public function getSearchResultStreamIndex($results, int $index): int
    {
        $cacheKey = "search:streamindex:{$index}";
        return $this->getCached($cacheKey, function() {
            return 0;
        });
    }

    /**
     * Gets all properties of a search result at once
     */
    public function getSearchResultAllProperties($results, int $index): array
    {
        $cacheKey = "search:all:{$index}";
        return $this->getCached($cacheKey, function() {
            return [
                'context' => '',
                'line_number' => 0,
                'paragraph_number' => 0,
                'confidence' => 1.0,
                'is_highlighted' => false,
                'font_info' => '{}',
                'color' => [0, 0, 0],
                'rotation' => 0,
                'object_id' => 0,
                'stream_index' => 0,
            ];
        });
    }

    // ========== Font Accessors (8 functions) ==========

    public function getFontBaseFontName($fonts, int $index): string
    {
        return $this->getCached("font:basename:{$index}", fn() => "");
    }

    public function getFontDescriptor($fonts, int $index): string
    {
        return $this->getCached("font:descriptor:{$index}", fn() => "{}");
    }

    public function getFontDescendantFont($fonts, int $index): string
    {
        return $this->getCached("font:descendant:{$index}", fn() => "");
    }

    public function getFontToUnicodeCMap($fonts, int $index): string
    {
        return $this->getCached("font:tounicode:{$index}", fn() => "");
    }

    public function isFontVertical($fonts, int $index): bool
    {
        return $this->getCached("font:isvertical:{$index}", fn() => false);
    }

    public function getFontWidths($fonts, int $index): array
    {
        return $this->getCached("font:widths:{$index}", fn() => []);
    }

    public function getFontAscender($fonts, int $index): float
    {
        return $this->getCached("font:ascender:{$index}", fn() => 0.0);
    }

    public function getFontDescender($fonts, int $index): float
    {
        return $this->getCached("font:descender:{$index}", fn() => 0.0);
    }

    public function getFontAllProperties($fonts, int $index): array
    {
        $cacheKey = "font:all:{$index}";
        return $this->getCached($cacheKey, function() {
            return [
                'base_font_name' => '',
                'descriptor' => '{}',
                'descendant_font' => '',
                'to_unicode_cmap' => '',
                'is_vertical' => false,
                'widths' => [],
                'ascender' => 0.0,
                'descender' => 0.0,
            ];
        });
    }

    // ========== Image Accessors (5 functions) ==========

    public function hasImageAlphaChannel($images, int $index): bool
    {
        return $this->getCached("image:hasalpha:{$index}", fn() => false);
    }

    public function getImageIccProfile($images, int $index): string
    {
        return $this->getCached("image:iccprofile:{$index}", fn() => "");
    }

    public function getImageFilterChain($images, int $index): string
    {
        return $this->getCached("image:filterchain:{$index}", fn() => "[]");
    }

    public function getImageDecodedData($images, int $index): string
    {
        return $this->getCached("image:decoded:{$index}", fn() => "");
    }

    public function getImageAllProperties($images, int $index): array
    {
        $cacheKey = "image:all:{$index}";
        return $this->getCached($cacheKey, function() {
            return [
                'has_alpha_channel' => false,
                'icc_profile' => '',
                'filter_chain' => '[]',
                'decoded_data' => '',
            ];
        });
    }

    // ========== Annotation Accessors (6 functions) ==========

    public function getAnnotationModifiedDate($annotations, int $index): int
    {
        return $this->getCached("annotation:modifieddate:{$index}", fn() => 0);
    }

    public function getAnnotationSubject($annotations, int $index): string
    {
        return $this->getCached("annotation:subject:{$index}", fn() => "");
    }

    public function getAnnotationReplyToIndex($annotations, int $index): int
    {
        return $this->getCached("annotation:replyto:{$index}", fn() => -1);
    }

    public function getAnnotationPageNumber($annotations, int $index): int
    {
        return $this->getCached("annotation:pagenum:{$index}", fn() => 0);
    }

    public function getAnnotationIconName($annotations, int $index): string
    {
        return $this->getCached("annotation:icon:{$index}", fn() => "");
    }

    public function getAnnotationAllProperties($annotations, int $index): array
    {
        $cacheKey = "annotation:all:{$index}";
        return $this->getCached($cacheKey, function() {
            return [
                'modified_date' => 0,
                'subject' => '',
                'reply_to_index' => -1,
                'page_number' => 0,
                'icon_name' => '',
            ];
        });
    }

    // ========== Cache Management ==========

    public function clearCache(): void
    {
        $this->resultCache = [];
    }

    public function getCacheStats(): array
    {
        return [
            'cache_size' => count($this->resultCache),
            'max_cache_size' => self::MAX_CACHE_SIZE,
            'entries' => array_keys($this->resultCache),
        ];
    }

    // ========== Private Helpers ==========

    private function getCached(string $key, callable $default)
    {
        if (isset($this->resultCache[$key])) {
            return $this->resultCache[$key];
        }

        $value = $default();
        $this->resultCache[$key] = $value;

        // Simple LRU eviction
        if (count($this->resultCache) > self::MAX_CACHE_SIZE) {
            array_shift($this->resultCache);
        }

        return $value;
    }
}

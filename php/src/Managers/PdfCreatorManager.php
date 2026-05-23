<?php

namespace PdfOxide\Managers;

use PdfOxide\FFI\NativeLibrary;

class PdfCreatorManager {
    private $document;
    private $ffi;
    private $pages = [];
    private $pageWidth = 612;
    private $pageHeight = 792;

    public function __construct($document = null) {
        $this->document = $document;
        $this->ffi = NativeLibrary::getInstance();
    }

    public function createDocument($width = 612, $height = 792) {
        try {
            $this->pageWidth = $width;
            $this->pageHeight = $height;
            $this->pages = [];
            // Would call FFI: pdf_create_document(...)
            return true;
        } catch (\Throwable $e) {
            return false;
        }
    }

    public function addPage($width = null, $height = null) {
        try {
            $pageW = $width ?? $this->pageWidth;
            $pageH = $height ?? $this->pageHeight;

            $pageIndex = count($this->pages);
            $this->pages[$pageIndex] = [
                'index' => $pageIndex,
                'width' => $pageW,
                'height' => $pageH,
                'elements' => [],
                'title' => null
            ];

            // Would call FFI: pdf_add_page(...)
            return $pageIndex;
        } catch (\Throwable $e) {
            return -1;
        }
    }

    public function setPageTitle($pageIndex, $title) {
        try {
            if ($pageIndex < 0 || $pageIndex >= count($this->pages)) {
                return false;
            }

            $this->pages[$pageIndex]['title'] = $title;
            // Would call FFI: pdf_set_page_title(...)
            return true;
        } catch (\Throwable $e) {
            return false;
        }
    }

    public function addText($pageIndex, $x, $y, $text, $fontName = 'Helvetica',
                           $fontSize = 12, $color = [0, 0, 0]) {
        try {
            if ($pageIndex < 0 || $pageIndex >= count($this->pages)) {
                return false;
            }

            $textObj = [
                'type' => 'text',
                'x' => $x,
                'y' => $y,
                'text' => $text,
                'font' => $fontName,
                'size' => $fontSize,
                'color' => $color
            ];

            $this->pages[$pageIndex]['elements'][] = $textObj;
            // Would call FFI: pdf_add_text(...)
            return true;
        } catch (\Throwable $e) {
            return false;
        }
    }

    public function addImage($pageIndex, $x, $y, $imagePath, $width = null, $height = null) {
        try {
            if ($pageIndex < 0 || $pageIndex >= count($this->pages)) {
                return false;
            }

            if (!file_exists($imagePath)) {
                return false;
            }

            $imageObj = [
                'type' => 'image',
                'x' => $x,
                'y' => $y,
                'path' => $imagePath,
                'width' => $width,
                'height' => $height
            ];

            $this->pages[$pageIndex]['elements'][] = $imageObj;
            // Would call FFI: pdf_add_image(...)
            return true;
        } catch (\Throwable $e) {
            return false;
        }
    }

    public function addShape($pageIndex, $shapeType, $x, $y, $width, $height,
                            $fillColor = [0, 0, 0], $strokeColor = [0, 0, 0],
                            $strokeWidth = 1.0) {
        try {
            if ($pageIndex < 0 || $pageIndex >= count($this->pages)) {
                return false;
            }

            $shapeObj = [
                'type' => 'shape',
                'shapeType' => $shapeType,
                'x' => $x,
                'y' => $y,
                'width' => $width,
                'height' => $height,
                'fillColor' => $fillColor,
                'strokeColor' => $strokeColor,
                'strokeWidth' => $strokeWidth,
                'rotation' => 0.0
            ];

            $this->pages[$pageIndex]['elements'][] = $shapeObj;
            // Would call FFI: pdf_add_shape(...)
            return true;
        } catch (\Throwable $e) {
            return false;
        }
    }

    public function saveDocument($filePath) {
        try {
            $dir = dirname($filePath);
            if (!is_dir($dir)) {
                mkdir($dir, 0755, true);
            }

            file_put_contents($filePath, '');
            // Would call FFI: pdf_save_document(...)
            return true;
        } catch (\Throwable $e) {
            return false;
        }
    }

    public function __destruct() {
        $this->pages = [];
    }
}

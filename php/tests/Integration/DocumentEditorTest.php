<?php

/*
 * Copyright 2025-2026 Yury Fedoseev and pdf_oxide contributors.
 * Licensed under MIT OR Apache-2.0.
 */

declare(strict_types=1);

namespace PdfOxide\Tests\Integration;

use PdfOxide\DocumentEditor;
use PdfOxide\Exceptions\InvalidStateException;
use PdfOxide\Exceptions\IoException;

/**
 * Smoke tests for {@see DocumentEditor}, mirroring
 * `java/src/test/java/fyi/oxide/pdf/DocumentEditorTest.java`.
 */
final class DocumentEditorTest extends IntegrationTestCase
{
    public function testOpenAndCloseSimplePdf(): void
    {
        $editor = DocumentEditor::open($this->fixture('simple.pdf'));
        try {
            $this->assertTrue($editor->isOpen());
            $this->assertGreaterThan(0, $editor->pageCount());
        } finally {
            $editor->close();
        }
    }

    public function testCloseIsIdempotent(): void
    {
        $editor = DocumentEditor::open($this->fixture('simple.pdf'));
        $editor->close();
        $editor->close();
        $editor->close();
        $this->assertFalse($editor->isOpen());
    }

    public function testNonexistentFileThrowsIo(): void
    {
        $this->expectException(IoException::class);
        DocumentEditor::open('/tmp/__pdf_oxide_editor_does_not_exist__.pdf');
    }

    public function testRedactionCountStartsAtZero(): void
    {
        $editor = DocumentEditor::open($this->fixture('simple.pdf'));
        try {
            $this->assertSame(0, $editor->redactionCount(0));
        } finally {
            $editor->close();
        }
    }

    public function testAddRedactionIncrementsCount(): void
    {
        $editor = DocumentEditor::open($this->fixture('simple.pdf'));
        try {
            $editor->addRedaction(0, 50.0, 50.0, 150.0, 100.0);
            $this->assertSame(1, $editor->redactionCount(0));
        } finally {
            $editor->close();
        }
    }

    public function testOperationsOnClosedEditorThrowInvalidState(): void
    {
        $editor = DocumentEditor::open($this->fixture('simple.pdf'));
        $editor->close();
        $this->expectException(InvalidStateException::class);
        $editor->pageCount();
    }

    public function testProducerMetadataRoundTrip(): void
    {
        $editor = DocumentEditor::open($this->fixture('simple.pdf'));
        try {
            $editor->setProducer('smoke-test-producer');
            $this->assertSame('smoke-test-producer', $editor->getProducer());
        } finally {
            $editor->close();
        }
    }

    public function testVersionReturnsMajorMinor(): void
    {
        $editor = DocumentEditor::open($this->fixture('simple.pdf'));
        try {
            $v = $editor->version();
            $this->assertArrayHasKey('major', $v);
            $this->assertArrayHasKey('minor', $v);
        } finally {
            $editor->close();
        }
    }

    public function testSavetoBytesProducesPdfHeader(): void
    {
        $editor = DocumentEditor::open($this->fixture('simple.pdf'));
        try {
            $bytes = $editor->save();
            $this->assertNotEmpty($bytes);
            $this->assertSame('%PDF-', substr($bytes, 0, 5));
        } finally {
            $editor->close();
        }
    }
}

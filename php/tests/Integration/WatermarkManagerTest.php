<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Integration;

use PdfOxide\Managers\WatermarkManager;
use PHPUnit\Framework\TestCase;

/**
 * Surface test for WatermarkManager.
 *
 * The watermark FFI surface operates on a page-builder handle
 * (`pdf_page_builder_*`); the v0.3.55 PHP creation API only partially
 * exposes the builder pipeline (Phase 6.10 is deferred to v0.3.56),
 * so a full end-to-end stamp-and-extract test is out of scope here.
 *
 * What this test covers:
 *  - WatermarkManager loads cleanly under composer autoload + the FFI
 *    extension.
 *  - All five public watermark / stamp / freetext entry points are
 *    reflected and callable.
 *
 * Full builder-flow tests land alongside the v0.3.56 creation API.
 *
 * @requires extension ffi
 */
final class WatermarkManagerTest extends TestCase
{
    public function testClassLoadsAndExposesExpectedMethods(): void
    {
        $rc = new \ReflectionClass(WatermarkManager::class);
        $this->assertTrue($rc->hasMethod('addText'));
        $this->assertTrue($rc->hasMethod('addConfidential'));
        $this->assertTrue($rc->hasMethod('addDraft'));
        $this->assertTrue($rc->hasMethod('addStamp'));
        $this->assertTrue($rc->hasMethod('addFreetext'));
    }

    public function testAddTextSignatureIsCorrect(): void
    {
        $rc = new \ReflectionClass(WatermarkManager::class);
        $method = $rc->getMethod('addText');
        $params = $method->getParameters();
        $this->assertCount(1, $params, 'addText takes exactly one string arg.');
        $this->assertSame('text', $params[0]->getName());
        $returnType = $method->getReturnType();
        $this->assertNotNull($returnType);
        $this->assertSame('int', (string)$returnType);
    }

    public function testAddFreetextSignatureIsCorrect(): void
    {
        $rc = new \ReflectionClass(WatermarkManager::class);
        $method = $rc->getMethod('addFreetext');
        $params = $method->getParameters();
        $this->assertCount(5, $params, 'addFreetext takes x,y,w,h,text.');
        $this->assertSame('x', $params[0]->getName());
        $this->assertSame('text', $params[4]->getName());
    }
}

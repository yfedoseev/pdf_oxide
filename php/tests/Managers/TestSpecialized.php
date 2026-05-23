<?php

namespace PdfOxide\Tests\Managers;

use PHPUnit\Framework\TestCase;
use PdfOxide\Managers\DOMElementsManager;
use PdfOxide\Managers\PDFCreatorManager;

class TestDOMElementsManager extends TestCase
{
    private DOMElementsManager $manager;

    protected function setUp(): void
    {
        $this->manager = new DOMElementsManager();
    }

    public function testGetElementByID(): void
    {
        $result = $this->manager->getElementByID('elem_1');
        $this->assertTrue($result === null || is_array($result));
    }

    public function testGetElementByType(): void
    {
        $result = $this->manager->getElementByType('text');
        $this->assertTrue($result === null || is_array($result));
    }

    public function testGetElementProperties(): void
    {
        $result = $this->manager->getElementProperties('elem_1');
        $this->assertTrue($result === null || is_array($result));
    }

    public function testSetElementProperty(): void
    {
        $result = $this->manager->setElementProperty('elem_1', 'color', '#FF0000');
        $this->assertIsBool($result);
    }

    public function testRemoveElement(): void
    {
        $result = $this->manager->removeElement('elem_1');
        $this->assertIsBool($result);
    }

    public function testGetElementChildren(): void
    {
        $result = $this->manager->getElementChildren('elem_1');
        $this->assertTrue($result === null || is_array($result));
    }

    public function testGetElementParent(): void
    {
        $result = $this->manager->getElementParent('elem_1');
        $this->assertTrue($result === null || is_array($result));
    }
}

class TestPDFCreatorManager extends TestCase
{
    private PDFCreatorManager $manager;

    protected function setUp(): void
    {
        $this->manager = new PDFCreatorManager();
    }

    public function testCreateBlankDocument(): void
    {
        $result = $this->manager->createBlankDocument(612, 792);
        $this->assertIsBool($result);
    }

    public function testCreateFromImages(): void
    {
        $result = $this->manager->createFromImages(['/img1.png', '/img2.png']);
        $this->assertIsBool($result);
    }

    public function testAddPageFromTemplate(): void
    {
        $result = $this->manager->addPageFromTemplate('template_1');
        $this->assertIsBool($result);
    }

    public function testCreateBooklet(): void
    {
        $result = $this->manager->createBooklet();
        $this->assertIsBool($result);
    }

    public function testCreateMultipleColumns(): void
    {
        $result = $this->manager->createMultipleColumns(2);
        $this->assertIsBool($result);
    }

    public function testAddCustomPageSize(): void
    {
        $result = $this->manager->addCustomPageSize(400, 600);
        $this->assertIsBool($result);
    }

    public function testSaveAsTemplate(): void
    {
        $result = $this->manager->saveAsTemplate('template_2');
        $this->assertIsBool($result);
    }
}

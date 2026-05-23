<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Unit\Xfa;

use PHPUnit\Framework\TestCase;
use PdfOxide\Xfa\XfaForm;
use PdfOxide\Xfa\XfaField;
use PdfOxide\FFI\FunctionBindings;
use FFI\CData;

/**
 * Tests for XfaForm and XfaField classes
 *
 * @covers \PdfOxide\Xfa\XfaForm
 * @covers \PdfOxide\Xfa\XfaField
 */
class XfaFormTest extends TestCase
{
    private FunctionBindings $bindingsMock;
    private CData $formHandleMock;

    protected function setUp(): void
    {
        $this->bindingsMock = $this->getMockBuilder(FunctionBindings::class)
            ->disableOriginalConstructor()
            ->onlyMethods([
                'pdfXfaFormFieldCount',
                'pdfXfaFormGetField',
                'pdfXfaFieldGetName',
                'pdfXfaFieldGetType',
                'pdfXfaFieldGetValue',
                'pdfXfaFieldSetValue',
                'pdfXfaFieldFree',
                'pdfXfaFormFree',
            ])
            ->getMock();

        $this->formHandleMock = $this->getMockBuilder(CData::class)->getMock();
    }

    /**
     * Test XfaForm initialization
     */
    public function testXfaFormInitialization(): void
    {
        $this->bindingsMock->method('pdfXfaFormFieldCount')->willReturn(0);
        $form = new XfaForm($this->formHandleMock, $this->bindingsMock);

        $this->assertInstanceOf(XfaForm::class, $form);
    }

    /**
     * Test getFieldCount
     */
    public function testGetFieldCount(): void
    {
        $this->bindingsMock->method('pdfXfaFormFieldCount')->willReturn(5);
        $form = new XfaForm($this->formHandleMock, $this->bindingsMock);

        $this->assertEquals(5, $form->getFieldCount());
    }

    /**
     * Test getField with invalid index throws exception
     */
    public function testGetFieldInvalidIndexThrows(): void
    {
        $this->bindingsMock->method('pdfXfaFormFieldCount')->willReturn(3);
        $form = new XfaForm($this->formHandleMock, $this->bindingsMock);

        $this->expectException(\OutOfRangeException::class);
        $form->getField(5);
    }

    /**
     * Test getFieldNames
     */
    public function testGetFieldNames(): void
    {
        $fieldMock = $this->getMockBuilder(CData::class)->getMock();

        $this->bindingsMock->method('pdfXfaFormFieldCount')->willReturn(2);
        $this->bindingsMock->method('pdfXfaFormGetField')->willReturn($fieldMock);
        $this->bindingsMock->method('pdfXfaFieldGetName')
            ->willReturnOnConsecutiveCalls('firstName', 'lastName');
        $this->bindingsMock->method('pdfXfaFieldGetType')->willReturn('text');
        $this->bindingsMock->method('pdfXfaFieldGetValue')->willReturn('');

        $form = new XfaForm($this->formHandleMock, $this->bindingsMock);
        $names = $form->getFieldNames();

        $this->assertCount(2, $names);
        $this->assertContains('firstName', $names);
        $this->assertContains('lastName', $names);
    }

    /**
     * Test getFieldByName
     */
    public function testGetFieldByName(): void
    {
        $fieldMock = $this->getMockBuilder(CData::class)->getMock();

        $this->bindingsMock->method('pdfXfaFormFieldCount')->willReturn(1);
        $this->bindingsMock->method('pdfXfaFormGetField')->willReturn($fieldMock);
        $this->bindingsMock->method('pdfXfaFieldGetName')->willReturn('email');
        $this->bindingsMock->method('pdfXfaFieldGetType')->willReturn('text');
        $this->bindingsMock->method('pdfXfaFieldGetValue')->willReturn('');

        $form = new XfaForm($this->formHandleMock, $this->bindingsMock);
        $field = $form->getFieldByName('email');

        $this->assertInstanceOf(XfaField::class, $field);
    }

    /**
     * Test getFieldByName returns null for missing field
     */
    public function testGetFieldByNameNotFound(): void
    {
        $this->bindingsMock->method('pdfXfaFormFieldCount')->willReturn(0);
        $form = new XfaForm($this->formHandleMock, $this->bindingsMock);

        $field = $form->getFieldByName('nonexistent');
        $this->assertNull($field);
    }

    /**
     * Test getFieldValues
     */
    public function testGetFieldValues(): void
    {
        $fieldMock = $this->getMockBuilder(CData::class)->getMock();

        $this->bindingsMock->method('pdfXfaFormFieldCount')->willReturn(2);
        $this->bindingsMock->method('pdfXfaFormGetField')->willReturn($fieldMock);
        $this->bindingsMock->method('pdfXfaFieldGetName')
            ->willReturnOnConsecutiveCalls('name', 'email');
        $this->bindingsMock->method('pdfXfaFieldGetType')->willReturn('text');
        $this->bindingsMock->method('pdfXfaFieldGetValue')
            ->willReturnOnConsecutiveCalls('John Doe', 'john@example.com');

        $form = new XfaForm($this->formHandleMock, $this->bindingsMock);
        $values = $form->getFieldValues();

        $this->assertArrayHasKey('name', $values);
        $this->assertArrayHasKey('email', $values);
        $this->assertEquals('John Doe', $values['name']);
    }

    /**
     * Test XfaField getName
     */
    public function testXfaFieldGetName(): void
    {
        $fieldMock = $this->getMockBuilder(CData::class)->getMock();

        $this->bindingsMock->method('pdfXfaFieldGetName')->willReturn('firstName');
        $field = new XfaField($fieldMock, $this->bindingsMock);

        $this->assertEquals('firstName', $field->getName());
    }

    /**
     * Test XfaField getType
     */
    public function testXfaFieldGetType(): void
    {
        $fieldMock = $this->getMockBuilder(CData::class)->getMock();

        $this->bindingsMock->method('pdfXfaFieldGetType')->willReturn('text');
        $field = new XfaField($fieldMock, $this->bindingsMock);

        $this->assertEquals('text', $field->getType());
    }

    /**
     * Test XfaField getValue
     */
    public function testXfaFieldGetValue(): void
    {
        $fieldMock = $this->getMockBuilder(CData::class)->getMock();

        $this->bindingsMock->method('pdfXfaFieldGetValue')->willReturn('test value');
        $field = new XfaField($fieldMock, $this->bindingsMock);

        $this->assertEquals('test value', $field->getValue());
    }

    /**
     * Test XfaField setValue
     */
    public function testXfaFieldSetValue(): void
    {
        $fieldMock = $this->getMockBuilder(CData::class)->getMock();

        $this->bindingsMock->expects($this->once())
            ->method('pdfXfaFieldSetValue');

        $field = new XfaField($fieldMock, $this->bindingsMock);
        $field->setValue('new value');
    }

    /**
     * Test XfaField isTextField
     */
    public function testXfaFieldIsTextField(): void
    {
        $fieldMock = $this->getMockBuilder(CData::class)->getMock();

        $this->bindingsMock->method('pdfXfaFieldGetType')->willReturn('text');
        $field = new XfaField($fieldMock, $this->bindingsMock);

        $this->assertTrue($field->isTextField());
    }

    /**
     * Test XfaField isCheckbox
     */
    public function testXfaFieldIsCheckbox(): void
    {
        $fieldMock = $this->getMockBuilder(CData::class)->getMock();

        $this->bindingsMock->method('pdfXfaFieldGetType')->willReturn('checkbox');
        $field = new XfaField($fieldMock, $this->bindingsMock);

        $this->assertTrue($field->isCheckbox());
    }

    /**
     * Test XfaField isRadio
     */
    public function testXfaFieldIsRadio(): void
    {
        $fieldMock = $this->getMockBuilder(CData::class)->getMock();

        $this->bindingsMock->method('pdfXfaFieldGetType')->willReturn('radio');
        $field = new XfaField($fieldMock, $this->bindingsMock);

        $this->assertTrue($field->isRadio());
    }

    /**
     * Test XfaField isDropdown
     */
    public function testXfaFieldIsDropdown(): void
    {
        $fieldMock = $this->getMockBuilder(CData::class)->getMock();

        $this->bindingsMock->method('pdfXfaFieldGetType')->willReturn('dropdown');
        $field = new XfaField($fieldMock, $this->bindingsMock);

        $this->assertTrue($field->isDropdown());
    }

    /**
     * Test XfaField toArray
     */
    public function testXfaFieldToArray(): void
    {
        $fieldMock = $this->getMockBuilder(CData::class)->getMock();

        $this->bindingsMock->method('pdfXfaFieldGetName')->willReturn('email');
        $this->bindingsMock->method('pdfXfaFieldGetType')->willReturn('text');
        $this->bindingsMock->method('pdfXfaFieldGetValue')->willReturn('test@example.com');

        $field = new XfaField($fieldMock, $this->bindingsMock);
        $array = $field->toArray();

        $this->assertArrayHasKey('name', $array);
        $this->assertArrayHasKey('type', $array);
        $this->assertArrayHasKey('value', $array);
        $this->assertEquals('email', $array['name']);
        $this->assertEquals('text', $array['type']);
    }

    /**
     * Test XfaField __toString
     */
    public function testXfaFieldToString(): void
    {
        $fieldMock = $this->getMockBuilder(CData::class)->getMock();

        $this->bindingsMock->method('pdfXfaFieldGetName')->willReturn('name');
        $this->bindingsMock->method('pdfXfaFieldGetType')->willReturn('text');
        $this->bindingsMock->method('pdfXfaFieldGetValue')->willReturn('John');

        $field = new XfaField($fieldMock, $this->bindingsMock);
        $str = (string)$field;

        $this->assertStringContainsString('name', $str);
        $this->assertStringContainsString('text', $str);
        $this->assertStringContainsString('John', $str);
    }

    /**
     * Test XfaForm setFieldValue
     */
    public function testXfaFormSetFieldValue(): void
    {
        $fieldMock = $this->getMockBuilder(CData::class)->getMock();

        $this->bindingsMock->method('pdfXfaFormFieldCount')->willReturn(1);
        $this->bindingsMock->method('pdfXfaFormGetField')->willReturn($fieldMock);
        $this->bindingsMock->method('pdfXfaFieldGetName')->willReturn('email');
        $this->bindingsMock->method('pdfXfaFieldGetType')->willReturn('text');
        $this->bindingsMock->method('pdfXfaFieldGetValue')->willReturn('');
        $this->bindingsMock->expects($this->once())->method('pdfXfaFieldSetValue');

        $form = new XfaForm($this->formHandleMock, $this->bindingsMock);
        $form->setFieldValue('email', 'new@example.com');
    }

    /**
     * Test XfaForm setFieldValue with nonexistent field throws
     */
    public function testXfaFormSetFieldValueNotFoundThrows(): void
    {
        $this->bindingsMock->method('pdfXfaFormFieldCount')->willReturn(0);

        $form = new XfaForm($this->formHandleMock, $this->bindingsMock);

        $this->expectException(\InvalidArgumentException::class);
        $form->setFieldValue('nonexistent', 'value');
    }

    /**
     * Test XfaForm toArray
     */
    public function testXfaFormToArray(): void
    {
        $this->bindingsMock->method('pdfXfaFormFieldCount')->willReturn(0);
        $form = new XfaForm($this->formHandleMock, $this->bindingsMock);

        $array = $form->toArray();

        $this->assertArrayHasKey('fieldCount', $array);
        $this->assertArrayHasKey('fieldNames', $array);
        $this->assertArrayHasKey('fieldValues', $array);
        $this->assertArrayHasKey('fields', $array);
    }
}

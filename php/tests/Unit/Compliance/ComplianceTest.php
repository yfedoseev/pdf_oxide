<?php

declare(strict_types=1);

namespace PdfOxide\Tests\Unit\Compliance;

use PHPUnit\Framework\TestCase;
use PdfOxide\Compliance\Compliance;
use PdfOxide\PdfDocument;
use PdfOxide\Types\ComplianceResult;

/**
 * Tests for Compliance utility class
 *
 * @covers \PdfOxide\Compliance\Compliance
 */
class ComplianceTest extends TestCase
{
    private PdfDocument $docMock;

    protected function setUp(): void
    {
        $this->docMock = $this->getMockBuilder(PdfDocument::class)
            ->disableOriginalConstructor()
            ->getMock();
    }

    /**
     * Test supported PDF/A levels constant
     */
    public function testSupportedPdfALevels(): void
    {
        $levels = Compliance::getSupportedPdfALevels();

        $this->assertIsArray($levels);
        $this->assertCount(6, $levels);
        $this->assertContains('1a', $levels);
        $this->assertContains('1b', $levels);
        $this->assertContains('2a', $levels);
        $this->assertContains('2b', $levels);
        $this->assertContains('3a', $levels);
        $this->assertContains('3b', $levels);
    }

    /**
     * Test supported PDF/X standards constant
     */
    public function testSupportedPdfXStandards(): void
    {
        $standards = Compliance::getSupportedPdfXStandards();

        $this->assertIsArray($standards);
        $this->assertCount(3, $standards);
        $this->assertContains('1a', $standards);
        $this->assertContains('3', $standards);
        $this->assertContains('4', $standards);
    }

    /**
     * Test isValidPdfALevel with valid levels
     */
    public function testIsValidPdfALevelValid(): void
    {
        $validLevels = ['1a', '1b', '2a', '2b', '3a', '3b'];

        foreach ($validLevels as $level) {
            $this->assertTrue(Compliance::isValidPdfALevel($level));
        }
    }

    /**
     * Test isValidPdfALevel with case insensitivity
     */
    public function testIsValidPdfALevelCaseInsensitive(): void
    {
        $this->assertTrue(Compliance::isValidPdfALevel('1A'));
        $this->assertTrue(Compliance::isValidPdfALevel('2B'));
        $this->assertTrue(Compliance::isValidPdfALevel('3a'));
    }

    /**
     * Test isValidPdfALevel with invalid levels
     */
    public function testIsValidPdfALevelInvalid(): void
    {
        $this->assertFalse(Compliance::isValidPdfALevel('1c'));
        $this->assertFalse(Compliance::isValidPdfALevel('4a'));
        $this->assertFalse(Compliance::isValidPdfALevel('invalid'));
    }

    /**
     * Test isValidPdfXStandard with valid standards
     */
    public function testIsValidPdfXStandardValid(): void
    {
        $validStandards = ['1a', '3', '4'];

        foreach ($validStandards as $standard) {
            $this->assertTrue(Compliance::isValidPdfXStandard($standard));
        }
    }

    /**
     * Test isValidPdfXStandard with case insensitivity
     */
    public function testIsValidPdfXStandardCaseInsensitive(): void
    {
        $this->assertTrue(Compliance::isValidPdfXStandard('1A'));
        $this->assertTrue(Compliance::isValidPdfXStandard('4'));
        $this->assertTrue(Compliance::isValidPdfXStandard('3'));
    }

    /**
     * Test isValidPdfXStandard with invalid standards
     */
    public function testIsValidPdfXStandardInvalid(): void
    {
        $this->assertFalse(Compliance::isValidPdfXStandard('1b'));
        $this->assertFalse(Compliance::isValidPdfXStandard('2'));
        $this->assertFalse(Compliance::isValidPdfXStandard('5'));
    }

    /**
     * Test isValidPdfALevel with whitespace
     */
    public function testIsValidPdfALevelWithWhitespace(): void
    {
        $this->assertTrue(Compliance::isValidPdfALevel('  1a  '));
        $this->assertTrue(Compliance::isValidPdfALevel(' 2b '));
    }

    /**
     * Test isValidPdfXStandard with whitespace
     */
    public function testIsValidPdfXStandardWithWhitespace(): void
    {
        $this->assertTrue(Compliance::isValidPdfXStandard('  1a  '));
        $this->assertTrue(Compliance::isValidPdfXStandard(' 4 '));
    }

    /**
     * Test convertToPdfA with valid level returns bytes
     */
    public function testConvertToPdfAValidLevel(): void
    {
        $mockComplianceManager = $this->getMockBuilder(\PdfOxide\Managers\ComplianceManager::class)
            ->disableOriginalConstructor()
            ->getMock();

        $this->docMock->method('compliance')->willReturn($mockComplianceManager);
        $this->docMock->method('toBytes')->willReturn('PDF_BYTES');

        $bytes = Compliance::convertToPdfA($this->docMock, '2b');

        $this->assertEquals('PDF_BYTES', $bytes);
    }

    /**
     * Test convertToPdfA with case insensitive level
     */
    public function testConvertToPdfACaseInsensitive(): void
    {
        $mockComplianceManager = $this->getMockBuilder(\PdfOxide\Managers\ComplianceManager::class)
            ->disableOriginalConstructor()
            ->getMock();

        $this->docMock->method('compliance')->willReturn($mockComplianceManager);
        $this->docMock->method('toBytes')->willReturn('PDF_BYTES');

        $bytes = Compliance::convertToPdfA($this->docMock, '2B');

        $this->assertEquals('PDF_BYTES', $bytes);
    }

    /**
     * Test convertToPdfA with invalid level throws exception
     */
    public function testConvertToPdfAInvalidLevelThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('Invalid PDF/A level');

        Compliance::convertToPdfA($this->docMock, 'invalid');
    }

    /**
     * Test convertToPdfUa returns bytes
     */
    public function testConvertToPdfUa(): void
    {
        $mockComplianceManager = $this->getMockBuilder(\PdfOxide\Managers\ComplianceManager::class)
            ->disableOriginalConstructor()
            ->getMock();

        $this->docMock->method('compliance')->willReturn($mockComplianceManager);
        $this->docMock->method('toBytes')->willReturn('PDF_UA_BYTES');

        $bytes = Compliance::convertToPdfUa($this->docMock);

        $this->assertEquals('PDF_UA_BYTES', $bytes);
    }

    /**
     * Test convertToPdfX with valid standard returns bytes
     */
    public function testConvertToPdfXValidStandard(): void
    {
        $mockComplianceManager = $this->getMockBuilder(\PdfOxide\Managers\ComplianceManager::class)
            ->disableOriginalConstructor()
            ->getMock();

        $this->docMock->method('compliance')->willReturn($mockComplianceManager);
        $this->docMock->method('toBytes')->willReturn('PDF_X_BYTES');

        $bytes = Compliance::convertToPdfX($this->docMock, '4');

        $this->assertEquals('PDF_X_BYTES', $bytes);
    }

    /**
     * Test convertToPdfX with case insensitive standard
     */
    public function testConvertToPdfXCaseInsensitive(): void
    {
        $mockComplianceManager = $this->getMockBuilder(\PdfOxide\Managers\ComplianceManager::class)
            ->disableOriginalConstructor()
            ->getMock();

        $this->docMock->method('compliance')->willReturn($mockComplianceManager);
        $this->docMock->method('toBytes')->willReturn('PDF_X_BYTES');

        $bytes = Compliance::convertToPdfX($this->docMock, '4');

        $this->assertEquals('PDF_X_BYTES', $bytes);
    }

    /**
     * Test convertToPdfX with invalid standard throws exception
     */
    public function testConvertToPdfXInvalidStandardThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('Invalid PDF/X standard');

        Compliance::convertToPdfX($this->docMock, 'invalid');
    }

    /**
     * Test validatePdfA returns ComplianceResult
     */
    public function testValidatePdfAReturnsResult(): void
    {
        $resultMock = $this->getMockBuilder(ComplianceResult::class)
            ->disableOriginalConstructor()
            ->getMock();

        $mockComplianceManager = $this->getMockBuilder(\PdfOxide\Managers\ComplianceManager::class)
            ->disableOriginalConstructor()
            ->getMock();

        $mockComplianceManager->method('validatePdfA')->willReturn($resultMock);

        $this->docMock->method('compliance')->willReturn($mockComplianceManager);

        $result = Compliance::validatePdfA($this->docMock, '2a');

        $this->assertInstanceOf(ComplianceResult::class, $result);
    }

    /**
     * Test validatePdfA with invalid level throws exception
     */
    public function testValidatePdfAInvalidLevelThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);

        Compliance::validatePdfA($this->docMock, 'invalid');
    }

    /**
     * Test validatePdfUa returns ComplianceResult
     */
    public function testValidatePdfUaReturnsResult(): void
    {
        $resultMock = $this->getMockBuilder(ComplianceResult::class)
            ->disableOriginalConstructor()
            ->getMock();

        $mockComplianceManager = $this->getMockBuilder(\PdfOxide\Managers\ComplianceManager::class)
            ->disableOriginalConstructor()
            ->getMock();

        $mockComplianceManager->method('validatePdfUa')->willReturn($resultMock);

        $this->docMock->method('compliance')->willReturn($mockComplianceManager);

        $result = Compliance::validatePdfUa($this->docMock);

        $this->assertInstanceOf(ComplianceResult::class, $result);
    }

    /**
     * Test validatePdfX returns ComplianceResult
     */
    public function testValidatePdfXReturnsResult(): void
    {
        $resultMock = $this->getMockBuilder(ComplianceResult::class)
            ->disableOriginalConstructor()
            ->getMock();

        $mockComplianceManager = $this->getMockBuilder(\PdfOxide\Managers\ComplianceManager::class)
            ->disableOriginalConstructor()
            ->getMock();

        $mockComplianceManager->method('validatePdfX')->willReturn($resultMock);

        $this->docMock->method('compliance')->willReturn($mockComplianceManager);

        $result = Compliance::validatePdfX($this->docMock, '1a');

        $this->assertInstanceOf(ComplianceResult::class, $result);
    }

    /**
     * Test validatePdfX with invalid standard throws exception
     */
    public function testValidatePdfXInvalidStandardThrows(): void
    {
        $this->expectException(\InvalidArgumentException::class);

        Compliance::validatePdfX($this->docMock, 'invalid');
    }

    /**
     * Test all PDF/A levels are valid
     */
    public function testAllPdfALevelsValid(): void
    {
        $levels = Compliance::getSupportedPdfALevels();

        foreach ($levels as $level) {
            $this->assertTrue(Compliance::isValidPdfALevel($level));
        }
    }

    /**
     * Test all PDF/X standards are valid
     */
    public function testAllPdfXStandardsValid(): void
    {
        $standards = Compliance::getSupportedPdfXStandards();

        foreach ($standards as $standard) {
            $this->assertTrue(Compliance::isValidPdfXStandard($standard));
        }
    }

    /**
     * Test conversion method error message formatting
     */
    public function testConversionErrorMessageFormatPdfA(): void
    {
        $mockComplianceManager = $this->getMockBuilder(\PdfOxide\Managers\ComplianceManager::class)
            ->disableOriginalConstructor()
            ->getMock();

        $mockComplianceManager
            ->method('convertToPdfA')
            ->willThrowException(new \Exception('FFI Error'));

        $this->docMock->method('compliance')->willReturn($mockComplianceManager);

        $this->expectException(\RuntimeException::class);
        $this->expectExceptionMessage('Failed to convert document to PDF/A');

        Compliance::convertToPdfA($this->docMock, '1a');
    }

    /**
     * Test conversion method error message formatting for PDF/X
     */
    public function testConversionErrorMessageFormatPdfX(): void
    {
        $mockComplianceManager = $this->getMockBuilder(\PdfOxide\Managers\ComplianceManager::class)
            ->disableOriginalConstructor()
            ->getMock();

        $mockComplianceManager
            ->method('convertToPdfX')
            ->willThrowException(new \Exception('FFI Error'));

        $this->docMock->method('compliance')->willReturn($mockComplianceManager);

        $this->expectException(\RuntimeException::class);
        $this->expectExceptionMessage('Failed to convert document to PDF/X');

        Compliance::convertToPdfX($this->docMock, '4');
    }

    /**
     * Test validation method error message formatting
     */
    public function testValidationErrorMessageFormatPdfA(): void
    {
        $mockComplianceManager = $this->getMockBuilder(\PdfOxide\Managers\ComplianceManager::class)
            ->disableOriginalConstructor()
            ->getMock();

        $mockComplianceManager
            ->method('validatePdfA')
            ->willThrowException(new \Exception('FFI Error'));

        $this->docMock->method('compliance')->willReturn($mockComplianceManager);

        $this->expectException(\RuntimeException::class);
        $this->expectExceptionMessage('PDF/A validation failed');

        Compliance::validatePdfA($this->docMock, '2b');
    }

    /**
     * Test getSupportedPdfALevels is consistent
     */
    public function testGetSupportedPdfALevelsConsistent(): void
    {
        $levels1 = Compliance::getSupportedPdfALevels();
        $levels2 = Compliance::getSupportedPdfALevels();

        $this->assertEquals($levels1, $levels2);
    }

    /**
     * Test getSupportedPdfXStandards is consistent
     */
    public function testGetSupportedPdfXStandardsConsistent(): void
    {
        $standards1 = Compliance::getSupportedPdfXStandards();
        $standards2 = Compliance::getSupportedPdfXStandards();

        $this->assertEquals($standards1, $standards2);
    }

    /**
     * Test Compliance is not instantiable
     */
    public function testComplianceNotInstantiable(): void
    {
        $this->expectException(\Error::class);

        new Compliance();
    }

    /**
     * Test convertToPdfA with all levels
     */
    public function testConvertToPdfAAllLevels(): void
    {
        $mockComplianceManager = $this->getMockBuilder(\PdfOxide\Managers\ComplianceManager::class)
            ->disableOriginalConstructor()
            ->getMock();

        $this->docMock->method('compliance')->willReturn($mockComplianceManager);
        $this->docMock->method('toBytes')->willReturn('BYTES');

        $levels = Compliance::getSupportedPdfALevels();

        foreach ($levels as $level) {
            $bytes = Compliance::convertToPdfA($this->docMock, $level);
            $this->assertEquals('BYTES', $bytes);
        }
    }

    /**
     * Test convertToPdfX with all standards
     */
    public function testConvertToPdfXAllStandards(): void
    {
        $mockComplianceManager = $this->getMockBuilder(\PdfOxide\Managers\ComplianceManager::class)
            ->disableOriginalConstructor()
            ->getMock();

        $this->docMock->method('compliance')->willReturn($mockComplianceManager);
        $this->docMock->method('toBytes')->willReturn('BYTES');

        $standards = Compliance::getSupportedPdfXStandards();

        foreach ($standards as $standard) {
            $bytes = Compliance::convertToPdfX($this->docMock, $standard);
            $this->assertEquals('BYTES', $bytes);
        }
    }
}

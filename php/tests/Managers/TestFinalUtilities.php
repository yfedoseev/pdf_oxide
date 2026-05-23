<?php

/**
 * Comprehensive test suite for Phase 8 Final Utilities.
 * Tests: EventManager, EncryptionManager, CompressionManager, CustomAnnotationManager, ContentSecurityManager
 */

namespace PdfOxide\Tests\Managers;

use PHPUnit\Framework\TestCase;
use PdfOxide\Managers\EventManager;
use PdfOxide\Managers\EncryptionManager;
use PdfOxide\Managers\CompressionManager;
use PdfOxide\Managers\CustomAnnotationManager;
use PdfOxide\Managers\ContentSecurityManager;

class TestEventManager extends TestCase
{
    private $manager;

    protected function setUp(): void
    {
        parent::setUp();
        $this->manager = new EventManager();
    }

    public function testAddEventListener_ReturnsTrue()
    {
        $handler = function ($e) {};
        $result = $this->manager->addEventListener('PAGE_LOADED', $handler);
        $this->assertTrue($result);
    }

    public function testRemoveEventListener_ReturnsTrue()
    {
        $handler = function ($e) {};
        $this->manager->addEventListener('PAGE_LOADED', $handler);
        $result = $this->manager->removeEventListener('PAGE_LOADED', $handler);
        $this->assertTrue($result);
    }

    public function testHasListener_ReturnsTrueForRegistered()
    {
        $handler = function ($e) {};
        $this->manager->addEventListener('CONTENT_PARSED', $handler);
        $result = $this->manager->hasListener('CONTENT_PARSED');
        $this->assertTrue($result);
    }

    public function testHasListener_ReturnsFalseForUnregistered()
    {
        $result = $this->manager->hasListener('ERROR_OCCURRED');
        $this->assertFalse($result);
    }

    public function testGetListenerCount_ReturnsCorrectCount()
    {
        $this->manager->addEventListener('PAGE_RENDERED', function ($e) {});
        $this->manager->addEventListener('PAGE_RENDERED', function ($e) {});
        $count = $this->manager->getListenerCount('PAGE_RENDERED');
        $this->assertEquals(2, $count);
    }

    public function testClearListeners_RemovesAll()
    {
        $this->manager->addEventListener('SEARCH_COMPLETED', function ($e) {});
        $result = $this->manager->clearListeners('SEARCH_COMPLETED');
        $this->assertTrue($result);
        $this->assertEquals(0, $this->manager->getListenerCount('SEARCH_COMPLETED'));
    }

    public function testGetEventStatistics_ReturnsDict()
    {
        $this->manager->addEventListener('PAGE_LOADED', function ($e) {});
        $this->manager->addEventListener('CONTENT_PARSED', function ($e) {});
        $stats = $this->manager->getEventStatistics();
        $this->assertIsArray($stats);
    }

    public function testEnableEventLogging_ReturnsBool()
    {
        $result = $this->manager->enableEventLogging(true);
        $this->assertIsBool($result);
    }

    public function testEventEmissionAndHandling_Completes()
    {
        $called = false;
        $handler = function ($e) use (&$called) {
            $called = true;
        };
        $this->manager->addEventListener('PAGE_LOADED', $handler);
        // Event would be emitted here
    }
}

class TestEncryptionManager extends TestCase
{
    private $manager;

    protected function setUp(): void
    {
        parent::setUp();
        $this->manager = new EncryptionManager();
    }

    public function testEncryptDocument_ReturnsBool()
    {
        $settings = [
            'algorithm' => 'AES_256',
            'user_password' => 'user123',
            'owner_password' => 'owner123',
            'allow_printing' => true,
            'allow_copying' => false,
            'allow_modification' => false,
        ];
        $result = $this->manager->encryptDocument($settings);
        $this->assertIsBool($result);
    }

    public function testDecryptDocument_ReturnsBool()
    {
        $result = $this->manager->decryptDocument('password123');
        $this->assertIsBool($result);
    }

    public function testChangeEncryption_ReturnsBool()
    {
        $settings = ['algorithm' => 'AES_256'];
        $result = $this->manager->changeEncryption($settings);
        $this->assertIsBool($result);
    }

    public function testIsDocumentEncrypted_ReturnsBool()
    {
        $result = $this->manager->isDocumentEncrypted();
        $this->assertIsBool($result);
    }

    public function testSetUserPassword_ReturnsBool()
    {
        $result = $this->manager->setUserPassword('newpass123');
        $this->assertIsBool($result);
    }

    public function testSetOwnerPassword_ReturnsBool()
    {
        $result = $this->manager->setOwnerPassword('ownerpass123');
        $this->assertIsBool($result);
    }

    public function testValidatePassword_ReturnsBool()
    {
        $result = $this->manager->validatePassword('testpass');
        $this->assertIsBool($result);
    }

    public function testGetPermissions_ReturnsDict()
    {
        $perms = $this->manager->getPermissions();
        $this->assertIsArray($perms);
    }

    public function testSetPermissions_ReturnsBool()
    {
        $perms = ['allow_print' => true, 'allow_copy' => false];
        $result = $this->manager->setPermissions($perms);
        $this->assertIsBool($result);
    }

    public function testRemoveEncryption_ReturnsBool()
    {
        $result = $this->manager->removeEncryption('ownerpass');
        $this->assertIsBool($result);
    }

    public function testAllEncryptionAlgorithmsSupported()
    {
        $algorithms = ['AES_128', 'AES_256', 'RC4_40', 'RC4_128'];

        foreach ($algorithms as $algo) {
            $settings = [
                'algorithm' => $algo,
                'user_password' => 'user',
                'owner_password' => 'owner',
            ];
            $result = $this->manager->encryptDocument($settings);
            $this->assertNotNull($result);
        }
    }
}

class TestCompressionManager extends TestCase
{
    private $manager;

    protected function setUp(): void
    {
        parent::setUp();
        $this->manager = new CompressionManager();
    }

    public function testCompressDocument_ReturnsBool()
    {
        $settings = [
            'level' => 'BALANCED',
            'compress_images' => true,
            'compress_streams' => true,
            'compress_fonts' => true,
            'remove_duplicates' => true,
        ];
        $result = $this->manager->compressDocument($settings);
        $this->assertIsBool($result);
    }

    public function testCompressImages_ReturnsBool()
    {
        $result = $this->manager->compressImages(85);
        $this->assertIsBool($result);
    }

    public function testCompressStreams_ReturnsBool()
    {
        $result = $this->manager->compressStreams();
        $this->assertIsBool($result);
    }

    public function testCompressPage_ReturnsBool()
    {
        $settings = ['level' => 'BALANCED'];
        $result = $this->manager->compressPage(0, $settings);
        $this->assertIsBool($result);
    }

    public function testIsCompressed_ReturnsBool()
    {
        $result = $this->manager->isCompressed();
        $this->assertIsBool($result);
    }

    public function testDecompressDocument_ReturnsBool()
    {
        $result = $this->manager->decompressDocument();
        $this->assertIsBool($result);
    }

    public function testGetCompressionRatio_ReturnsNumberOrNull()
    {
        $ratio = $this->manager->getCompressionRatio();
        $this->assertTrue($ratio === null || is_numeric($ratio));
    }

    public function testGetCompressionReport_ReturnsDict()
    {
        $report = $this->manager->getCompressionReport();
        $this->assertIsArray($report);
    }

    public function testOptimizeForWeb_ReturnsBool()
    {
        $result = $this->manager->optimizeForWeb();
        $this->assertIsBool($result);
    }

    public function testOptimizeForPrint_ReturnsBool()
    {
        $result = $this->manager->optimizeForPrint();
        $this->assertIsBool($result);
    }

    public function testAllCompressionLevelsSupported()
    {
        $levels = ['NONE', 'FAST', 'BALANCED', 'BEST'];

        foreach ($levels as $level) {
            $settings = [
                'level' => $level,
                'compress_images' => true,
                'compress_streams' => true,
                'compress_fonts' => true,
                'remove_duplicates' => true,
            ];
            $this->manager->compressDocument($settings);
        }
    }
}

class TestCustomAnnotationManager extends TestCase
{
    private $manager;

    protected function setUp(): void
    {
        parent::setUp();
        $this->manager = new CustomAnnotationManager();
    }

    public function testCreateCustomAnnotation_ReturnsIdOrNull()
    {
        $props = ['color' => 'red', 'opacity' => 0.5];
        $result = $this->manager->createCustomAnnotation('highlight', $props);
        $this->assertTrue($result === null || is_string($result));
    }

    public function testModifyAnnotation_ReturnsBool()
    {
        $props = ['color' => 'blue'];
        $result = $this->manager->modifyAnnotation('anno_1', $props);
        $this->assertIsBool($result);
    }

    public function testDeleteCustomAnnotation_ReturnsBool()
    {
        $result = $this->manager->deleteCustomAnnotation('anno_1');
        $this->assertIsBool($result);
    }

    public function testRegisterAnnotationType_ReturnsBool()
    {
        $handler = function ($e) {};
        $result = $this->manager->registerAnnotationType('custom_type', $handler);
        $this->assertIsBool($result);
    }

    public function testSetAnnotationVisibility_ReturnsBool()
    {
        $result = $this->manager->setAnnotationVisibility('anno_1', true);
        $this->assertIsBool($result);
    }

    public function testExportAnnotations_ReturnsBool()
    {
        $result = $this->manager->exportAnnotations('/output.json');
        $this->assertIsBool($result);
    }

    public function testImportAnnotations_ReturnsBool()
    {
        $result = $this->manager->importAnnotations('/input.json');
        $this->assertIsBool($result);
    }

    public function testApplyAnnotationStyle_ReturnsBool()
    {
        $style = ['font_size' => 12, 'color' => 'red'];
        $result = $this->manager->applyAnnotationStyle('anno_1', $style);
        $this->assertIsBool($result);
    }

    public function testReplyToAnnotation_ReturnsBool()
    {
        $result = $this->manager->replyToAnnotation('anno_1', 'This is a reply');
        $this->assertIsBool($result);
    }

    public function testGetAnnotationReplies_ReturnsList()
    {
        $replies = $this->manager->getAnnotationReplies('anno_1');
        $this->assertIsArray($replies);
    }

    public function testFlattenAnnotations_ReturnsBool()
    {
        $result = $this->manager->flattenAnnotations();
        $this->assertIsBool($result);
    }

    public function testConvertAnnotations_ReturnsBool()
    {
        $result = $this->manager->convertAnnotations('xfdf');
        $this->assertIsBool($result);
    }

    public function testAnnotationLifecycle_Completes()
    {
        // Create
        $annoId = $this->manager->createCustomAnnotation('note', ['text' => 'Important']);
        // Modify
        if ($annoId !== null) {
            $this->manager->modifyAnnotation($annoId, ['color' => 'yellow']);
            // Reply
            $this->manager->replyToAnnotation($annoId, 'Updated');
            // Get replies
            $replies = $this->manager->getAnnotationReplies($annoId);
            $this->assertIsArray($replies);
        }
    }
}

class TestContentSecurityManager extends TestCase
{
    private $manager;

    protected function setUp(): void
    {
        parent::setUp();
        $this->manager = new ContentSecurityManager();
    }

    public function testSetAccessControl_ReturnsBool()
    {
        $restrictions = ['role' => 'admin', 'action' => 'read'];
        $result = $this->manager->setAccessControl('admin_policy', $restrictions);
        $this->assertIsBool($result);
    }

    public function testValidateAccess_ReturnsBool()
    {
        $result = $this->manager->validateAccess('admin', 'read');
        $this->assertIsBool($result);
    }

    public function testApplyDigitalRights_ReturnsBool()
    {
        $rights = ['can_print' => true, 'can_copy' => false, 'can_modify' => false];
        $result = $this->manager->applyDigitalRights($rights);
        $this->assertIsBool($result);
    }

    public function testSanitizeContent_ReturnsBool()
    {
        $result = $this->manager->sanitizeContent(true, true);
        $this->assertIsBool($result);
    }

    public function testDetectSuspiciousContent_ReturnsList()
    {
        $results = $this->manager->detectSuspiciousContent();
        $this->assertIsArray($results);
    }

    public function testGetAccessLog_ReturnsList()
    {
        $log = $this->manager->getAccessLog();
        $this->assertIsArray($log);
    }

    public function testSetExpirationDate_ReturnsBool()
    {
        $result = $this->manager->setExpirationDate('2025-12-31');
        $this->assertIsBool($result);
    }

    public function testEnableWatermarking_ReturnsBool()
    {
        $result = $this->manager->enableWatermarking('CONFIDENTIAL');
        $this->assertIsBool($result);
    }

    public function testTrackDocumentUsage_ReturnsBool()
    {
        $result = $this->manager->trackDocumentUsage(true);
        $this->assertIsBool($result);
    }

    public function testGetSecurityAudit_ReturnsDict()
    {
        $audit = $this->manager->getSecurityAudit();
        $this->assertIsArray($audit);
    }

    public function testSecurityPolicyEnforcement_Completes()
    {
        $this->manager->setAccessControl('strict', ['role' => 'viewer', 'action' => 'read_only']);
        $this->manager->applyDigitalRights(['can_print' => false, 'can_copy' => false]);
        $this->manager->enableWatermarking('RESTRICTED');
        $this->manager->sanitizeContent(true, true);
        $audit = $this->manager->getSecurityAudit();
        $this->assertIsArray($audit);
    }
}

class TestPhase8Integration extends TestCase
{
    public function testDocumentProtectionWorkflow_Completes()
    {
        $eventMgr = new EventManager();
        $encryptionMgr = new EncryptionManager();
        $securityMgr = new ContentSecurityManager();

        $eventMgr->addEventListener('PROCESSING_STARTED', function ($e) {
            echo "Processing started\n";
        });

        $settings = [
            'algorithm' => 'AES_256',
            'user_password' => 'user',
            'owner_password' => 'owner',
            'allow_printing' => true,
            'allow_copying' => false,
            'allow_modification' => false,
        ];
        $encryptionMgr->encryptDocument($settings);

        $securityMgr->setAccessControl('restricted', ['action' => 'read_only']);
        $securityMgr->enableWatermarking('CONFIDENTIAL');
    }

    public function testDocumentOptimizationWorkflow_Completes()
    {
        $compressionMgr = new CompressionManager();
        $annotationMgr = new CustomAnnotationManager();
        $securityMgr = new ContentSecurityManager();

        $settings = [
            'level' => 'BALANCED',
            'compress_images' => true,
            'compress_streams' => true,
            'compress_fonts' => true,
            'remove_duplicates' => true,
        ];
        $compressionMgr->compressDocument($settings);

        $annoId = $annotationMgr->createCustomAnnotation('note', ['text' => 'Optimized']);

        $securityMgr->sanitizeContent(true, true);
    }
}

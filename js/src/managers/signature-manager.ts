/**
 * SignatureManager - Canonical Signature Manager (merged from 3 implementations)
 *
 * Consolidates:
 * - src/signature-manager.ts SignatureManager (verification + basic field management)
 * - src/managers/barcode-signature-rendering.ts SignaturesManager (certificate loading + signing + detail info)
 * - src/managers/signature-creation-manager.ts SignatureCreationManager (complete signing workflow + LTV + timestamps)
 *
 * Provides comprehensive digital signature operations with full type safety
 * and FFI integration.
 */

import { EventEmitter } from 'events';
import { promises as fs } from 'fs';
import { mapFfiErrorCode, SignatureException } from '../errors';

// =============================================================================
// Type Definitions (merged from all 3 sources)
// =============================================================================

/**
 * Signature algorithms
 */
export enum SignatureAlgorithm {
  RSA2048 = 'RSA2048',
  RSA3072 = 'RSA3072',
  RSA4096 = 'RSA4096',
  ECDSAP256 = 'ECDSA_P256',
  ECDSAP384 = 'ECDSA_P384',
  ECDSAP521 = 'ECDSA_P521',
  RSA_SHA256 = 'RSA_SHA256',
  RSA_SHA384 = 'RSA_SHA384',
  RSA_SHA512 = 'RSA_SHA512',
  ECDSA_SHA256 = 'ECDSA_SHA256',
  ECDSA_SHA384 = 'ECDSA_SHA384',
  ECDSA_SHA512 = 'ECDSA_SHA512',
  ED25519 = 'ED25519',
}

/**
 * Digest algorithms for signature
 */
export enum DigestAlgorithm {
  SHA256 = 'SHA256',
  SHA384 = 'SHA384',
  SHA512 = 'SHA512',
  SHA512_256 = 'SHA512_256',
}

/**
 * Signature type enumeration
 */
export enum SignatureType {
  APPROVAL = 'approval',
  CERTIFICATION = 'certification',
  USAGE_RIGHTS = 'usage_rights',
}

/**
 * Certification permission level
 */
export enum CertificationPermission {
  NO_CHANGES = 1,
  FORM_FILLING = 2,
  FORM_FILLING_ANNOTATIONS = 3,
}

/**
 * Certificate format enumeration
 */
export enum CertificateFormat {
  PFX = 'pfx',
  PEM = 'pem',
  DER = 'der',
  P12 = 'p12',
  CER = 'cer',
}

/**
 * Timestamp response status
 */
export enum TimestampStatus {
  SUCCESS = 'success',
  FAILED = 'failed',
  TIMEOUT = 'timeout',
  INVALID_RESPONSE = 'invalid_response',
}

// --- Interfaces from root-level SignatureManager ---

export interface DigitalSignature {
  signatureName: string;
  signingDate?: Date;
  reason?: string;
  location?: string;
  signer?: string;
  isCertified: boolean;
  algorithm?: SignatureAlgorithm;
}

export interface SignatureField {
  fieldName: string;
  pageIndex: number;
  isSigned: boolean;
  signature?: DigitalSignature;
}

export interface SignatureValidationResult {
  isValid: boolean;
  signatures: DigitalSignature[];
  issues: string[];
}

export interface SignatureConfig {
  algorithm?: SignatureAlgorithm;
  digestAlgorithm?: DigestAlgorithm;
  reason?: string;
  location?: string;
}

// --- Interfaces from SignaturesManager ---

export interface Certificate {
  subject: string;
  issuer: string;
  serial: string;
  notBefore: number;
  notAfter: number;
  isValid: boolean;
}

export interface Signature {
  signerName: string;
  signingTime: number;
  reason?: string;
  location?: string;
  certificate: Certificate;
  isValid: boolean;
}

// --- Interfaces from SignatureCreationManager ---

export interface CertificateInfo {
  readonly subject: string;
  readonly issuer: string;
  readonly serialNumber: string;
  readonly validFrom: Date;
  readonly validTo: Date;
  readonly isValid: boolean;
  readonly isSelfSigned: boolean;
  readonly keyUsage?: readonly string[];
  readonly extendedKeyUsage?: readonly string[];
  readonly subjectAltNames?: readonly string[];
  readonly thumbprint?: string;
  readonly publicKeyAlgorithm?: string;
  readonly signatureAlgorithm?: string;
}

export interface CertificateChain {
  readonly certificates: readonly CertificateInfo[];
  readonly isComplete: boolean;
  readonly validationStatus: 'valid' | 'invalid' | 'unknown';
  readonly validationMessages?: readonly string[];
}

export interface LoadedCertificate {
  readonly certificateId: string;
  readonly info: CertificateInfo;
  readonly hasPrivateKey: boolean;
  readonly chain?: CertificateChain;
}

export interface SignatureAppearance {
  readonly showName?: boolean;
  readonly showDate?: boolean;
  readonly showReason?: boolean;
  readonly showLocation?: boolean;
  readonly showLabels?: boolean;
  readonly imageData?: Buffer;
  readonly imagePath?: string;
  readonly backgroundColor?: string;
  readonly textColor?: string;
  readonly borderColor?: string;
  readonly borderWidth?: number;
  readonly font?: string;
  readonly fontSize?: number;
  readonly customText?: string;
}

export interface SignatureFieldConfig {
  readonly fieldName: string;
  readonly pageIndex: number;
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
  readonly appearance?: SignatureAppearance;
  readonly tooltip?: string;
  readonly isRequired?: boolean;
  readonly isReadOnly?: boolean;
}

export interface SigningOptions {
  readonly reason?: string;
  readonly location?: string;
  readonly contactInfo?: string;
  readonly signatureType?: SignatureType;
  readonly certificationPermission?: CertificationPermission;
  readonly algorithm?: SignatureAlgorithm;
  readonly digestAlgorithm?: DigestAlgorithm;
  readonly appearance?: SignatureAppearance;
  readonly embedTimestamp?: boolean;
  readonly timestampServerUrl?: string;
  readonly enableLtv?: boolean;
  readonly ocspResponderUrl?: string;
  readonly crlDistributionPoints?: readonly string[];
}

export interface TimestampConfig {
  readonly serverUrl: string;
  readonly username?: string;
  readonly password?: string;
  readonly hashAlgorithm?: DigestAlgorithm;
  readonly timeout?: number;
  readonly policy?: string;
  readonly nonce?: boolean;
  readonly certReq?: boolean;
}

export interface SigningResult {
  readonly success: boolean;
  readonly signatureId?: string;
  readonly signingTime?: Date;
  readonly timestampTime?: Date;
  readonly error?: string;
  readonly warnings?: readonly string[];
}

export interface TimestampResult {
  readonly status: TimestampStatus;
  readonly timestamp?: Date;
  readonly serialNumber?: string;
  readonly tsaName?: string;
  readonly error?: string;
}

// --- Interfaces for FFI-based signing (Phase 1) ---

/**
 * Opaque handle to signing credentials loaded via FFI.
 * Wraps a native pointer returned by pdf_credentials_from_pkcs12 or pdf_credentials_from_pem.
 */
export interface SigningCredentials {
  /** Internal native handle - do not access directly */
  readonly _handle: any;
  /** Source type: 'pkcs12' or 'pem' */
  readonly sourceType: 'pkcs12' | 'pem';
}

/**
 * Options for FFI-based document signing operations.
 */
export interface SignOptions {
  /** Reason for signing the document */
  readonly reason?: string;
  /** Location where the document was signed */
  readonly location?: string;
  /** Contact information for the signer */
  readonly contact?: string;
  /** Digest algorithm (0=SHA1, 1=SHA256, 2=SHA384, 3=SHA512). Defaults to SHA256 (1). */
  readonly algorithm?: number;
  /** Signature subfilter (0=PKCS7_DETACHED, 1=PKCS7_SHA1, 2=CADES_DETACHED). Defaults to PKCS7_DETACHED (0). */
  readonly subfilter?: number;
}

/**
 * FFI digest algorithm constants for use with SignOptions.algorithm
 */
export enum FfiDigestAlgorithm {
  SHA1 = 0,
  SHA256 = 1,
  SHA384 = 2,
  SHA512 = 3,
}

/**
 * FFI signature subfilter constants for use with SignOptions.subfilter
 */
export enum FfiSignatureSubFilter {
  PKCS7_DETACHED = 0,
  PKCS7_SHA1 = 1,
  CADES_DETACHED = 2,
}

// =============================================================================
// Canonical SignatureManager
// =============================================================================

/**
 * Canonical Signature Manager - all signature operations in one class.
 *
 * Provides:
 * - Verification (from root SignatureManager)
 * - Certificate loading (from SignatureCreationManager)
 * - Document signing with timestamps and LTV (from SignatureCreationManager)
 * - Detailed signature info (from SignaturesManager)
 */
export class SignatureManager extends EventEmitter {
  private document: any;
  private resultCache = new Map<string, any>();
  private maxCacheSize = 100;
  private native: any;
  private readonly loadedCertificates: Map<string, LoadedCertificate> = new Map();
  private readonly createdFields: Map<string, SignatureFieldConfig> = new Map();

  constructor(document: any) {
    super();
    if (!document) {
      throw new Error('Document cannot be null or undefined');
    }
    this.document = document;
    try {
      this.native = require('../../index.node');
    } catch {
      this.native = null;
    }
  }

  // ===========================================================================
  // Verification (from root SignatureManager)
  // ===========================================================================

  async getSignatures(): Promise<DigitalSignature[]> {
    const cacheKey = 'signatures:all';
    if (this.resultCache.has(cacheKey)) return this.resultCache.get(cacheKey);
    let signatures: DigitalSignature[] = [];
    if (this.native?.get_signatures) {
      try {
        const signaturesJson = this.native.get_signatures() ?? [];
        signatures = signaturesJson.length > 0 ? JSON.parse(signaturesJson[0] || '[]') : [];
      } catch {
        signatures = [];
      }
    }
    this.setCached(cacheKey, signatures);
    this.emit('signaturesRetrieved', { count: signatures.length });
    return signatures;
  }

  async getSignatureFields(): Promise<SignatureField[]> {
    const cacheKey = 'signatures:fields';
    if (this.resultCache.has(cacheKey)) return this.resultCache.get(cacheKey);
    let fields: SignatureField[] = [];
    if (this.native?.get_signature_fields) {
      try {
        const fieldsJson = this.native.get_signature_fields() ?? [];
        fields = fieldsJson.length > 0 ? JSON.parse(fieldsJson[0] || '[]') : [];
      } catch {
        fields = [];
      }
    }
    this.setCached(cacheKey, fields);
    return fields;
  }

  async verifySignatures(): Promise<SignatureValidationResult> {
    const cacheKey = 'signatures:verification';
    if (this.resultCache.has(cacheKey)) return this.resultCache.get(cacheKey);
    let result: SignatureValidationResult = { isValid: true, signatures: [], issues: [] };
    if (this.native?.verify_signatures) {
      try {
        result = JSON.parse(this.native.verify_signatures());
      } catch {
        /* defaults */
      }
    }
    this.setCached(cacheKey, result);
    this.emit('signaturesVerified', { isValid: result.isValid, issueCount: result.issues.length });
    return result;
  }

  async verifySignature(signatureName: string): Promise<SignatureValidationResult> {
    const cacheKey = `signatures:verify:${signatureName}`;
    if (this.resultCache.has(cacheKey)) return this.resultCache.get(cacheKey);
    let result: SignatureValidationResult = { isValid: true, signatures: [], issues: [] };
    if (this.native?.verify_signature) {
      try {
        result = JSON.parse(this.native.verify_signature(signatureName));
      } catch {
        /* defaults */
      }
    }
    this.setCached(cacheKey, result);
    return result;
  }

  async isCertified(): Promise<boolean> {
    const cacheKey = 'signatures:is_certified';
    if (this.resultCache.has(cacheKey)) return this.resultCache.get(cacheKey);
    const result = this.document?.isCertified?.() ?? this.native?.is_certified?.() ?? false;
    this.setCached(cacheKey, result);
    return result;
  }

  async getSignatureCount(): Promise<number> {
    const cacheKey = 'signatures:count';
    if (this.resultCache.has(cacheKey)) return this.resultCache.get(cacheKey);
    const count = this.document?.getSignatureCount?.() ?? this.native?.get_signature_count?.() ?? 0;
    this.setCached(cacheKey, count);
    return count;
  }

  async isSigned(): Promise<boolean> {
    return (await this.getSignatureCount()) > 0;
  }

  // ===========================================================================
  // Certificate Loading (from SignatureCreationManager)
  // ===========================================================================

  async loadCertificateFromFile(
    filePath: string,
    password?: string,
    format?: CertificateFormat
  ): Promise<LoadedCertificate | null> {
    try {
      const certData = await fs.readFile(filePath);
      return this.loadCertificateFromBytes(certData, password, format);
    } catch (error) {
      this.emit('error', error);
      return null;
    }
  }

  async loadCertificateFromBytes(
    certData: Buffer,
    password?: string,
    format?: CertificateFormat
  ): Promise<LoadedCertificate | null> {
    try {
      const certId = `cert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const nativeResult = await this.document?.loadCertificate?.(certData, password, format);
      const info: CertificateInfo = nativeResult?.info ?? {
        subject: nativeResult?.subject ?? 'Unknown',
        issuer: nativeResult?.issuer ?? 'Unknown',
        serialNumber: nativeResult?.serialNumber ?? '',
        validFrom: new Date(nativeResult?.validFrom ?? Date.now()),
        validTo: new Date(nativeResult?.validTo ?? Date.now() + 365 * 24 * 60 * 60 * 1000),
        isValid: nativeResult?.isValid ?? false,
        isSelfSigned: nativeResult?.isSelfSigned ?? false,
      };
      const loadedCert: LoadedCertificate = {
        certificateId: certId,
        info,
        hasPrivateKey: nativeResult?.hasPrivateKey ?? true,
        chain: nativeResult?.chain,
      };
      this.loadedCertificates.set(certId, loadedCert);
      this.emit('certificate-loaded', { certificateId: certId, subject: info.subject });
      return loadedCert;
    } catch (error) {
      this.emit('error', error);
      return null;
    }
  }

  async loadCertificateFromPem(
    certificatePem: string,
    privateKeyPem?: string,
    privateKeyPassword?: string
  ): Promise<LoadedCertificate | null> {
    try {
      const certId = `cert_pem_${Date.now()}`;
      const nativeResult = await this.document?.loadCertificateFromPem?.(
        certificatePem,
        privateKeyPem,
        privateKeyPassword
      );
      if (!nativeResult) throw new Error('Failed to load PEM certificate');
      const info: CertificateInfo = {
        subject: nativeResult.subject ?? 'Unknown',
        issuer: nativeResult.issuer ?? 'Unknown',
        serialNumber: nativeResult.serialNumber ?? '',
        validFrom: new Date(nativeResult.validFrom ?? Date.now()),
        validTo: new Date(nativeResult.validTo ?? Date.now() + 365 * 24 * 60 * 60 * 1000),
        isValid: nativeResult.isValid ?? false,
        isSelfSigned: nativeResult.isSelfSigned ?? false,
      };
      const loadedCert: LoadedCertificate = {
        certificateId: certId,
        info,
        hasPrivateKey: !!privateKeyPem,
      };
      this.loadedCertificates.set(certId, loadedCert);
      this.emit('certificate-loaded', { certificateId: certId, subject: info.subject });
      return loadedCert;
    } catch (error) {
      this.emit('error', error);
      return null;
    }
  }

  async getCertificateInfo(certificateId: string): Promise<CertificateInfo | null> {
    return this.loadedCertificates.get(certificateId)?.info ?? null;
  }

  async getCertificateChain(certificateId: string): Promise<CertificateChain | null> {
    try {
      const cert = this.loadedCertificates.get(certificateId);
      if (!cert) return null;
      if (cert.chain) return cert.chain;
      return (await this.document?.getCertificateChain?.(certificateId)) ?? null;
    } catch (error) {
      this.emit('error', error);
      return null;
    }
  }

  async validateCertificate(
    certificateId: string
  ): Promise<{ valid: boolean; errors: string[]; warnings: string[] }> {
    try {
      const cert = this.loadedCertificates.get(certificateId);
      if (!cert) return { valid: false, errors: ['Certificate not found'], warnings: [] };
      const result = await this.document?.validateCertificate?.(certificateId);
      return (
        result ?? {
          valid: cert.info.isValid,
          errors: cert.info.isValid ? [] : ['Certificate validation failed'],
          warnings: [],
        }
      );
    } catch (error) {
      this.emit('error', error);
      return {
        valid: false,
        errors: [error instanceof Error ? error.message : 'Unknown error'],
        warnings: [],
      };
    }
  }

  getLoadedCertificates(): readonly LoadedCertificate[] {
    return Array.from(this.loadedCertificates.values());
  }

  unloadCertificate(certificateId: string): boolean {
    const deleted = this.loadedCertificates.delete(certificateId);
    if (deleted) this.emit('certificate-unloaded', { certificateId });
    return deleted;
  }

  // ===========================================================================
  // Signature Field Management (from SignatureCreationManager)
  // ===========================================================================

  async addSignatureField(config: SignatureFieldConfig): Promise<boolean> {
    try {
      const result = await this.document?.addSignatureField?.(
        config.fieldName,
        config.pageIndex,
        config.x,
        config.y,
        config.width,
        config.height,
        {
          appearance: config.appearance,
          tooltip: config.tooltip,
          isRequired: config.isRequired,
          isReadOnly: config.isReadOnly,
        }
      );
      if (result !== false) {
        this.createdFields.set(config.fieldName, config);
        this.clearCachePattern('signatures:');
        this.emit('field-added', { fieldName: config.fieldName });
      }
      return result !== false;
    } catch (error) {
      this.emit('error', error);
      return false;
    }
  }

  async removeSignatureField(fieldName: string): Promise<boolean> {
    try {
      const result = await this.document?.removeSignatureField?.(fieldName);
      if (result) {
        this.createdFields.delete(fieldName);
        this.clearCachePattern('signatures:');
        this.emit('field-removed', { fieldName });
      }
      return !!result;
    } catch (error) {
      this.emit('error', error);
      return false;
    }
  }

  async getSignatureFieldNames(): Promise<string[]> {
    try {
      return (await this.document?.getSignatureFields?.()) ?? [];
    } catch (error) {
      this.emit('error', error);
      return [];
    }
  }

  async hasSignatureField(fieldName: string): Promise<boolean> {
    try {
      const fields = await this.getSignatureFieldNames();
      return fields.includes(fieldName);
    } catch (error) {
      this.emit('error', error);
      return false;
    }
  }

  async updateSignatureFieldAppearance(
    fieldName: string,
    appearance: SignatureAppearance
  ): Promise<boolean> {
    try {
      const result = await this.document?.updateSignatureFieldAppearance?.(fieldName, appearance);
      this.emit('appearance-updated', { fieldName });
      return !!result;
    } catch (error) {
      this.emit('error', error);
      return false;
    }
  }

  // ===========================================================================
  // Document Signing (from SignatureCreationManager)
  // ===========================================================================

  async signDocument(
    fieldName: string,
    certificate: LoadedCertificate | string,
    options?: SigningOptions
  ): Promise<SigningResult> {
    try {
      const certId = typeof certificate === 'string' ? certificate : certificate.certificateId;
      const cert = this.loadedCertificates.get(certId);
      if (!cert) return { success: false, error: 'Certificate not found' };
      if (!cert.hasPrivateKey)
        return { success: false, error: 'Certificate does not have private key' };
      const nativeResult = await this.document?.signDocument?.(fieldName, certId, {
        reason: options?.reason,
        location: options?.location,
        contactInfo: options?.contactInfo,
        signatureType: options?.signatureType ?? SignatureType.APPROVAL,
        certificationPermission: options?.certificationPermission,
        algorithm: options?.algorithm ?? SignatureAlgorithm.RSA_SHA256,
        digestAlgorithm: options?.digestAlgorithm ?? DigestAlgorithm.SHA256,
        appearance: options?.appearance,
      });
      const signingTime = new Date();
      let timestampTime: Date | undefined;
      if (options?.embedTimestamp && options?.timestampServerUrl) {
        const tsResult = await this.embedTimestamp(fieldName, {
          serverUrl: options.timestampServerUrl,
          hashAlgorithm: options.digestAlgorithm,
        });
        if (tsResult.status === TimestampStatus.SUCCESS) timestampTime = tsResult.timestamp;
      }
      if (options?.enableLtv)
        await this.enableLtvForSignature(fieldName, {
          ocspResponderUrl: options.ocspResponderUrl,
          crlDistributionPoints: options.crlDistributionPoints,
        });
      const signatureId = `sig_${fieldName}_${Date.now()}`;
      this.clearCachePattern('signatures:');
      this.emit('document-signed', { signatureId, fieldName, signingTime, timestampTime });
      return {
        success: true,
        signatureId,
        signingTime,
        timestampTime,
        warnings: nativeResult?.warnings,
      };
    } catch (error) {
      this.emit('error', error);
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  async certifyDocument(
    fieldName: string,
    certificate: LoadedCertificate | string,
    permission: CertificationPermission,
    options?: Omit<SigningOptions, 'signatureType' | 'certificationPermission'>
  ): Promise<SigningResult> {
    return this.signDocument(fieldName, certificate, {
      ...options,
      signatureType: SignatureType.CERTIFICATION,
      certificationPermission: permission,
    });
  }

  async signInvisibly(
    certificate: LoadedCertificate | string,
    options?: Omit<SigningOptions, 'appearance'>
  ): Promise<SigningResult> {
    try {
      const fieldName = `InvisibleSig_${Date.now()}`;
      await this.addSignatureField({ fieldName, pageIndex: 0, x: 0, y: 0, width: 0, height: 0 });
      return this.signDocument(fieldName, certificate, options);
    } catch (error) {
      this.emit('error', error);
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  async counterSign(
    certificate: LoadedCertificate | string,
    options?: SigningOptions
  ): Promise<SigningResult> {
    try {
      const fieldName = `CounterSig_${Date.now()}`;
      await this.addSignatureField({
        fieldName,
        pageIndex: 0,
        x: 100,
        y: 100,
        width: 200,
        height: 50,
        appearance: options?.appearance,
      });
      return this.signDocument(fieldName, certificate, options);
    } catch (error) {
      this.emit('error', error);
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  async signMultipleFields(
    signings: Array<{
      fieldName: string;
      certificate: LoadedCertificate | string;
      options?: SigningOptions;
    }>
  ): Promise<SigningResult[]> {
    const results: SigningResult[] = [];
    for (const signing of signings) {
      const result = await this.signDocument(
        signing.fieldName,
        signing.certificate,
        signing.options
      );
      results.push(result);
      if (!result.success) break;
    }
    return results;
  }

  async prepareForExternalSigning(
    fieldName: string,
    options?: { estimatedSize?: number; digestAlgorithm?: DigestAlgorithm }
  ): Promise<{ hash: Buffer; byteRange: [number, number, number, number] } | null> {
    try {
      const result = await this.document?.prepareForExternalSigning?.(fieldName, {
        estimatedSize: options?.estimatedSize ?? 8192,
        digestAlgorithm: options?.digestAlgorithm ?? DigestAlgorithm.SHA256,
      });
      this.emit('prepared-for-external-signing', { fieldName });
      return result ?? null;
    } catch (error) {
      this.emit('error', error);
      return null;
    }
  }

  // ===========================================================================
  // Timestamp Operations (from SignatureCreationManager)
  // ===========================================================================

  async embedTimestamp(fieldName: string, config: TimestampConfig): Promise<TimestampResult> {
    try {
      const result = await this.document?.embedTimestamp?.(fieldName, {
        serverUrl: config.serverUrl,
        username: config.username,
        password: config.password,
        hashAlgorithm: config.hashAlgorithm ?? DigestAlgorithm.SHA256,
        timeout: config.timeout ?? 30000,
        policy: config.policy,
        nonce: config.nonce ?? true,
        certReq: config.certReq ?? true,
      });
      if (result?.success) {
        this.emit('timestamp-embedded', { fieldName, timestamp: result.timestamp });
        return {
          status: TimestampStatus.SUCCESS,
          timestamp: new Date(result.timestamp),
          serialNumber: result.serialNumber,
          tsaName: result.tsaName,
        };
      }
      return {
        status: TimestampStatus.FAILED,
        error: result?.error ?? 'Timestamp embedding failed',
      };
    } catch (error) {
      this.emit('error', error);
      return {
        status: TimestampStatus.FAILED,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  async addDocumentTimestamp(config: TimestampConfig): Promise<TimestampResult> {
    try {
      const fieldName = `DocTimestamp_${Date.now()}`;
      await this.addSignatureField({ fieldName, pageIndex: 0, x: 0, y: 0, width: 0, height: 0 });
      const result = await this.document?.addDocumentTimestamp?.(fieldName, config);
      if (result?.success) {
        this.emit('document-timestamp-added', { timestamp: result.timestamp });
        return {
          status: TimestampStatus.SUCCESS,
          timestamp: new Date(result.timestamp),
          serialNumber: result.serialNumber,
          tsaName: result.tsaName,
        };
      }
      return {
        status: TimestampStatus.FAILED,
        error: result?.error ?? 'Document timestamp failed',
      };
    } catch (error) {
      this.emit('error', error);
      return {
        status: TimestampStatus.FAILED,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  async validateTimestamp(
    fieldName: string
  ): Promise<{ valid: boolean; timestamp?: Date; errors: string[] }> {
    try {
      return (
        (await this.document?.validateTimestamp?.(fieldName)) ?? {
          valid: false,
          errors: ['Timestamp validation not available'],
        }
      );
    } catch (error) {
      this.emit('error', error);
      return { valid: false, errors: [error instanceof Error ? error.message : 'Unknown error'] };
    }
  }

  async getTimestampInfo(fieldName: string): Promise<{
    timestamp?: Date;
    tsaName?: string;
    serialNumber?: string;
    policy?: string;
  } | null> {
    try {
      return (await this.document?.getTimestampInfo?.(fieldName)) ?? null;
    } catch (error) {
      this.emit('error', error);
      return null;
    }
  }

  // ===========================================================================
  // LTV Operations (from SignatureCreationManager)
  // ===========================================================================

  async enableLtvForSignature(
    fieldName: string,
    options?: { ocspResponderUrl?: string; crlDistributionPoints?: readonly string[] }
  ): Promise<boolean> {
    try {
      const result = await this.document?.enableLtvForSignature?.(fieldName, {
        ocspResponderUrl: options?.ocspResponderUrl,
        crlDistributionPoints: options?.crlDistributionPoints,
      });
      if (result) this.emit('ltv-enabled', { fieldName });
      return !!result;
    } catch (error) {
      this.emit('error', error);
      return false;
    }
  }

  async enableLtvForAllSignatures(options?: {
    ocspResponderUrl?: string;
    crlDistributionPoints?: readonly string[];
  }): Promise<number> {
    try {
      const fields = await this.getSignatureFieldNames();
      let count = 0;
      for (const field of fields) {
        if (await this.enableLtvForSignature(field, options)) count++;
      }
      return count;
    } catch (error) {
      this.emit('error', error);
      return 0;
    }
  }

  async addValidationInfo(
    fieldName: string,
    info: { ocspResponse?: Buffer; crl?: Buffer; certificates?: readonly Buffer[] }
  ): Promise<boolean> {
    try {
      const result = await this.document?.addValidationInfo?.(fieldName, info);
      if (result) this.emit('validation-info-added', { fieldName });
      return !!result;
    } catch (error) {
      this.emit('error', error);
      return false;
    }
  }

  async hasLtvEnabled(fieldName: string): Promise<boolean> {
    try {
      return (await this.document?.hasLtvEnabled?.(fieldName)) ?? false;
    } catch (error) {
      this.emit('error', error);
      return false;
    }
  }

  // ===========================================================================
  // Signature Details (from SignaturesManager)
  // ===========================================================================

  async getSignerName(index: number): Promise<string> {
    try {
      return (await this.document?.getSignerName?.(index)) || '';
    } catch {
      return '';
    }
  }

  async getSigningTime(index: number): Promise<number> {
    try {
      return (await this.document?.getSigningTime?.(index)) || 0;
    } catch {
      return 0;
    }
  }

  async getSigningReason(index: number): Promise<string | null> {
    try {
      return (await this.document?.getSigningReason?.(index)) || null;
    } catch {
      return null;
    }
  }

  async getSigningLocation(index: number): Promise<string | null> {
    try {
      return (await this.document?.getSigningLocation?.(index)) || null;
    } catch {
      return null;
    }
  }

  async getCertificateSubject(index: number): Promise<string> {
    try {
      return (await this.document?.getCertificateSubject?.(index)) || '';
    } catch {
      return '';
    }
  }

  async getCertificateIssuer(index: number): Promise<string> {
    try {
      return (await this.document?.getCertificateIssuer?.(index)) || '';
    } catch {
      return '';
    }
  }

  async getCertificateSerial(index: number): Promise<string> {
    try {
      return (await this.document?.getCertificateSerial?.(index)) || '';
    } catch {
      return '';
    }
  }

  async getCertificateValidity(index: number): Promise<[number, number]> {
    try {
      return (await this.document?.getCertificateValidity?.(index)) || [0, 0];
    } catch {
      return [0, 0];
    }
  }

  async getSignatureDetails(index: number): Promise<Signature | null> {
    try {
      const [notBefore, notAfter] = await this.getCertificateValidity(index);
      const certificate: Certificate = {
        subject: await this.getCertificateSubject(index),
        issuer: await this.getCertificateIssuer(index),
        serial: await this.getCertificateSerial(index),
        notBefore,
        notAfter,
        isValid: await this.isCertificateValidByIndex(index),
      };
      return {
        signerName: await this.getSignerName(index),
        signingTime: await this.getSigningTime(index),
        reason: (await this.getSigningReason(index)) || undefined,
        location: (await this.getSigningLocation(index)) || undefined,
        certificate,
        isValid: true,
      };
    } catch (error) {
      this.emit('error', error);
      return null;
    }
  }

  async isCertificateValidByIndex(index: number): Promise<boolean> {
    try {
      return (await this.document?.isCertificateValid?.(index)) || false;
    } catch {
      return false;
    }
  }

  // ===========================================================================
  // FFI-Based Document Signing (Phase 1)
  // ===========================================================================

  /**
   * Load signing credentials from a PKCS#12 (.p12/.pfx) file.
   *
   * Calls the native `pdf_credentials_from_pkcs12` FFI function to load
   * a certificate and private key from a PKCS#12 container.
   *
   * @param filePath - Path to the .p12 or .pfx file
   * @param password - Password to decrypt the PKCS#12 file
   * @returns SigningCredentials handle for use with signing methods
   * @throws SignatureException if the file cannot be loaded or the password is incorrect
   *
   * @example
   * ```typescript
   * const credentials = await sigManager.loadCredentialsPkcs12('/path/to/cert.p12', 'password');
   * const signed = await sigManager.signWithPkcs12(pdfData, '/path/to/cert.p12', 'password');
   * ```
   */
  async loadCredentialsPkcs12(filePath: string, password: string): Promise<SigningCredentials> {
    if (!this.native?.pdf_credentials_from_pkcs12) {
      throw new SignatureException(
        'Native signing not available: pdf_credentials_from_pkcs12 not found'
      );
    }

    const errorCode = Buffer.alloc(4);
    const handle = this.native.pdf_credentials_from_pkcs12(filePath, password, errorCode);
    const code = errorCode.readInt32LE(0);

    if (code !== 0 || !handle) {
      throw mapFfiErrorCode(code, `Failed to load PKCS#12 credentials from ${filePath}`);
    }

    this.emit('credentials-loaded', { sourceType: 'pkcs12', filePath });
    return { _handle: handle, sourceType: 'pkcs12' };
  }

  /**
   * Load signing credentials from PEM certificate and key files.
   *
   * Calls the native `pdf_credentials_from_pem` FFI function to load
   * credentials from separate PEM-encoded certificate and private key files.
   *
   * @param certFile - Path to the PEM certificate file
   * @param keyFile - Path to the PEM private key file
   * @param keyPassword - Optional password for an encrypted private key
   * @returns SigningCredentials handle for use with signing methods
   * @throws SignatureException if the files cannot be loaded
   *
   * @example
   * ```typescript
   * const credentials = await sigManager.loadCredentialsPem('/path/to/cert.pem', '/path/to/key.pem');
   * ```
   */
  async loadCredentialsPem(
    certFile: string,
    keyFile: string,
    keyPassword?: string
  ): Promise<SigningCredentials> {
    if (!this.native?.pdf_credentials_from_pem) {
      throw new SignatureException(
        'Native signing not available: pdf_credentials_from_pem not found'
      );
    }

    const errorCode = Buffer.alloc(4);
    const handle = this.native.pdf_credentials_from_pem(
      certFile,
      keyFile,
      keyPassword ?? null,
      errorCode
    );
    const code = errorCode.readInt32LE(0);

    if (code !== 0 || !handle) {
      throw mapFfiErrorCode(code, `Failed to load PEM credentials from ${certFile} and ${keyFile}`);
    }

    this.emit('credentials-loaded', { sourceType: 'pem', certFile, keyFile });
    return { _handle: handle, sourceType: 'pem' };
  }

  /**
   * Free signing credentials when they are no longer needed.
   *
   * Calls the native `pdf_credentials_free` FFI function to release
   * memory associated with the credentials handle.
   *
   * @param credentials - The credentials handle to free
   *
   * @example
   * ```typescript
   * const credentials = await sigManager.loadCredentialsPkcs12(path, password);
   * // ... use credentials for signing ...
   * sigManager.freeCredentials(credentials);
   * ```
   */
  freeCredentials(credentials: SigningCredentials): void {
    if (credentials._handle && this.native?.pdf_credentials_free) {
      this.native.pdf_credentials_free(credentials._handle);
      this.emit('credentials-freed', { sourceType: credentials.sourceType });
    }
  }

  /**
   * Sign a PDF document in memory using PKCS#12 credentials.
   *
   * Loads credentials from a PKCS#12 file, signs the PDF data, and returns
   * the signed PDF bytes. Credentials are automatically freed after signing.
   *
   * @param pdfData - Buffer containing the PDF document bytes
   * @param filePath - Path to the .p12 or .pfx certificate file
   * @param password - Password for the PKCS#12 file
   * @param options - Optional signing parameters (reason, location, contact, algorithm, subfilter)
   * @returns Buffer containing the signed PDF document
   * @throws SignatureException if credential loading or signing fails
   *
   * @example
   * ```typescript
   * const pdfData = await fs.readFile('document.pdf');
   * const signed = await sigManager.signWithPkcs12(pdfData, 'cert.p12', 'password', {
   *   reason: 'Approval',
   *   location: 'New York',
   *   contact: 'signer@example.com',
   * });
   * await fs.writeFile('signed.pdf', signed);
   * ```
   */
  async signWithPkcs12(
    pdfData: Buffer,
    filePath: string,
    password: string,
    options?: SignOptions
  ): Promise<Buffer> {
    const credentials = await this.loadCredentialsPkcs12(filePath, password);
    try {
      return await this.signWithCredentials(pdfData, credentials, options);
    } finally {
      this.freeCredentials(credentials);
    }
  }

  /**
   * Sign a PDF document in memory using PEM credentials.
   *
   * Loads credentials from PEM files, signs the PDF data, and returns
   * the signed PDF bytes. Credentials are automatically freed after signing.
   *
   * @param pdfData - Buffer containing the PDF document bytes
   * @param certFile - Path to the PEM certificate file
   * @param keyFile - Path to the PEM private key file
   * @param options - Optional signing parameters (reason, location, contact, algorithm, subfilter)
   * @returns Buffer containing the signed PDF document
   * @throws SignatureException if credential loading or signing fails
   *
   * @example
   * ```typescript
   * const pdfData = await fs.readFile('document.pdf');
   * const signed = await sigManager.signWithPem(pdfData, 'cert.pem', 'key.pem', {
   *   reason: 'Review complete',
   *   location: 'London',
   * });
   * await fs.writeFile('signed.pdf', signed);
   * ```
   */
  async signWithPem(
    pdfData: Buffer,
    certFile: string,
    keyFile: string,
    options?: SignOptions
  ): Promise<Buffer> {
    const credentials = await this.loadCredentialsPem(certFile, keyFile);
    try {
      return await this.signWithCredentials(pdfData, credentials, options);
    } finally {
      this.freeCredentials(credentials);
    }
  }

  /**
   * Sign a PDF document in memory using pre-loaded credentials.
   *
   * Calls the native `pdf_document_sign` FFI function to apply a digital
   * signature to the PDF data using the provided credentials handle.
   *
   * @param pdfData - Buffer containing the PDF document bytes
   * @param credentials - Pre-loaded signing credentials handle
   * @param options - Optional signing parameters
   * @returns Buffer containing the signed PDF document
   * @throws SignatureException if signing fails
   *
   * @example
   * ```typescript
   * const credentials = await sigManager.loadCredentialsPkcs12(path, password);
   * const signed = await sigManager.signWithCredentials(pdfData, credentials, {
   *   reason: 'Approved',
   *   algorithm: FfiDigestAlgorithm.SHA256,
   *   subfilter: FfiSignatureSubFilter.PKCS7_DETACHED,
   * });
   * sigManager.freeCredentials(credentials);
   * ```
   */
  async signWithCredentials(
    pdfData: Buffer,
    credentials: SigningCredentials,
    options?: SignOptions
  ): Promise<Buffer> {
    if (!this.native?.pdf_document_sign) {
      throw new SignatureException('Native signing not available: pdf_document_sign not found');
    }

    if (!credentials._handle) {
      throw new SignatureException('Invalid credentials: handle is null');
    }

    const errorCode = Buffer.alloc(4);
    const outDataPtr = Buffer.alloc(8); // pointer to output data
    const outLen = Buffer.alloc(8); // output length (size_t)

    const algorithm = options?.algorithm ?? FfiDigestAlgorithm.SHA256;
    const subfilter = options?.subfilter ?? FfiSignatureSubFilter.PKCS7_DETACHED;

    const success = this.native.pdf_document_sign(
      pdfData,
      pdfData.length,
      credentials._handle,
      options?.reason ?? null,
      options?.location ?? null,
      options?.contact ?? null,
      algorithm,
      subfilter,
      outDataPtr,
      outLen,
      errorCode
    );

    const code = errorCode.readInt32LE(0);

    if (!success || code !== 0) {
      throw mapFfiErrorCode(code, 'Failed to sign PDF document');
    }

    // Read the output buffer from the native pointer
    const resultLen = Number(outLen.readBigUInt64LE(0));
    const resultData = this.native.pdf_read_signed_bytes(outDataPtr, resultLen);

    // Free the native signed bytes buffer
    if (this.native.pdf_signed_bytes_free) {
      this.native.pdf_signed_bytes_free(outDataPtr, resultLen);
    }

    this.clearCachePattern('signatures:');
    this.emit('document-signed-ffi', {
      reason: options?.reason,
      location: options?.location,
      algorithm,
      subfilter,
    });

    return Buffer.from(resultData);
  }

  /**
   * Sign a PDF file on disk and write the signed output to another file.
   *
   * Calls the native `pdf_document_sign_file` FFI function, which reads
   * the input file, applies a digital signature, and writes the result
   * to the output path.
   *
   * @param inputPath - Path to the input PDF file
   * @param outputPath - Path to write the signed PDF file
   * @param credentials - Pre-loaded signing credentials handle
   * @param options - Optional signing parameters
   * @throws SignatureException if file signing fails
   *
   * @example
   * ```typescript
   * const credentials = await sigManager.loadCredentialsPkcs12('cert.p12', 'pass');
   * await sigManager.signFile('input.pdf', 'signed.pdf', credentials, {
   *   reason: 'Final approval',
   *   location: 'Berlin',
   * });
   * sigManager.freeCredentials(credentials);
   * ```
   */
  async signFile(
    inputPath: string,
    outputPath: string,
    credentials: SigningCredentials,
    options?: SignOptions
  ): Promise<void> {
    if (!this.native?.pdf_document_sign_file) {
      throw new SignatureException(
        'Native signing not available: pdf_document_sign_file not found'
      );
    }

    if (!credentials._handle) {
      throw new SignatureException('Invalid credentials: handle is null');
    }

    const errorCode = Buffer.alloc(4);
    const algorithm = options?.algorithm ?? FfiDigestAlgorithm.SHA256;
    const subfilter = options?.subfilter ?? FfiSignatureSubFilter.PKCS7_DETACHED;

    const success = this.native.pdf_document_sign_file(
      inputPath,
      outputPath,
      credentials._handle,
      options?.reason ?? null,
      options?.location ?? null,
      options?.contact ?? null,
      algorithm,
      subfilter,
      errorCode
    );

    const code = errorCode.readInt32LE(0);

    if (!success || code !== 0) {
      throw mapFfiErrorCode(code, `Failed to sign PDF file ${inputPath}`);
    }

    this.clearCachePattern('signatures:');
    this.emit('file-signed', { inputPath, outputPath });
  }

  /**
   * Embed Long-Term Validation (LTV) data into a signed PDF.
   *
   * Calls the native `pdf_embed_ltv_data` FFI function to add OCSP responses
   * and/or CRL data to the document's DSS (Document Security Store), enabling
   * long-term signature validation even after certificates expire.
   *
   * @param pdfData - Buffer containing the signed PDF document bytes
   * @param ocspData - Optional OCSP response data to embed
   * @param crlData - Optional CRL data to embed
   * @returns Buffer containing the PDF with embedded LTV data
   * @throws SignatureException if LTV embedding fails
   *
   * @example
   * ```typescript
   * const signedPdf = await sigManager.signWithPkcs12(pdfData, 'cert.p12', 'pass');
   * const ocspResponse = await fetch('http://ocsp.example.com/...').then(r => r.buffer());
   * const ltvPdf = await sigManager.embedLtv(signedPdf, ocspResponse);
   * await fs.writeFile('signed-ltv.pdf', ltvPdf);
   * ```
   */
  async embedLtv(pdfData: Buffer, ocspData?: Buffer, crlData?: Buffer): Promise<Buffer> {
    if (!this.native?.pdf_embed_ltv_data) {
      throw new SignatureException(
        'Native LTV embedding not available: pdf_embed_ltv_data not found'
      );
    }

    const errorCode = Buffer.alloc(4);
    const outDataPtr = Buffer.alloc(8);
    const outLen = Buffer.alloc(8);

    const success = this.native.pdf_embed_ltv_data(
      pdfData,
      pdfData.length,
      ocspData ?? null,
      ocspData?.length ?? 0,
      crlData ?? null,
      crlData?.length ?? 0,
      outDataPtr,
      outLen,
      errorCode
    );

    const code = errorCode.readInt32LE(0);

    if (!success || code !== 0) {
      throw mapFfiErrorCode(code, 'Failed to embed LTV data into PDF');
    }

    const resultLen = Number(outLen.readBigUInt64LE(0));
    const resultData = this.native.pdf_read_signed_bytes(outDataPtr, resultLen);

    if (this.native.pdf_signed_bytes_free) {
      this.native.pdf_signed_bytes_free(outDataPtr, resultLen);
    }

    this.emit('ltv-embedded-ffi', {
      hasOcsp: !!ocspData,
      hasCrl: !!crlData,
    });

    return Buffer.from(resultData);
  }

  /**
   * Save signed PDF bytes to a file.
   *
   * Calls the native `pdf_document_save_signed` FFI function to write
   * signed PDF data to disk.
   *
   * @param pdfData - Buffer containing the signed PDF bytes
   * @param outputPath - Path to write the signed PDF file
   * @throws SignatureException or IoException if saving fails
   *
   * @example
   * ```typescript
   * const signed = await sigManager.signWithPkcs12(pdfData, 'cert.p12', 'pass');
   * await sigManager.saveSigned(signed, '/output/signed.pdf');
   * ```
   */
  async saveSigned(pdfData: Buffer, outputPath: string): Promise<void> {
    if (!this.native?.pdf_document_save_signed) {
      // Fallback to Node.js file I/O if native function is not available
      await fs.writeFile(outputPath, pdfData);
      this.emit('signed-saved', { outputPath, method: 'node-fs' });
      return;
    }

    const errorCode = Buffer.alloc(4);
    const success = this.native.pdf_document_save_signed(
      pdfData,
      pdfData.length,
      outputPath,
      errorCode
    );

    const code = errorCode.readInt32LE(0);

    if (!success || code !== 0) {
      throw mapFfiErrorCode(code, `Failed to save signed PDF to ${outputPath}`);
    }

    this.emit('signed-saved', { outputPath, method: 'native-ffi' });
  }

  // ===========================================================================
  // FFI-Based DER Credential Loading
  // ===========================================================================

  /**
   * Load signing credentials from raw DER-encoded certificate and key bytes.
   *
   * Calls the native `pdf_credentials_from_der` FFI function to create
   * credentials from in-memory DER-encoded certificate data and an optional
   * private key.
   *
   * @param certData - Buffer containing DER-encoded certificate bytes
   * @param keyData - Optional Buffer containing DER-encoded private key bytes
   * @returns SigningCredentials handle for use with signing methods
   * @throws SignatureException if credential loading fails
   *
   * @example
   * ```typescript
   * const certDer = await fs.readFile('cert.der');
   * const keyDer = await fs.readFile('key.der');
   * const credentials = await sigManager.loadCredentialsFromDer(certDer, keyDer);
   * const signed = await sigManager.signWithCredentials(pdfData, credentials);
   * sigManager.freeCredentials(credentials);
   * ```
   */
  async loadCredentialsFromDer(certData: Buffer, keyData?: Buffer): Promise<SigningCredentials> {
    if (!this.native?.pdf_credentials_from_der) {
      throw new SignatureException(
        'Native signing not available: pdf_credentials_from_der not found'
      );
    }

    if (!certData || certData.length === 0) {
      throw new SignatureException('Certificate data cannot be null or empty');
    }

    const errorCode = Buffer.alloc(4);
    const handle = this.native.pdf_credentials_from_der(
      certData,
      certData.length,
      keyData ?? null,
      keyData?.length ?? 0,
      errorCode
    );
    const code = errorCode.readInt32LE(0);

    if (code !== 0 || !handle) {
      throw mapFfiErrorCode(code, 'Failed to load DER credentials');
    }

    this.emit('credentials-loaded', { sourceType: 'der' });
    return { _handle: handle, sourceType: 'pkcs12' as const }; // DER is treated as raw cert/key
  }

  /**
   * Add a certificate chain entry to existing signing credentials.
   *
   * Calls the native `pdf_credentials_add_chain_cert` FFI function to append
   * an intermediate or root CA certificate to the credential's certificate chain.
   * This is used to build a complete certification chain for signature validation.
   *
   * @param credentials - The signing credentials handle to modify
   * @param certData - Buffer containing DER-encoded certificate bytes
   * @throws SignatureException if the chain certificate cannot be added
   *
   * @example
   * ```typescript
   * const credentials = await sigManager.loadCredentialsFromDer(certDer, keyDer);
   * const intermediateCa = await fs.readFile('intermediate-ca.der');
   * await sigManager.addChainCert(credentials, intermediateCa);
   * ```
   */
  async addChainCert(credentials: SigningCredentials, certData: Buffer): Promise<void> {
    if (!this.native?.pdf_credentials_add_chain_cert) {
      throw new SignatureException(
        'Native signing not available: pdf_credentials_add_chain_cert not found'
      );
    }

    if (!credentials._handle) {
      throw new SignatureException('Invalid credentials: handle is null');
    }

    if (!certData || certData.length === 0) {
      throw new SignatureException('Certificate data cannot be null or empty');
    }

    const errorCode = Buffer.alloc(4);
    const success = this.native.pdf_credentials_add_chain_cert(
      credentials._handle,
      certData,
      certData.length,
      errorCode
    );
    const code = errorCode.readInt32LE(0);

    if (!success || code !== 0) {
      throw mapFfiErrorCode(code, 'Failed to add chain certificate');
    }

    this.emit('chain-cert-added', { sourceType: credentials.sourceType });
  }

  /**
   * Get the certificate handle from signing credentials.
   *
   * Calls the native `pdf_credentials_get_certificate` FFI function to extract
   * the certificate handle from a credentials object. The returned handle can be
   * used with certificate inspection methods like getCertificateCn, getCertificateIssuer,
   * and getCertificateSize.
   *
   * @param credentials - The signing credentials handle
   * @returns An opaque certificate handle for use with certificate inspection methods
   * @throws SignatureException if the certificate cannot be retrieved
   *
   * @example
   * ```typescript
   * const credentials = await sigManager.loadCredentialsPkcs12('cert.p12', 'pass');
   * const certHandle = await sigManager.getCertificate(credentials);
   * const cn = await sigManager.getCertificateCn(certHandle);
   * console.log(`Certificate CN: ${cn}`);
   * sigManager.freeCredentials(credentials);
   * ```
   */
  async getCertificate(credentials: SigningCredentials): Promise<any> {
    if (!this.native?.pdf_credentials_get_certificate) {
      throw new SignatureException(
        'Native signing not available: pdf_credentials_get_certificate not found'
      );
    }

    if (!credentials._handle) {
      throw new SignatureException('Invalid credentials: handle is null');
    }

    const errorCode = Buffer.alloc(4);
    const certHandle = this.native.pdf_credentials_get_certificate(credentials._handle, errorCode);
    const code = errorCode.readInt32LE(0);

    if (code !== 0 || !certHandle) {
      throw mapFfiErrorCode(code, 'Failed to get certificate from credentials');
    }

    return certHandle;
  }

  /**
   * Load a certificate from raw DER bytes for inspection.
   *
   * Calls the native `pdf_certificate_load_from_bytes` FFI function to create
   * a certificate handle from DER-encoded bytes. This is useful for inspecting
   * certificate properties without creating full signing credentials.
   *
   * @param certData - Buffer containing DER-encoded certificate bytes
   * @returns An opaque certificate handle for use with certificate inspection methods
   * @throws SignatureException if the certificate cannot be loaded
   *
   * @example
   * ```typescript
   * const certDer = await fs.readFile('cert.der');
   * const certHandle = await sigManager.loadCertificateFromDerBytes(certDer);
   * const cn = await sigManager.getCertificateCn(certHandle);
   * const issuer = await sigManager.getCertificateIssuerFromHandle(certHandle);
   * const size = await sigManager.getCertificateSize(certHandle);
   * ```
   */
  async loadCertificateFromDerBytes(certData: Buffer): Promise<any> {
    if (!this.native?.pdf_certificate_load_from_bytes) {
      throw new SignatureException(
        'Native signing not available: pdf_certificate_load_from_bytes not found'
      );
    }

    if (!certData || certData.length === 0) {
      throw new SignatureException('Certificate data cannot be null or empty');
    }

    const errorCode = Buffer.alloc(4);
    const certHandle = this.native.pdf_certificate_load_from_bytes(
      certData,
      certData.length,
      errorCode
    );
    const code = errorCode.readInt32LE(0);

    if (code !== 0 || !certHandle) {
      throw mapFfiErrorCode(code, 'Failed to load certificate from bytes');
    }

    return certHandle;
  }

  /**
   * Get the common name (CN) from a certificate handle.
   *
   * Calls the native `pdf_certificate_get_cn` FFI function to extract
   * the subject common name from a certificate.
   *
   * @param certHandle - An opaque certificate handle obtained from getCertificate or loadCertificateFromDerBytes
   * @returns The certificate common name string
   * @throws SignatureException if the CN cannot be retrieved
   *
   * @example
   * ```typescript
   * const credentials = await sigManager.loadCredentialsPkcs12('cert.p12', 'pass');
   * const certHandle = await sigManager.getCertificate(credentials);
   * const cn = await sigManager.getCertificateCn(certHandle);
   * console.log(`Signer: ${cn}`);
   * ```
   */
  async getCertificateCn(certHandle: any): Promise<string> {
    if (!this.native?.pdf_certificate_get_cn) {
      throw new SignatureException(
        'Native signing not available: pdf_certificate_get_cn not found'
      );
    }

    if (!certHandle) {
      throw new SignatureException('Invalid certificate handle: handle is null');
    }

    const errorCode = Buffer.alloc(4);
    const resultPtr = this.native.pdf_certificate_get_cn(certHandle, errorCode);
    const code = errorCode.readInt32LE(0);

    if (code !== 0 || !resultPtr) {
      throw mapFfiErrorCode(code, 'Failed to get certificate CN');
    }

    const cn = typeof resultPtr === 'string' ? resultPtr : resultPtr.toString();
    return cn;
  }

  /**
   * Get the issuer name from a certificate handle.
   *
   * Calls the native `pdf_certificate_get_issuer` FFI function to extract
   * the issuer distinguished name from a certificate.
   *
   * @param certHandle - An opaque certificate handle obtained from getCertificate or loadCertificateFromDerBytes
   * @returns The certificate issuer name string
   * @throws SignatureException if the issuer cannot be retrieved
   *
   * @example
   * ```typescript
   * const certHandle = await sigManager.loadCertificateFromDerBytes(certDer);
   * const issuer = await sigManager.getCertificateIssuerFromHandle(certHandle);
   * console.log(`Issued by: ${issuer}`);
   * ```
   */
  async getCertificateIssuerFromHandle(certHandle: any): Promise<string> {
    if (!this.native?.pdf_certificate_get_issuer) {
      throw new SignatureException(
        'Native signing not available: pdf_certificate_get_issuer not found'
      );
    }

    if (!certHandle) {
      throw new SignatureException('Invalid certificate handle: handle is null');
    }

    const errorCode = Buffer.alloc(4);
    const resultPtr = this.native.pdf_certificate_get_issuer(certHandle, errorCode);
    const code = errorCode.readInt32LE(0);

    if (code !== 0 || !resultPtr) {
      throw mapFfiErrorCode(code, 'Failed to get certificate issuer');
    }

    const issuer = typeof resultPtr === 'string' ? resultPtr : resultPtr.toString();
    return issuer;
  }

  /**
   * Get the size in bytes of a certificate.
   *
   * Calls the native `pdf_certificate_get_size` FFI function to get the
   * size of the DER-encoded certificate data.
   *
   * @param certHandle - An opaque certificate handle obtained from getCertificate or loadCertificateFromDerBytes
   * @returns The certificate size in bytes
   * @throws SignatureException if the size cannot be retrieved
   *
   * @example
   * ```typescript
   * const certHandle = await sigManager.loadCertificateFromDerBytes(certDer);
   * const size = await sigManager.getCertificateSize(certHandle);
   * console.log(`Certificate size: ${size} bytes`);
   * ```
   */
  async getCertificateSize(certHandle: any): Promise<number> {
    if (!this.native?.pdf_certificate_get_size) {
      throw new SignatureException(
        'Native signing not available: pdf_certificate_get_size not found'
      );
    }

    if (!certHandle) {
      throw new SignatureException('Invalid certificate handle: handle is null');
    }

    const errorCode = Buffer.alloc(4);
    const size = this.native.pdf_certificate_get_size(certHandle, errorCode);
    const code = errorCode.readInt32LE(0);

    if (code !== 0) {
      throw mapFfiErrorCode(code, 'Failed to get certificate size');
    }

    return typeof size === 'number' ? size : Number(size);
  }

  /**
   * Free a certificate handle when it is no longer needed.
   *
   * Calls the native `pdf_certificate_free` FFI function to release
   * memory associated with the certificate handle.
   *
   * @param certHandle - The certificate handle to free
   *
   * @example
   * ```typescript
   * const certHandle = await sigManager.loadCertificateFromDerBytes(certDer);
   * const cn = await sigManager.getCertificateCn(certHandle);
   * sigManager.freeCertificate(certHandle);
   * ```
   */
  freeCertificate(certHandle: any): void {
    if (certHandle && this.native?.pdf_certificate_free) {
      this.native.pdf_certificate_free(certHandle);
    }
  }

  /**
   * Add an RFC 3161 timestamp to an existing signature via a Time Stamp Authority.
   *
   * Calls the native `pdf_add_timestamp` FFI function.
   *
   * @param pdfData - Buffer containing the signed PDF document bytes
   * @param signatureIndex - Index of the signature to timestamp (0-based)
   * @param tsaUrl - URL of the Time Stamp Authority server
   * @returns Buffer containing the timestamped PDF bytes
   * @throws SignatureException if timestamping fails
   */
  async addTimestamp(pdfData: Buffer, signatureIndex: number, tsaUrl: string): Promise<Buffer> {
    if (!this.native?.pdf_add_timestamp) {
      throw new SignatureException(
        'Native timestamping not available: pdf_add_timestamp not found'
      );
    }

    if (!pdfData || pdfData.length === 0) {
      throw new SignatureException('Invalid pdfData: buffer is empty or null');
    }

    if (!tsaUrl || tsaUrl.length === 0) {
      throw new SignatureException('Invalid tsaUrl: URL is empty or null');
    }

    const errorCode = Buffer.alloc(4);
    const outDataPtr = Buffer.alloc(8);
    const outLen = Buffer.alloc(8);

    const success = this.native.pdf_add_timestamp(
      pdfData,
      pdfData.length,
      signatureIndex,
      tsaUrl,
      outDataPtr,
      outLen,
      errorCode
    );

    const code = errorCode.readInt32LE(0);

    if (!success || code !== 0) {
      throw mapFfiErrorCode(code, 'Failed to add timestamp to PDF signature');
    }

    const resultLen = Number(outLen.readBigUInt64LE(0));
    const resultData = this.native.pdf_read_signed_bytes(outDataPtr, resultLen);

    if (this.native.pdf_signed_bytes_free) {
      this.native.pdf_signed_bytes_free(outDataPtr, resultLen);
    }

    this.emit('timestamp-added', {
      signatureIndex,
      tsaUrl,
    });

    return Buffer.from(resultData);
  }

  /**
   * Sign PDF data with a visible signature appearance on a specific page.
   *
   * @param pdfData - Buffer containing PDF bytes to sign
   * @param credentials - Pre-loaded signing credentials
   * @param pageNum - Page number for appearance (0-based)
   * @param x - X coordinate of appearance box
   * @param y - Y coordinate of appearance box
   * @param width - Width of appearance box
   * @param height - Height of appearance box
   * @param options - Optional signing parameters
   * @returns Buffer containing signed PDF bytes
   * @throws SignatureException if signing fails
   */
  async signWithAppearance(
    pdfData: Buffer,
    credentials: SigningCredentials,
    pageNum: number,
    x: number,
    y: number,
    width: number,
    height: number,
    options?: SignOptions
  ): Promise<Buffer> {
    if (!this.native?.pdf_document_sign_with_appearance) {
      throw new SignatureException(
        'Native signing not available: pdf_document_sign_with_appearance not found'
      );
    }

    if (!credentials._handle) {
      throw new SignatureException('Invalid credentials: handle is null');
    }

    const errorCode = Buffer.alloc(4);
    const outDataPtr = Buffer.alloc(8);
    const outLen = Buffer.alloc(8);

    const algorithm = options?.algorithm ?? FfiDigestAlgorithm.SHA256;

    const success = this.native.pdf_document_sign_with_appearance(
      pdfData,
      pdfData.length,
      credentials._handle,
      pageNum,
      x,
      y,
      width,
      height,
      options?.reason ?? null,
      options?.location ?? null,
      options?.contact ?? null,
      algorithm,
      outDataPtr,
      outLen,
      errorCode
    );

    const code = errorCode.readInt32LE(0);

    if (!success || code !== 0) {
      throw mapFfiErrorCode(code, 'Failed to sign PDF with appearance');
    }

    const resultLen = Number(outLen.readBigUInt64LE(0));
    const resultData = this.native.pdf_read_signed_bytes(outDataPtr, resultLen);

    if (this.native.pdf_signed_bytes_free) {
      this.native.pdf_signed_bytes_free(outDataPtr, resultLen);
    }

    this.clearCachePattern('signatures:');
    this.emit('signed-with-appearance', {
      pageNum,
      x,
      y,
      width,
      height,
      reason: options?.reason,
      location: options?.location,
      algorithm,
    });

    return Buffer.from(resultData);
  }

  // ===========================================================================
  // Cache
  // ===========================================================================

  clearCache(): void {
    this.resultCache.clear();
    this.emit('cacheCleared');
  }

  getCacheStats(): Record<string, any> {
    return {
      cacheSize: this.resultCache.size,
      maxCacheSize: this.maxCacheSize,
      entries: Array.from(this.resultCache.keys()),
    };
  }

  destroy(): void {
    this.loadedCertificates.clear();
    this.createdFields.clear();
    this.resultCache.clear();
    this.removeAllListeners();
  }

  /**
   * Sign a PDF from raw bytes using PEM credentials.
   *
   * Calls the native `signPdfBytes` FFI function (two-pass ByteRange writer).
   * Credentials are loaded and freed within this call.
   *
   * @param pdfData - Buffer containing the PDF document bytes
   * @param certPem - PEM-encoded certificate string
   * @param keyPem  - PEM-encoded private key string
   * @param reason  - Optional signature reason
   * @param location - Optional signature location
   * @returns Buffer containing the signed PDF
   */
  async signPdfData(
    pdfData: Buffer,
    certPem: string,
    keyPem: string,
    reason?: string,
    location?: string
  ): Promise<Buffer> {
    if (!this.native?.certificateLoadFromPem) {
      throw new SignatureException(
        'Native signing not available: certificateLoadFromPem not found'
      );
    }
    if (!this.native?.signPdfBytes) {
      throw new SignatureException('Native signing not available: signPdfBytes not found');
    }
    const certHandle = this.native.certificateLoadFromPem(certPem, keyPem);
    if (!certHandle) {
      throw new SignatureException('Failed to load PEM certificate');
    }
    try {
      const result = this.native.signPdfBytes(
        pdfData,
        certHandle,
        reason ?? null,
        location ?? null
      );
      if (!result) throw new SignatureException('signPdfBytes returned null');
      return Buffer.from(result);
    } finally {
      if (this.native?.pdf_certificate_free) {
        this.native.pdf_certificate_free(certHandle);
      }
    }
  }

  /**
   * Sign a PDF at a PAdES baseline level (ETSI EN 319 142-1) using PEM
   * credentials (#235). `level`: 0=B-B, 1=B-T, 2=B-LT, 3=B-LTA (frozen
   * mapping shared with every binding). For B-T/B-LT/B-LTA an
   * `options.tsaUrl` (RFC 3161 source) is required; `options.revocation`
   * supplies the B-LT DSS material.
   */
  async signPdfDataPades(
    pdfData: Buffer,
    certPem: string,
    keyPem: string,
    level: number,
    options?: {
      tsaUrl?: string;
      reason?: string;
      location?: string;
      revocation?: { certs?: Buffer[]; crls?: Buffer[]; ocsps?: Buffer[] };
    }
  ): Promise<Buffer> {
    if (!this.native?.certificateLoadFromPem) {
      throw new SignatureException(
        'Native signing not available: certificateLoadFromPem not found'
      );
    }
    if (!this.native?.signPdfBytesPades) {
      throw new SignatureException('Native signing not available: signPdfBytesPades not found');
    }
    const certHandle = this.native.certificateLoadFromPem(certPem, keyPem);
    if (!certHandle) {
      throw new SignatureException('Failed to load PEM certificate');
    }
    try {
      const result = this.native.signPdfBytesPades(
        pdfData,
        certHandle,
        level,
        options?.tsaUrl ?? null,
        options?.reason ?? null,
        options?.location ?? null,
        options?.revocation
          ? {
              certs: options.revocation.certs ?? [],
              crls: options.revocation.crls ?? [],
              ocsps: options.revocation.ocsps ?? [],
            }
          : undefined
      );
      if (!result) throw new SignatureException('signPdfBytesPades returned null');
      return Buffer.from(result);
    } finally {
      this.native?.certificateFree?.(certHandle);
    }
  }

  /**
   * Read the document's Document Security Store (`/DSS`,
   * ISO 32000-2:2020 §12.8.4.3) — PAdES-B-LT long-term-validation
   * material — or `null` when the PDF has no DSS (#235).
   */
  getDocumentSecurityStore(): {
    certs: Buffer[];
    crls: Buffer[];
    ocsps: Buffer[];
    vriCount: number;
  } | null {
    if (!this.native?.documentGetDss) {
      throw new SignatureException('Native DSS reader not available: documentGetDss not found');
    }
    return this.native.documentGetDss(this.document.handle) ?? null;
  }

  /**
   * Whether the document carries a document-scoped RFC 3161
   * `/DocTimeStamp` archival timestamp (PAdES-B-LTA,
   * ISO 32000-2:2020 §12.8.5) (#235).
   */
  hasDocumentTimestamp(): boolean {
    if (!this.native?.documentHasTimestamp) {
      throw new SignatureException(
        'Native B-LTA reader not available: documentHasTimestamp not found'
      );
    }
    return this.native.documentHasTimestamp(this.document.handle) === true;
  }

  // Private helpers
  private setCached(key: string, value: any): void {
    this.resultCache.set(key, value);
    if (this.resultCache.size > this.maxCacheSize) {
      const firstKey = this.resultCache.keys().next().value;
      if (firstKey !== undefined) this.resultCache.delete(firstKey);
    }
  }

  private clearCachePattern(prefix: string): void {
    const keysToDelete = Array.from(this.resultCache.keys()).filter((key) =>
      key.startsWith(prefix)
    );
    keysToDelete.forEach((key) => {
      this.resultCache.delete(key);
    });
  }
}

export default SignatureManager;

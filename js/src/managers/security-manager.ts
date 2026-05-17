/**
 * Manager for PDF document security and encryption
 *
 * Provides methods to check document encryption status, access permissions,
 * and security properties.
 *
 * @example
 * ```typescript
 * import { SecurityManager } from 'pdf_oxide';
 *
 * const doc = PdfDocument.open('document.pdf');
 * const securityManager = new SecurityManager(doc);
 *
 * if (securityManager.isEncrypted()) {
 *   console.log('Document is encrypted');
 *   console.log(`Can print: ${securityManager.canPrint()}`);
 *   console.log(`Can modify: ${securityManager.canModify()}`);
 * }
 * ```
 */

export interface PermissionsSummary {
  canPrint: boolean;
  canCopy: boolean;
  canModify: boolean;
  canAnnotate: boolean;
  canFillForms: boolean;
  isViewOnly: boolean;
  isEncrypted: boolean;
  requiresPassword: boolean;
  encryptionAlgorithm: string | null;
}

export interface SecurityLevel {
  level: 'high' | 'medium' | 'low' | 'none';
  description: string;
  isEncrypted: boolean;
  algorithm: string | null;
  restrictedAccess: boolean;
}

export interface AccessibilityValidation {
  canExtractText: boolean;
  canExtractImages: boolean;
  canAnalyzeContent: boolean;
  canSearch: boolean;
  canViewContent: boolean;
  isAccessible: boolean;
  issues: string[];
}

export class SecurityManager {
  private _document: any;

  /**
   * Creates a new SecurityManager for the given document
   * @param document - The PDF document
   * @throws Error if document is null or undefined
   */
  constructor(document: any) {
    if (!document) {
      throw new Error('Document is required');
    }
    this._document = document;
  }

  /**
   * Checks if the document is encrypted
   * @returns True if document is encrypted
   */
  isEncrypted(): boolean {
    try {
      return this._document.isEncrypted();
    } catch (error) {
      return false;
    }
  }

  /**
   * Checks if document requires a password
   * @returns True if password is required to open
   */
  requiresPassword(): boolean {
    return this.isEncrypted();
  }

  /**
   * Gets encryption algorithm used
   * @returns Encryption algorithm (e.g., 'RC4', 'AES-128', 'AES-256') or null
   */
  getEncryptionAlgorithm(): string | null {
    try {
      if (!this.isEncrypted()) return null;
      return this._document.getEncryptionAlgorithm() || null;
    } catch (error) {
      return null;
    }
  }

  /**
   * Checks if user can print the document
   * @returns True if printing is allowed
   */
  canPrint(): boolean {
    try {
      return this._document.canPrint();
    } catch (error) {
      return true;
    }
  }

  /**
   * Checks if user can copy text from document
   * @returns True if copying is allowed
   */
  canCopy(): boolean {
    try {
      return this._document.canCopy();
    } catch (error) {
      return true;
    }
  }

  /**
   * Checks if user can modify the document
   * @returns True if modification is allowed
   */
  canModify(): boolean {
    try {
      return this._document.canModify();
    } catch (error) {
      return true;
    }
  }

  /**
   * Checks if user can add or modify annotations
   * @returns True if annotation modification is allowed
   */
  canAnnotate(): boolean {
    try {
      return this._document.canAnnotate();
    } catch (error) {
      return true;
    }
  }

  /**
   * Checks if user can fill form fields
   * @returns True if form filling is allowed
   */
  canFillForms(): boolean {
    try {
      return this._document.canFillForms();
    } catch (error) {
      return true;
    }
  }

  /**
   * Checks if document is view-only
   * @returns True if no modifications are allowed
   */
  isViewOnly(): boolean {
    return !this.canModify() && !this.canCopy() && !this.canAnnotate();
  }

  /**
   * Gets all permissions as a summary object
   * @returns Permissions summary
   *
   * @example
   * ```typescript
   * const perms = manager.getPermissionsSummary();
   * console.log(JSON.stringify(perms, null, 2));
   * // {
   * //   canPrint: true,
   * //   canCopy: true,
   * //   canModify: false,
   * //   canAnnotate: true,
   * //   canFillForms: true,
   * //   isViewOnly: false,
   * //   isEncrypted: true
   * // }
   * ```
   */
  getPermissionsSummary(): PermissionsSummary {
    return {
      canPrint: this.canPrint(),
      canCopy: this.canCopy(),
      canModify: this.canModify(),
      canAnnotate: this.canAnnotate(),
      canFillForms: this.canFillForms(),
      isViewOnly: this.isViewOnly(),
      isEncrypted: this.isEncrypted(),
      requiresPassword: this.requiresPassword(),
      encryptionAlgorithm: this.getEncryptionAlgorithm(),
    };
  }

  /**
   * Gets security level information
   * @returns Security level details
   *
   * @example
   * ```typescript
   * const secLevel = manager.getSecurityLevel();
   * console.log(`Security level: ${secLevel.level}`); // 'high', 'medium', 'low', 'none'
   * ```
   */
  getSecurityLevel(): SecurityLevel {
    const isEncrypted = this.isEncrypted();
    const algorithm = this.getEncryptionAlgorithm();

    let level: 'high' | 'medium' | 'low' | 'none' = 'none';
    let description = 'No encryption';

    if (isEncrypted) {
      if (algorithm === 'AES-256') {
        level = 'high';
        description = 'AES-256 encryption';
      } else if (algorithm === 'AES-128') {
        level = 'medium';
        description = 'AES-128 encryption';
      } else if (algorithm === 'RC4') {
        level = 'low';
        description = 'RC4 encryption (deprecated)';
      } else {
        level = 'medium';
        description = 'Encrypted with unknown algorithm';
      }

      // Reduce level if permissions are very restrictive
      if (this.isViewOnly()) {
        description += ' (read-only)';
      }
    }

    return {
      level,
      description,
      isEncrypted,
      algorithm,
      restrictedAccess: !this.canModify() && !this.canCopy(),
    };
  }

  /**
   * Validates document accessibility based on security settings
   * @returns Accessibility validation result
   *
   * @example
   * ```typescript
   * const validation = manager.validateAccessibility();
   * if (!validation.canExtractText) {
   *   console.log('Warning: Document does not allow text extraction');
   * }
   * ```
   */
  validateAccessibility(): AccessibilityValidation {
    return {
      canExtractText: this.canCopy(),
      canExtractImages: this.canCopy(),
      canAnalyzeContent: this.canCopy(),
      canSearch: true, // Always allowed
      canViewContent: true, // Always allowed
      isAccessible: this.canCopy(),
      issues: this.getAccessibilityIssues(),
    };
  }

  /**
   * Gets list of accessibility issues
   * @returns Array of accessibility issues
   * @private
   */
  private getAccessibilityIssues(): string[] {
    const issues: string[] = [];

    if (this.isEncrypted()) {
      issues.push('Document is encrypted');
    }

    if (this.requiresPassword()) {
      issues.push('Document requires password');
    }

    if (!this.canCopy()) {
      issues.push('Text extraction is restricted');
    }

    if (!this.canModify()) {
      issues.push('Document modification is restricted');
    }

    if (!this.canPrint()) {
      issues.push('Printing is restricted');
    }

    if (!this.canAnnotate()) {
      issues.push('Annotation is restricted');
    }

    return issues;
  }

  /**
   * Generates security report
   * @returns Human-readable security report
   *
   * @example
   * ```typescript
   * console.log(manager.generateSecurityReport());
   * ```
   */
  generateSecurityReport(): string {
    const lines: string[] = [];
    const perms = this.getPermissionsSummary();
    const secLevel = this.getSecurityLevel();

    lines.push('=== PDF Security Report ===\n');

    lines.push(`Encryption Status: ${perms.isEncrypted ? 'Encrypted' : 'Not encrypted'}`);
    if (perms.isEncrypted) {
      lines.push(`  Algorithm: ${perms.encryptionAlgorithm || 'Unknown'}`);
      lines.push(`  Requires Password: ${perms.requiresPassword}`);
    }

    lines.push(`\nSecurity Level: ${secLevel.level.toUpperCase()}`);
    lines.push(`  Description: ${secLevel.description}`);

    lines.push('\nAccess Permissions:');
    lines.push(`  Print: ${perms.canPrint ? 'Allowed' : 'Restricted'}`);
    lines.push(`  Copy: ${perms.canCopy ? 'Allowed' : 'Restricted'}`);
    lines.push(`  Modify: ${perms.canModify ? 'Allowed' : 'Restricted'}`);
    lines.push(`  Annotate: ${perms.canAnnotate ? 'Allowed' : 'Restricted'}`);
    lines.push(`  Fill Forms: ${perms.canFillForms ? 'Allowed' : 'Restricted'}`);

    if (perms.isViewOnly) {
      lines.push('\n⚠️ WARNING: Document is VIEW-ONLY');
    }

    return lines.join('\n');
  }

  // ── Runtime crypto-governance policy (#230) — process-wide ──
  //
  // The policy is a process-global, set-once decorator over the crypto
  // provider, so these are static (no document handle required).

  private static loadNativeAddon(): any {
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      return require('../../index.node');
    } catch {
      return null;
    }
  }

  /**
   * Install the process-wide runtime crypto policy from its grammar
   * string (`"compat"|"strict"|"fips-strict"[;…]`). Fail-closed: throws
   * on an unparseable spec or if a policy is already set (#230).
   */
  static setCryptoPolicy(spec: string): void {
    const n = SecurityManager.loadNativeAddon();
    if (!n?.setCryptoPolicy) {
      throw new Error('Native crypto-governance not available: setCryptoPolicy not found');
    }
    n.setCryptoPolicy(spec);
  }

  /** The active crypto policy as its canonical grammar string (#230). */
  static getCryptoPolicy(): string {
    const n = SecurityManager.loadNativeAddon();
    return n?.cryptoPolicy ? (n.cryptoPolicy() as string) : 'compat';
  }

  /** Algorithm tokens exercised so far this process (governance) (#230). */
  static cryptoInventory(): string[] {
    const n = SecurityManager.loadNativeAddon();
    return n?.cryptoInventory ? (n.cryptoInventory() as string[]) : [];
  }

  /** CycloneDX 1.6 Cryptographic Bill of Materials (JSON) (#230). */
  static cryptoCbom(): string {
    const n = SecurityManager.loadNativeAddon();
    return n?.cryptoCbom ? (n.cryptoCbom() as string) : '';
  }
}

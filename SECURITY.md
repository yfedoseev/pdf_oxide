# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Currently supported versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :x:                |

## Reporting a Vulnerability

We take the security of pdf_oxide seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Where to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please email security reports to **yfedoseev@gmail.com**, or open a
private advisory through GitHub's
[Security Advisories](https://github.com/yfedoseev/pdf_oxide/security/advisories/new)
form.

### What to Include

Please include the following information in your report:

* Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
* Full paths of source file(s) related to the manifestation of the issue
* The location of the affected source code (tag/branch/commit or direct URL)
* Any special configuration required to reproduce the issue
* Step-by-step instructions to reproduce the issue
* Proof-of-concept or exploit code (if possible)
* Impact of the issue, including how an attacker might exploit it

### What to Expect

* We will acknowledge your email within 48 hours
* We will send a more detailed response within 7 days indicating the next steps
* We will keep you informed about progress towards a fix
* We may ask for additional information or guidance
* Once fixed, we will publicly disclose the vulnerability (crediting you if desired)

## PDF Security Considerations

PDF files can contain potentially malicious content. This library:

* **Validates input**: All PDF inputs are validated for structure and size limits
* **Limits recursion**: Maximum recursion depth prevents stack overflow
* **Resource limits**: Maximum file size, object count, and memory usage limits
* **Safe parsing**: No unsafe code in critical parsing paths
* **Sandboxing recommended**: For processing untrusted PDFs, run in a sandboxed environment

### Known Risks

When processing untrusted PDF files:

1. **Decompression bombs**: PDFs with highly compressed content can consume excessive memory
   - Mitigation: Size limits on decompressed streams

2. **Resource exhaustion**: Large or complex PDFs can consume significant CPU/memory
   - Mitigation: Configurable resource limits

3. **Malformed PDFs**: Crafted PDFs may trigger edge cases
   - Mitigation: Extensive validation and error handling

### Best Practices

When using pdf_oxide with untrusted PDFs:

1. **Set resource limits**: Configure `ParserLimits` appropriately
2. **Timeout operations**: Use timeouts for PDF processing
3. **Sandbox execution**: Run in containers or VMs when processing untrusted files
4. **Validate sources**: Only process PDFs from trusted sources when possible
5. **Monitor resources**: Track memory and CPU usage
6. **Update regularly**: Keep pdf_oxide updated with latest security patches

## Disclosure Policy

When we receive a security bug report, we will:

1. Confirm the problem and determine affected versions
2. Audit code to find similar problems
3. Prepare fixes for all supported versions
4. Release patches as soon as possible

We ask security researchers to:

* Give us reasonable time to respond before public disclosure
* Make a good faith effort to avoid privacy violations and service disruption
* Not access or modify other users' data

## Comments on this Policy

If you have suggestions on how this process could be improved, please submit a pull request.

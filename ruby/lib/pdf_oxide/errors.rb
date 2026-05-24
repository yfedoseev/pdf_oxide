# frozen_string_literal: true

module PdfOxide
  # Base error class for all PdfOxide exceptions.  Mirrors the Java
  # exception hierarchy at fyi.oxide.pdf.exception.* — every native
  # error maps to one of the subclasses below.
  class Error < StandardError; end

  # Raised when the host platform isn't supported by the bundled cdylib.
  class UnsupportedPlatformError < Error; end

  # Raised when a user-supplied argument fails validation BEFORE the
  # native call (nil check, range check, etc.).  Wrapper around
  # ::ArgumentError so it composes with Ruby's standard library.
  class ArgumentError < Error; end

  # Filesystem / I/O failures (file-not-found, EACCES, EIO, …).
  class IoError < Error; end

  # `IoError` specialisation for missing files.
  class FileNotFoundError < IoError; end

  # PDF parse / structure errors (malformed header, corrupt xref, …).
  class ParseError < Error; end

  # Resource / state errors — closed handle, wrong operation order.
  class StateError < Error; end

  # Operation called on an already-closed document/editor/Pdf.
  class InvalidStateError < StateError; end

  # Encryption / wrong-password failures.
  class EncryptedError < Error; end

  # Permission denied (encrypted PDF lacking extract / sign perm).
  class PermissionError < Error; end

  # Feature requested but not compiled into this cdylib build
  # (e.g. signatures without the `signatures` Cargo feature).
  class UnsupportedFeatureError < Error; end

  # Digital-signature failure (PAdES B/T/LT signing / verifying).
  class SignatureError < Error; end

  # Destructive-redaction failure (#231).  Security op: fails closed.
  class RedactionError < Error; end

  # PDF/A · PDF/X · PDF/UA compliance failure.
  class ComplianceError < Error; end

  # Native text-search operation failed (cdylib error code 7 /
  # `ERR_SEARCH`). Mirrors C#'s `PdfOxide.Exceptions.SearchException`
  # and Java's `PdfException(SEARCH)`.
  class SearchError < Error; end

  # Generic native-side failure that didn't map to a specific subclass.
  class InternalError < Error; end
end

# frozen_string_literal: true

module PdfOxide
  # Process-global crypto-governance policy (v0.3.50 #230).
  #
  # Mirrors `fyi.oxide.pdf.PdfPolicy`.  Selects which cryptographic
  # algorithms are accepted for reads and writes.  Composes with the
  # build-time feature flags (`legacy-crypto`, `fips`) — if a build
  # lacks `legacy-crypto`, COMPAT can't enable RC4/MD5-KDF regardless
  # of policy.
  #
  # **Set-once semantics.**  pdf_oxide installs the policy at most
  # once per process: call {.set} **before** any other pdf_oxide
  # operation.  A second `.set` call — or one after any document has
  # been opened — raises with a message containing "already set".
  module PdfPolicy
    # Policy modes (mirrors Java's `PolicyMode` enum).
    MODES = { compat: 0, strict: 1, fips_strict: 2 }.freeze
    ORDINAL_TO_MODE = MODES.invert.freeze

    module_function

    # @return [Symbol] the current process policy mode (:compat / :strict / :fips_strict).
    def current
      ord = Bindings.pdf_oxide_policy_current_ordinal if Bindings.respond_to?(:pdf_oxide_policy_current_ordinal)
      ord ||= 0 # default COMPAT if accessor not exposed in this build
      ORDINAL_TO_MODE.fetch(ord, :compat)
    rescue ::FFI::NotFoundError
      :compat
    end

    # Set the process-global policy mode.  Call before any other
    # pdf_oxide operation.
    # @param mode [Symbol]
    # @raise [InternalError] policy was already set.
    def set(mode)
      ordinal = MODES.fetch(mode) do
        raise ::PdfOxide::ArgumentError, "mode must be one of #{MODES.keys.inspect}, got #{mode.inspect}"
      end
      raise UnsupportedFeatureError, 'policy not supported by this cdylib build' \
        unless Bindings.respond_to?(:pdf_oxide_policy_set_by_ordinal)

      rc = Bindings.pdf_oxide_policy_set_by_ordinal(ordinal)
      raise InternalError, 'policy already set' if rc != 0

      mode
    end

    # @return [Symbol] :compat preset (accept all algorithms).
    def compat
      :compat
    end

    # @return [Symbol] :strict preset (reject legacy algorithms).
    def strict
      :strict
    end

    # @return [Symbol] :fips_strict preset (FIPS 140-3 only).
    def fips_strict
      :fips_strict
    end
  end
end

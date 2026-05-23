# frozen_string_literal: true

# Phase 3 acceptance test for PadesSigner (v0.3.50 #235 + v0.3.51
# 5-arg shim follow-up).
#
# Verifies the API surface and security-op contract.  A full
# end-to-end signing test requires PKCS#12 / PEM credentials and a
# live TSA, neither of which is available in this CI environment;
# those land in Phase 4's empirical-smoke run.  This spec asserts
# the wire interface is plumbed correctly: struct layout, level
# enum codes, fail-closed input validation.

require 'spec_helper'

RSpec.describe PdfOxide::PadesSigner, :skip_mock do
  it 'declares the canonical PAdES level enum codes' do
    expect(PdfOxide::PadesSigner::LEVELS).to eq(b: 0, t: 1, lt: 2, lta: 3)
  end

  it 'PadesSignOptions struct matches PadesSignOptionsC layout (size 112 bytes on x86_64)' do
    # Field count + types come from the C header — drift here means
    # the Rust `#[repr(C)]` struct changed and binding callers will
    # corrupt memory.  Sentinel guard for the layout.
    expected_size_x86_64 = 14 * 8 # 14 fields × 8 bytes (pointers + size_t + int32 padded)
    expect(PdfOxide::PadesSigner::PadesSignOptions.size).to eq(expected_size_x86_64)
  end

  it 'fails closed on a null certificate handle (security op)' do
    expect {
      PdfOxide::PadesSigner.sign_pades(
        pdf_bytes:          '%PDF-1.7',
        certificate_handle: nil,
        level:              :b
      )
    }.to raise_error(PdfOxide::ArgumentError, /certificate_handle/)
  end

  it 'rejects unknown PAdES levels' do
    fake_cert = ::FFI::Pointer.new(0xdeadbeef)
    expect {
      PdfOxide::PadesSigner.sign_pades(
        pdf_bytes:          '%PDF-1.7',
        certificate_handle: fake_cert,
        level:              :forged
      )
    }.to raise_error(PdfOxide::ArgumentError, /level must be one of/)
  end

  it 'rejects empty pdf_bytes' do
    fake_cert = ::FFI::Pointer.new(0xdeadbeef)
    expect {
      PdfOxide::PadesSigner.sign_pades(
        pdf_bytes:          '',
        certificate_handle: fake_cert,
        level:              :b
      )
    }.to raise_error(PdfOxide::ArgumentError, /pdf_bytes/)
  end
end

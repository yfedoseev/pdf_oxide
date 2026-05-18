//go:build !cgo

package pdfoxide

import "testing"

// Regression: the purego backend registers every FFI symbol in a single
// sync.Once on the first call. pdf_sign_bytes_pades has 18 scalar
// parameters, which exceeds purego's SysV/AMD64 argument limit —
// purego.RegisterLibFunc panicked ("too many stack arguments") so the
// ENTIRE purego backend was unusable at runtime (any first call paniced).
// CI never caught it because purego integration tests skip when
// PDF_OXIDE_LIB_PATH is unset (build-only coverage).
//
// The fix routes purego through the 5-arg pdf_sign_bytes_pades_opts
// struct-options shim. This test forces the registration path and fails
// loudly on any panic.
func TestPuregoFFIRegistrationDoesNotPanic(t *testing.T) {
	requireLib(t)
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("purego FFI registration panicked — regression of the "+
				"pdf_sign_bytes_pades 18-arg / purego ABI-limit bug: %v", r)
		}
	}()
	// Any FFI call triggers loadLib → registerFFI (the sync.Once that
	// registered the offending symbol). FromMarkdown is cgo-free.
	if _, err := FromMarkdown("# regression\n\npurego registration smoke."); err != nil {
		t.Fatalf("FromMarkdown after FFI registration: %v", err)
	}
}

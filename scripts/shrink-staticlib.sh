#!/usr/bin/env bash
# Shrink a Rust-produced staticlib (.a / .lib) by removing sections that are
# useless to non-Rust downstream consumers.
#
# Rust staticlibs embed per-object `.llvmbc` + `.llvmcmd` (LLVM bitcode for
# cross-crate LTO) and DWARF `.debug_*` sections. Neither is used by:
#   - CGo's linker (Go staticlib consumer)
#   - node-gyp / gyp-ng (Node.js addon staticlib consumer)
#   - MSVC's LINK.EXE in default mode (C# NuGet path uses cdylib anyway)
#
# Empirically on this repo (v0.3.55):
#   - linux-x86_64: 35 MB .llvmbc + 4 MB DWARF (correctly shrunk)
#   - linux-aarch64: 109 MB .llvmbc UNSTRIPPED before the cross-arch fix
#     because the host's x86_64 `objcopy` silently no-op'd on ARM64
#     objects (exited 0 but produced byte-identical output)
#   - macOS x86_64/arm64: ~80-100 MB Mach-O `__LLVM,__bitcode` UNSTRIPPED
#     before the bitcode_strip fix (the prior `strip -S` only touched
#     DWARF; the intermediate llvm-objcopy attempt never ran because
#     Xcode does not ship `llvm-objcopy` under any `xcrun`-resolvable
#     name and macos-latest has no `llvm-objcopy` on PATH)
#
# Usage: shrink-staticlib.sh <path-to-staticlib> [<target-triple>]
#
# The second argument is the Rust target triple (e.g. `aarch64-unknown-linux-gnu`,
# `x86_64-apple-darwin`). It is consulted on Linux to pick the right
# arch-specific `objcopy` for cross-compiled archives — when omitted the
# script falls back to the host's `objcopy`, which is correct for native
# builds but a silent no-op for cross-compiled ones.

set -euo pipefail

LIB="${1:?path to .a / .lib required}"
TARGET="${2:-}"

if [[ ! -f "$LIB" ]]; then
  echo "shrink-staticlib: file not found: $LIB" >&2
  exit 1
fi

before=$(wc -c < "$LIB")

# Pick the right `objcopy` for the archive's target. Generic Ubuntu `objcopy`
# is built with BFD support for many targets but in practice silently no-ops
# on objects whose ELF machine type the host binutils wasn't compiled to
# touch — exit 0 + zero bytes saved. Using the cross-compile toolchain's
# arch-specific objcopy (already installed for the cross-compile build
# itself) avoids the silent-failure path entirely.
pick_objcopy_linux() {
  case "$TARGET" in
    aarch64-unknown-linux-*)
      if command -v aarch64-linux-gnu-objcopy >/dev/null 2>&1; then
        echo "aarch64-linux-gnu-objcopy"
        return
      fi
      ;;
    x86_64-pc-windows-gnu)
      if command -v x86_64-w64-mingw32-objcopy >/dev/null 2>&1; then
        echo "x86_64-w64-mingw32-objcopy"
        return
      fi
      ;;
  esac
  # Native target (or unknown — defer to host objcopy).
  echo "objcopy"
}

case "$(uname -s)" in
  Linux|MINGW*|MSYS*|CYGWIN*)
    OBJCOPY=$(pick_objcopy_linux)
    if ! command -v "$OBJCOPY" >/dev/null 2>&1; then
      echo "shrink-staticlib: $OBJCOPY not available; skipping $LIB" >&2
      exit 0
    fi
    echo "shrink-staticlib: using $OBJCOPY (target=${TARGET:-host})"
    # Split-debug `.dwo` archive members (emitted by the mingw cross-compile
    # toolchain on x86_64-pc-windows-gnu) contain *only* DWARF sections.
    # `objcopy --strip-debug` removes their only sections and then aborts
    # the whole archive with "has no sections". Drop these members first so
    # objcopy has nothing debug-only left to process.
    if command -v ar >/dev/null 2>&1; then
      mapfile -t dwo_members < <(ar t "$LIB" 2>/dev/null | grep -E '\.dwo$' || true)
      if [[ ${#dwo_members[@]} -gt 0 ]]; then
        for m in "${dwo_members[@]}"; do
          ar d "$LIB" "$m" || true
        done
      fi
    fi
    # llvm-objcopy rejects "same input and output" on some distros; write to
    # a sibling tmp file and move it into place atomically.
    tmp="${LIB}.shrink.tmp"
    "$OBJCOPY" \
      --remove-section=.llvmbc \
      --remove-section=.llvmcmd \
      --strip-debug \
      "$LIB" "$tmp"
    mv "$tmp" "$LIB"
    ;;
  Darwin)
    # macOS staticlibs from Rust w/ `lto = true` carry per-object
    # `__LLVM,__bitcode` (+ `__cmdline`) segments — multi-MB each, unused by
    # CGo, node-gyp, or NuGet.
    #
    # We deliberately do NOT use Apple's `bitcode_strip -r`: for MH_OBJECT (.o)
    # inputs it does not strip the segment itself, it shells out to
    #   ld -keep_private_externs -r -bitcode_process_mode strip <in> -o <out>
    # (cctools/misc/bitcode_strip.c). Apple's default linker since Xcode 15
    # (`ld-prime`) dropped `-bitcode_process_mode`, so `ld` misreads the mode
    # token `strip` as an input file and dies with
    #   "ld: file cannot be open()ed, errno=2 path=strip"
    # → "bitcode_strip: internal link edit command failed". No invocation tweak
    # fixes this; the failure is inside ld. (dotnet/macios#22806, #22591.)
    #
    # Instead use `llvm-objcopy` from the Rust toolchain's `llvm-tools`
    # component — the same LLVM that produced these objects, with native Mach-O
    # `SEGNAME,SECTNAME` section removal. (This is the approach the tweag
    # "shrinking static libs" guide lands on for macOS.) llvm-objcopy operates
    # per-Mach-O, not on archives, so explode the .a, strip each member, then
    # reassemble via libtool.
    llvm_bin="$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/^host: //p')/bin"
    OBJCOPY="$llvm_bin/llvm-objcopy"
    if [[ ! -x "$OBJCOPY" ]]; then
      rustup component add llvm-tools-preview >/dev/null 2>&1 \
        || rustup component add llvm-tools >/dev/null 2>&1 || true
    fi
    if [[ ! -x "$OBJCOPY" ]]; then
      echo "shrink-staticlib: llvm-objcopy not available ($OBJCOPY); running strip -S only (bitcode will survive)" >&2
      strip -S "$LIB"
    else
      echo "shrink-staticlib: using $OBJCOPY (Mach-O __LLVM,__bitcode strip)"
      abs_lib=$(cd "$(dirname "$LIB")" && pwd)/$(basename "$LIB")
      workdir=$(mktemp -d)
      (
        cd "$workdir"
        ar x "$abs_lib"
        for obj in *.o; do
          [[ -f "$obj" ]] || continue
          "$OBJCOPY" \
            --remove-section=__LLVM,__bitcode \
            --remove-section=__LLVM,__cmdline \
            --strip-debug \
            "$obj" "$obj.stripped"
          mv "$obj.stripped" "$obj"
        done
        xcrun libtool -static -o "$abs_lib" *.o
      )
      rm -rf "$workdir"
      strip -S "$LIB"
    fi
    ;;
  *)
    echo "shrink-staticlib: unknown OS $(uname -s); skipping" >&2
    ;;
esac

after=$(wc -c < "$LIB")
saved=$((before - after))
pct=$(awk -v b="$before" -v a="$after" 'BEGIN{ if(b>0) printf "%.1f", (b-a)*100/b; else print "0.0" }')
echo "shrink-staticlib: $LIB  ${before} -> ${after} bytes  (saved ${saved}, ${pct}%)"

# Defensive: refuse to ship a "shrunk" archive that's still gigantic.
# A correctly-stripped pdf_oxide staticlib (default-features + ocr) lands at
# ~50-90 MB depending on target. Anything >= 130 MB almost certainly means
# bitcode survived — fail loudly so CI catches it instead of silently
# uploading another bloated artifact.
MAX_BYTES=$((130 * 1024 * 1024))
if [[ "$after" -gt "$MAX_BYTES" ]]; then
  echo "::error::shrink-staticlib: $LIB is $after bytes after stripping (> 130 MB cap). Bitcode / debug sections likely survived. Target=${TARGET:-host}" >&2
  exit 1
fi

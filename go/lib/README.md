# Native Libraries — installed on demand

Starting with **v0.3.31**, native `pdf_oxide` libraries are no longer
committed to this module. Instead they're downloaded from GitHub Releases
on demand by a small Go installer, which removes ~310 MB of per-release
repository bloat (Rust staticlibs for 6 platforms).

## Install (one-time per machine)

```bash
# Always resolves to the matching FFI assets for the installer's own version.
go run github.com/yfedoseev/pdf_oxide/go/cmd/install@latest
# Or pin to a specific version:
go run github.com/yfedoseev/pdf_oxide/go/cmd/install@v0.3.36
```

The installer detects `GOOS`/`GOARCH`, downloads the matching asset from
`https://github.com/yfedoseev/pdf_oxide/releases/download/v0.3.36/…`, and
extracts `libpdf_oxide.a` + `pdf_oxide.h` into `~/.pdf_oxide/v0.3.36/`.

It then prints the `CGO_CFLAGS` / `CGO_LDFLAGS` you need to export:

```
export CGO_CFLAGS="-I$HOME/.pdf_oxide/v0.3.36/include"
export CGO_LDFLAGS="$HOME/.pdf_oxide/v0.3.36/lib/linux_amd64/libpdf_oxide.a -lm -lpthread -ldl -lrt -lgcc_s -lutil -lc"
```

After that, `go build` / `go test` work normally.

## Alternative: `go generate`

If you prefer to wire installation into your own project's build, add this
to any `.go` file in your project:

```go
//go:generate go run github.com/yfedoseev/pdf_oxide/go/cmd/install@latest --write-flags=.
```

Running `go generate ./...` then drops a `cgo_flags.go` next to your
`//go:generate` directive with the right `#cgo LDFLAGS` baked in for your
machine's install path. That file is per-machine — add it to `.gitignore`.

## Development / monorepo builds

If you're working inside the `pdf_oxide` monorepo and have already run
`cargo build --release --lib`, build the Go module with the `pdf_oxide_dev`
tag to use the workspace `target/` path directly:

```bash
cd go && go build -tags pdf_oxide_dev ./...
```

No installer needed in that mode.

## Windows ARM64

Windows ARM64 currently ships a dynamic `pdf_oxide.dll` (not a staticlib)
because Rust's `aarch64-pc-windows-gnullvm` target is still Tier 3. Go
binaries for this platform must ship `pdf_oxide.dll` alongside the
executable at runtime.

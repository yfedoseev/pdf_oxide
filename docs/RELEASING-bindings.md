# Releasing the language bindings

This guide covers publishing the **11 native-language bindings** introduced in
v0.3.69 — C++, Swift, Objective-C, Kotlin, Scala, Clojure, Dart, Elixir, R,
Julia, and Zig — to their respective package registries. All eleven are thin
layers over the same **pdf_oxide C ABI** (the JVM trio — Kotlin/Scala/Clojure —
go through the Java JNI binding).

The canonical workspace version is **0.3.69** (kept in lock-step across every
manifest by `scripts/sync_version.py`).

For the **established** bindings (Python→PyPI, Node→npm, Java→Maven Central,
.NET→NuGet, Ruby→RubyGems, Go→module tags, PHP→Packagist) publishing is already
wired into `.github/workflows/release.yml` and is not repeated here.

> **Status note (v0.3.69).** `release.yml` now contains jobs for six of the new
> bindings — `publish-kotlin`, `publish-scala`, `publish-clojure`,
> `publish-elixir`, `publish-dart` (publishing), and `package-cocoapods`
> (uploads a Trunk-free release asset — no central publish). They are hard-gated
> exactly like the established jobs (never on a PR dry-run; only a real tag push
> or `workflow_dispatch` with `publish=true`; never for a prerelease version).
> The five publishing jobs require the GitHub Actions secret(s) in the "Required
> secret(s)" column to be provisioned **before** the next tagged release, or the
> job will fail. CocoaPods needs **no secret** (it does not use Trunk).
> The remaining bindings — **C++** (vcpkg/Conan), **R** (CRAN), **Julia**
> (General registry), and **Swift/Zig** (git tag only) — cannot be driven from
> `release.yml` and stay **manual**; follow their step-by-steps below.
>
> Not yet verified on real infrastructure (no secrets/Apple toolchain in dev):
> the Maven Central, Hex, pub.dev, Clojars, and CocoaPods jobs run end-to-end
> for the first time on the next tag — watch that run. The CocoaPods job builds
> a macOS `xcframework` and uploads it + the podspec as release assets (no
> Trunk push); consumers install via the `:podspec =>` URL form.

## Registry / automation matrix

| Binding | Registry | Coordinates | Automated? (target) | Required secret(s) |
|---|---|---|---|---|
| C++ | vcpkg + Conan | port `pdf-oxide` / `pdf_oxide_cpp` | manual (PR + local upload) | none (vcpkg PR is human; Conan remote creds local) |
| Swift | SwiftPM (git tag) | `github.com/yfedoseev/pdf_oxide` | manual (just push the `v0.3.69` tag) | none (no registry upload) |
| Objective-C | CocoaPods (Trunk-free, release-asset podspec) | pod `PdfOxide` via `:podspec =>` URL | automatable via release.yml | none (no Trunk; Trunk read-only 2026-12-02) |
| Kotlin | Maven Central | `fyi.oxide:pdf-oxide-kotlin` | automatable via release.yml | `MAVEN_CENTRAL_USERNAME`, `MAVEN_CENTRAL_PASSWORD`, `MAVEN_GPG_PRIVATE_KEY`, `MAVEN_GPG_PASSPHRASE` |
| Scala | Maven Central | `fyi.oxide:pdf-oxide-scala_3` | automatable via release.yml | `MAVEN_CENTRAL_USERNAME`, `MAVEN_CENTRAL_PASSWORD`, `MAVEN_GPG_PRIVATE_KEY`, `MAVEN_GPG_PASSPHRASE` |
| Clojure | Clojars | `fyi.oxide/pdf-oxide-clojure` | automatable via release.yml | `CLOJARS_USERNAME`, `CLOJARS_PASSWORD` |
| Dart | pub.dev | `pdf_oxide` | automatable via release.yml | none — uses GitHub Actions **OIDC** ("Automated publishing") |
| Elixir | Hex.pm | `pdf_oxide` | automatable via release.yml | `HEX_API_KEY` |
| R | CRAN | `pdfoxide` | **manual only** (human email confirmation) | none |
| Julia | General registry | `PdfOxide` | **manual only** (Registrator comment) | none (GitHub App) |
| Zig | none (git tag + tarball hash) | `pdf_oxide` | manual (push the tag) | none (no registry) |

The Maven Central + GPG secrets are **the same four** the Java binding already
uses (`publish-maven` job in `release.yml`); Kotlin and Scala reuse them
verbatim. Clojars needs its own pair.

## Automated targets — secret provisioning

A maintainer adds these under **repo Settings → Secrets and variables →
Actions** before enabling the corresponding publish job.

### Maven Central (Kotlin, Scala) — reused from Java

- `MAVEN_CENTRAL_USERNAME` — Sonatype Central Portal token name.
- `MAVEN_CENTRAL_PASSWORD` — Sonatype Central Portal token secret.
- `MAVEN_GPG_PRIVATE_KEY` — ASCII-armored private key (GPG key **DC1DB87A**).
- `MAVEN_GPG_PASSPHRASE` — passphrase for that key.

Kotlin publishes via the `com.vanniktech.maven.publish` Gradle plugin (reads
`ORG_GRADLE_PROJECT_mavenCentralUsername` / `*Password` and the signing key from
the env). Scala publishes via `sbt-ci-release` + `sbt-sonatype` targeting
`central.sonatype.com` (reads `SONATYPE_USERNAME` / `SONATYPE_PASSWORD` and
`PGP_SECRET` / `PGP_PASSPHRASE`). Map the four canonical secrets above into
whichever env var names each tool expects in the job step. Both publish to the
Central Portal with the publish gate left **manual** (matching the Java binding:
the maintainer flips "Publish" in the Portal UI per the release gate).

### Clojars (Clojure)

- `CLOJARS_USERNAME` — Clojars username (or deploy-token name).
- `CLOJARS_PASSWORD` — Clojars deploy token.

`clojure -T:build deploy` (via `deps-deploy`) reads these two env vars directly.

### Hex.pm (Elixir)

- `HEX_API_KEY` — Hex API key with publish permission.

`mix hex.publish --yes` reads `HEX_API_KEY` from the env (`mix hex.config
api_key` is the local equivalent).

### CocoaPods (Objective-C) — Trunk-FREE, no secret

CocoaPods [Trunk goes read-only on 2026-12-02](https://blog.cocoapods.org/CocoaPods-Specs-Repo/)
and new Trunk pushes/accounts are no longer viable. We therefore do **not** use
Trunk and need **no `COCOAPODS_TRUNK_TOKEN`**. Instead the `package-cocoapods`
job assembles `PdfOxide.xcframework` on a macOS runner, zips it with the objc/
sources, and uploads BOTH that zip and `PdfOxide.podspec` as GitHub release
assets. Consumers install via a direct podspec URL (no central index):

```ruby
pod 'PdfOxide', :podspec =>
  'https://github.com/yfedoseev/pdf_oxide/releases/download/v0.3.69/PdfOxide.podspec'
```

The job validates with `pod spec lint` (no account needed). See
`objc/PUBLISHING.md` for the full xcframework assembly sequence.

### pub.dev (Dart) — OIDC, no long-lived secret

pub.dev supports **automated publishing from GitHub Actions via OIDC**, so there
is *no* API token to store. One-time setup by a maintainer:

1. Publish the first `0.3.69` version manually once (`dart pub publish`) to claim
   the `pdf_oxide` package and become an uploader.
2. On <https://pub.dev/packages/pdf_oxide/admin>, enable **Automated
   publishing → Publishing from GitHub Actions**, set:
   - Repository: `yfedoseev/pdf_oxide`
   - Tag pattern: `v{{version}}` (matches the `v0.3.69` release tag).
3. A workflow job then uses the
   `dart-lang/setup-dart` action's OIDC support and runs `dart pub publish
   --force`; pub.dev validates the OIDC token against the configured repo + tag.

## Manual bindings — step-by-step

### R → CRAN (`pdfoxide`)

CRAN submission ends in a human email confirmation and **cannot be fully
automated**.

```bash
# from repo root; build the source tarball
R CMD build r/
# strict CRAN checks on the produced tarball
R CMD check --as-cran pdfoxide_0.3.69.tar.gz
```

Resolve all NOTES/WARNINGs, then submit `pdfoxide_0.3.69.tar.gz` via the web
form at <https://cran.r-project.org/submit.html>. CRAN emails the maintainer
(`yfedoseev@gmail.com`) a confirmation link that must be clicked, followed by
the reviewer's accept/reject. Once accepted, end users install with
`install.packages("pdfoxide")`.

### Julia → General registry (`PdfOxide`)

Requires the **JuliaRegistrator** GitHub App installed on the repo.

1. Push the release tag `v0.3.69`.
2. On the **release commit** (the GitHub commit page, in a comment), post:

   ```
   @JuliaRegistrator register subdir=julia
   ```

   `subdir=julia` is required because the package lives in the `julia/`
   subdirectory rather than the repo root.
3. Registrator opens an automated PR against the General registry; after the
   3-day auto-merge cooldown it lands and `Pkg.add("PdfOxide")` works.
4. Install **TagBot** (`JuliaRegistries/TagBot` action) so subsequent registered
   versions get their git tags + GitHub releases created automatically.

### C++ → vcpkg + Conan

**vcpkg** (consumer registry — open a PR to `microsoft/vcpkg`):

1. Add a port under `ports/pdf-oxide/` with a `vcpkg.json` (version `0.3.69`)
   and a `portfile.cmake` that fetches the release tarball via
   `vcpkg_from_github` using the release **tag** and its **SHA512**.
2. Compute the SHA512 vcpkg expects:

   ```bash
   vcpkg_url=https://github.com/yfedoseev/pdf_oxide/archive/refs/tags/v0.3.69.tar.gz
   curl -sL "$vcpkg_url" | sha512sum
   ```
3. Run `vcpkg x-add-version pdf-oxide` to update the version DB, then open the PR
   to `microsoft/vcpkg`. Merge is a human review by the vcpkg maintainers.

**Conan** (publish to a Conan remote):

> Note: `cpp/` does **not** currently contain a `conanfile.py`; one must be added
> (a `ConanFile` recipe declaring the header-only `pdf_oxide_cpp` target + the
> native `libpdf_oxide` dependency) before the steps below can run. The task
> brief assumed a `conanfile.py` already exists in `cpp/` — it does not.

```bash
cd cpp
conan create . --version 0.3.69            # build + package the recipe locally
conan remote add oxide <REMOTE_URL>        # one-time: point at your Conan remote
conan upload "pdf_oxide_cpp/0.3.69" -r oxide --confirm
```

The C++ wrapper is header-only and links the prebuilt `libpdf_oxide` cdylib;
both the vcpkg port and the Conan recipe must declare that native dependency (or
build it from the Rust source).

### Zig → git tag + tarball hash (no registry)

Zig has no central package registry. Consumers depend on a release **tarball
URL + content hash** recorded in their own `build.zig.zon`. Releasing is just
pushing the `v0.3.69` tag; consumers then run:

```bash
# in the consuming project — adds the dependency with the correct hash
zig fetch --save https://github.com/yfedoseev/pdf_oxide/archive/refs/tags/v0.3.69.tar.gz
```

`zig fetch --save` downloads the tarball, computes the hash, and writes the
`.dependencies.pdf_oxide = .{ .url = …, .hash = … }` entry into the consumer's
`build.zig.zon`. The binding is pinned to **Zig 0.15.x** (the README pins
0.15.1; `build.zig.zon` declares `minimum_zig_version = "0.15.0"`).

### Swift → SwiftPM (git tag, no registry)

Swift's idiomatic channel is SwiftPM by git tag — no upload. Pushing the
`v0.3.69` tag is the entire "release"; consumers add
`.package(url: "https://github.com/yfedoseev/pdf_oxide", from: "0.3.69")`. See
`swift/README.md` for the native-library env-var setup consumers still need.

## Release order checklist

1. **Version sync** — run `scripts/sync_version.py`; confirm every binding
   manifest reads `0.3.69` (Cargo.toml, `build.gradle.kts`, `build.sbt`,
   `build.clj`, `pubspec.yaml`, `mix.exs`, `DESCRIPTION`, `Project.toml`,
   `build.zig.zon`, `PdfOxide.podspec`, `CMakeLists.txt`).
2. **Green CI** — the established `release.yml` pipeline plus the per-binding
   workflows (`kotlin.yml`, `scala.yml`, … `zig.yml`) all pass.
3. **Tag** — merge the release branch and push `v0.3.69` (this alone releases
   Swift and Zig).
4. **JVM trio** — publish the Java binding first (Kotlin/Scala/Clojure depend on
   `fyi.oxide:pdf-oxide`), then Maven Central for Kotlin + Scala, then Clojars
   for Clojure.
5. **OIDC / token registries** — Dart (pub.dev OIDC), Elixir (`mix hex.publish`).
   Objective-C is **Trunk-free**: the `package-cocoapods` job assembles the
   xcframework and uploads it + `PdfOxide.podspec` as release assets — nothing to
   push. Consumers install via the `:podspec =>` URL (see objc/README.md).
6. **C++** — `conan upload` to the remote; open the `microsoft/vcpkg` port PR.
7. **Human-gated registries** — submit R to CRAN
   (<https://cran.r-project.org/submit.html>, confirm the email) and comment
   `@JuliaRegistrator register subdir=julia` on the Julia release commit.
8. **Verify** — once indexes propagate, smoke-test each install snippet from the
   binding READMEs against `0.3.69`.

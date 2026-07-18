# Contributing to pdf_oxide

Thank you for your interest in contributing. pdf_oxide is a correctness-critical
PDF library with 20+ language bindings built on one Rust core — a change in the
core reaches every binding, and a subtle extraction regression can silently
corrupt output for thousands of real-world documents. The rules below exist so
that **good-faith work gets merged quickly and low-effort submissions don't
drown it out.**

These rules apply to **everyone, including the maintainer.** A contribution has
to be worth more to the project than the time it takes to review it. We are not
anti-newcomer and not anti-AI — we are anti-slop.

## Table of Contents

- [The two rules that matter most](#the-two-rules-that-matter-most)
- [Code of Conduct](#code-of-conduct)
- [Before you open a pull request](#before-you-open-a-pull-request)
- [Development setup](#development-setup)
- [Testing and regression requirements](#testing-and-regression-requirements)
- [Test fixture policy](#test-fixture-policy)
- [AI-assisted contributions](#ai-assisted-contributions)
- [Coding standards](#coding-standards)
- [Continuous integration gates](#continuous-integration-gates)
- [Commits, DCO, and review](#commits-dco-and-review)
- [Security reports](#security-reports)
- [License](#license)

---

## The two rules that matter most

Almost every rejected or stalled PR in this project's history failed one of
these two rules. Read them first:

1. **Every non-trivial PR must reference an issue the maintainer has accepted.**
   Open (or find) an issue, agree on the *approach* there, and only then write
   code. **Drive-by pull requests that do not reference an accepted issue may be
   closed without detailed review.** This protects you as much as us: it stops
   you sinking hours into a change we can't take, or a design we'd build
   differently.

2. **Any change to extraction, layout, rendering, or font handling must be
   proven not to regress on a corpus of real PDFs.** These paths are heuristic:
   a change that looks obviously correct routinely deletes headings, fuses
   words, drops spans, or reverses scripts on documents you didn't test. A
   passing unit test is *not* evidence of no regression. See
   [Testing and regression requirements](#testing-and-regression-requirements).

Bug fixes, typo fixes, and documentation corrections can skip the "accepted
issue" step (rule 1) — but never rule 2.

## Code of Conduct

This project adheres to the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating you agree to uphold it. Report unacceptable behavior by opening an
issue or contacting the maintainer.

## Before you open a pull request

**1. Start from an accepted issue.**
- Search [existing issues](https://github.com/yfedoseev/pdf_oxide/issues) and
  open PRs first — duplicate and fragmented PRs for one problem waste review
  time.
- For a **new feature or a behavior change**, open an issue *before* writing
  code and wait for a maintainer to agree it's in scope and agree on the
  approach. Large features built without that agreement are frequently declined
  no matter how good the code is — the design, not the effort, is the sticking
  point.
- For a **bug fix**, an issue is recommended but not required; the reproducer
  (below) is what matters.
- Reference the issue in the **PR description** with `Closes #NNN` (not in the
  commit message — it survives rebases better there).

**2. One logical change per PR.** Do not bundle a bug fix with a refactor, or
correctness with performance. Grab-bag PRs are unreviewable and get split.
Complete the change within the PR — no "finish it later" TODOs for new features.

**3. Write the PR in your own words.** The description must explain *what
problem you're solving and why this approach*. If you can't describe the change
in your own words, it isn't ready.

**4. Run the full local gate before pushing** (see
[CI gates](#continuous-integration-gates)) — a green subset is necessary, not
sufficient.

## Development setup

### Prerequisites
- **Rust**: the pinned MSRV is in `Cargo.toml` (`rust-version`, currently
  **1.88**). CI builds the library at exactly this floor — don't use newer
  language features without raising it deliberately. ([Install Rust](https://rustup.rs/))
- **Python**: **3.9+** for the Python bindings (3.8 is EOL).
- **C compiler**: gcc or clang, for native dependencies.

### Build and test
```bash
git clone https://github.com/YOUR_USERNAME/pdf_oxide.git
cd pdf_oxide
cargo build
cargo test            # default features (icc, legacy-crypto)
```

Install the git hooks so formatting/lint/tests run on commit:
```bash
./scripts/setup-hooks.sh
```

> **Never use `cargo test --all-features`** — the `fips` feature and the
> default-on `legacy-crypto` feature are mutually exclusive (FIPS 140-3 forbids
> the MD5 `legacy-crypto` pulls in) and enforced with `compile_error!`. This
> applies to any command that compiles the crate. Verify FIPS-gated code
> separately:
> ```bash
> cargo test --no-default-features --features fips,icc
> ```

## Testing and regression requirements

We adopt the industry-standard bar for correctness-critical parsers: **we do not
merge code that isn't tested** (à la qpdf), and **a bug-fix test must fail
before your change and pass after** (à la pdfplumber/pypdf). Concretely:

### 1. Every bug fix ships a regression test that fails without the fix
Add the test in a commit **before** the fix commit (or otherwise be ready to
show it goes red when the fix is reverted). Reviewers verify this. A test that
passes even when the feature is a no-op is worse than no test — it gives false
confidence.

### 2. Build the reproducer as a minimal *synthetic* PDF, in code
Construct the smallest PDF that triggers the defect as bytes inside the test —
this is the pervasive pattern across `tests/` (e.g. building `%PDF-1.x` byte
strings, or via the writer API). **Do not commit a real-world, reporter-supplied,
or third-party PDF.** If a reporter attached a document, treat it as a
*specification*: reproduce its relevant structure synthetically. Reduce to the
fewest objects/operators that still reproduce the bug.

### 3. Prove no corpus regression — on *your own* corpus
**The project's regression corpus is private and is not distributed.** You are
expected to assemble your **own** set of representative real-world PDFs (scanned,
tagged, CJK/RTL, forms, multi-column, math, rotated — whatever your change could
affect) and prove your change doesn't regress them. The tooling is provided; the
PDFs are yours to source.

Run the native signature sweep against **two baselines** and confirm zero
regressions (no new word-fusions, over-splits, dropped spans, reversed scripts,
or crashes):

```bash
# Build the sweep tool once
cargo build --release --bin corpus_sig

# Your PR branch
./target/release/corpus_sig <your-corpus-dir> > head.txt

# Baseline A: main (current dev tip)
git worktree add /tmp/base-main main && \
  ( cd /tmp/base-main && cargo build --release --bin corpus_sig && \
    ./target/release/corpus_sig <your-corpus-dir> ) > base-main.txt

# Baseline B: the latest released version
git worktree add /tmp/base-rel v0.3.74 && \
  ( cd /tmp/base-rel && cargo build --release --bin corpus_sig && \
    ./target/release/corpus_sig <your-corpus-dir> ) > base-release.txt

diff base-main.txt head.txt        # expect: only your intended changes
diff base-release.txt head.txt     # expect: only your intended changes
```

For text-quality comparison use `scripts/regression_harness.py` (compares
extraction against a baseline and external references). **Judge regressions with
a structural metric — word-Jaccard plus a space/spacing-outlier check — not raw
character-Levenshtein, which is blind to word-gluing.**

In the PR you must **describe the corpus you used** (how many PDFs, what kinds,
where they came from) and **summarize the diff against both `main` and the
latest release**. "It builds and the unit test passes" is not a regression
result.

The maintainer runs the authoritative private-corpus sweep before merge — a
clean sweep on your own corpus is necessary but not sufficient.

### 4. Green across the whole feature matrix, not just `cargo test`
Default `cargo test` does **not** compile the rendering, FIPS, OCR, or binding
tiers, and PRs regularly break exactly those. Run the tiers your change touches:
```bash
cargo test --features rendering        # tiny-skia / shaping / fonts
cargo test --no-default-features --features fips,icc
cargo test --features ml               # OCR / table detection
# plus the relevant binding when you touch the C ABI (python / wasm / go / …)
```

### 5. Prefer semantic, tolerant comparison over byte-exact goldens
Compare extracted text/structure with whitespace/newline **normalization**.
For any rendered-pixel check, use a bounded per-channel tolerance at a fixed DPI
— never byte-exact; rendering is font- and platform-fragile. New parsing/decoding
paths should add a property-based or fuzz test where practical.

## Test fixture policy

- **No third-party, copyrighted, or reporter-supplied PDF binaries in the repo.**
  `tools/benchmark-harness/validate_fixtures.sh --strict` (the `fixture-hygiene`
  CI job) enforces this.
- **Build fixtures as minimal synthetic PDFs in code.** If a defect genuinely
  cannot be reproduced synthetically and needs a real specimen, it must be
  fetched at test time and pinned by hash, and the test must **skip gracefully
  when the file is absent** — never committed.
- **Name tests by the defect *class*, not by an issue or PR number**, and put no
  contributor or company names in code, comments, or fixtures. Good:
  `type0_identity_h_tj_word_seam`. Bad: `issue847`, `acme_corp_pdf`. Credit
  reporters in `CHANGELOG.md`, not in code.

## AI-assisted contributions

AI tools may be used **assistively**, and disclosed. The rules exist because
low-effort AI output consumes disproportionate reviewer time.

- **Autonomous agents may not open issues or PRs on their own.** PRs that appear
  to be agent-generated may be closed, perhaps without notice.
- **We do not accept PRs that are fully or predominantly AI-generated.** Code
  that an AI wrote and you then edited still counts as AI-generated.
- **You must understand and be able to explain every line you submit.** "The AI
  wrote it" is not an answer to a review question. If you can't explain the
  change without the AI, don't submit it.
- **Write your issues, PR descriptions, and review replies yourself**, in your
  own words. Don't paste AI-generated prose as your description.
- **Disclose AI assistance** in the PR (which tool, and how much). You — the
  human author — remain fully responsible for the code's correctness, licensing,
  and provenance regardless of how it was produced.
- **AI-generated code must be verified on real input you actually ran.** For this
  project that means: include the synthetic reproducer and the corpus-sweep
  result. Do not submit hypothetically-correct code you haven't executed.
- The effort test: **if the effort you put in is less than the effort we'd spend
  reviewing it, please don't open the PR.**

Good-faith first-timers who slip up will simply be pointed back here. Repeated,
bad-faith, time-wasting submissions lead to being blocked from the project.

## Coding standards

### Rust
- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/);
  `rustfmt` (config in `rustfmt.toml`), 100-char lines, 4-space indent.
- Use `Result<T>` and `?`; add context when wrapping errors. **No `unwrap()` in
  library code** (tests/examples are fine); `expect()` only with a descriptive,
  invariant-explaining message.
- **Fail loudly, don't fall back silently.** Extraction features may warn and
  degrade gracefully; security-critical operations fail closed. Never swallow an
  error into an empty/plausible-but-wrong result.
- `unsafe` requires a `// SAFETY:` comment stating the invariant that makes it
  sound. Prefer safe abstractions.
- All public items carry doc comments with an example where meaningful.
- **Follow the PDF specification, not linguistic heuristics.** See `AGENTS.md`
  and `docs/spec/` — e.g. word boundaries come from TJ offsets and geometry
  (ISO 32000-1 §9.4.4), never from CamelCase/dictionary guessing.

### Python
- `ruff` for formatting and linting (`ruff format` / `ruff check`); type hints on
  public functions; Google-style docstrings. Target **py39**.

## Continuous integration gates

Every PR runs the full matrix; all of it must be green before review. The
mandatory floor mirrors what you should run locally:

- `cargo fmt --check`
- `cargo clippy -- -D warnings`
- `cargo test` (default) **and** the affected feature tiers (`rendering`,
  `fips,icc`, `ml`, bindings)
- `features-powerset` (`cargo hack`), `msrv` build at the pinned floor,
  `semver-checks` on the public API
- `audit` / `deny` / `geiger` (advisories, licenses, `unsafe` surface)
- `fixture-hygiene` (no third-party fixtures), `taplo`/`shear` (TOML/unused deps)
- `dco` (sign-off), plus the per-binding jobs (python, wasm, go, csharp, java, …)

## Commits, DCO, and review

- **Conventional Commits**: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`,
  `perf:`, `chore:`. Each commit should build and pass tests on its own.
- **DCO sign-off is required** on every commit — certify you wrote the code and
  may contribute it under the project's licenses:
  ```bash
  git commit -s     # adds: Signed-off-by: Your Name <you@example.com>
  ```
  The `dco` CI job enforces this. There is no CLA.
- **Fill in the template.** Issues and pull requests opened without filling in
  their template are **closed automatically** (you'll get a comment explaining
  how) — edit yours to fill it in, then reopen and we'll pick it up. Maintainers,
  drafts, and items labelled `skip-template-check` are exempt.
- **Review**: a maintainer reviews and merges. Address feedback by pushing
  follow-up commits. **PRs left in "changes requested" without a response will
  be closed** to keep the queue clean — reopen when you're ready to continue.

## Security reports

Never report a vulnerability you cannot **reproduce and understand**; include a
working proof of concept and disclose whether AI was used to produce the report.
See [SECURITY.md](SECURITY.md) if present for private disclosure channels.
Speculative, unreproducible "findings" will be closed.

## License

By contributing you agree your contributions are dual-licensed under **MIT OR
Apache-2.0** (see [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE)),
with no additional terms. Inbound = outbound.

---

Questions? Check `docs/spec/` and `AGENTS.md`, or open an issue. Thank you for
contributing.

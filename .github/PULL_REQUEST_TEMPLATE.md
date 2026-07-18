<!--
Read CONTRIBUTING.md first. The two rules that matter most:
  1. Non-trivial PRs must reference an issue the maintainer has ACCEPTED.
     Drive-by PRs with no accepted issue may be closed without detailed review.
  2. Any extraction/layout/rendering/font change must be proven not to regress
     on a corpus of real PDFs (yours — the project corpus is private).
Write this PR in your own words. Delete the guidance comments as you fill it in.

NOTE: a PR opened without filling in this template is closed automatically —
just fill it in and reopen. (Maintainers, drafts, and `skip-template-check` are
exempt.)
-->

## Linked issue

<!-- Required for features/behavior changes. Bug/typo/docs fixes may omit. -->
Closes #

## What and why

<!-- In your own words: what problem does this solve, and why this approach?
     What did you consider and reject? -->

## Type of change

- [ ] Bug fix
- [ ] New feature (has an accepted issue: #___)
- [ ] Performance
- [ ] Refactor / internal
- [ ] Docs / CI / chore
- [ ] Breaking change

## Tests

- [ ] I added a test that **fails before this change and passes after**
      (revert-checked). For bug fixes the reproducer is a **minimal synthetic
      PDF built in code** — no third-party/reporter PDF is committed.
- [ ] Test named by defect **class**, not an issue/PR number; no
      contributor/company names in code or fixtures.
- [ ] `cargo test` passes, **and** the affected feature tiers:
      `rendering` / `fips,icc` / `ml` / binding (list which): ____
- [ ] `cargo fmt --check` and `cargo clippy -- -D warnings` are clean.

## Regression on real PDFs

<!-- Required for ANY extraction / layout / rendering / font change.
     The project corpus is private; use your OWN corpus. -->

**Corpus I tested** (count + kinds + source): <!-- e.g. "~120 PDFs: scanned OCR,
tagged, CJK+RTL, forms, 2-column academic, from my own collection" -->

- [ ] Ran `corpus_sig` and diffed my branch against **`main`** — no unintended
      regressions (no new word-fusions, over-splits, dropped spans, reversed
      scripts, or crashes).
- [ ] Diffed my branch against the **latest release** (`v0.3.__`) — same.
- [ ] Judged with a structural metric (word-Jaccard + spacing outliers), not
      char-Levenshtein.

**Diff summary vs `main`:** <!-- e.g. "3 docs changed, all the intended fix" -->

**Diff summary vs latest release:** <!--  -->

<!-- N/A only if this PR touches no extraction/layout/rendering/font code. -->

## AI assistance disclosure

- [ ] AI assistance: **none**, or **assisted** — tool: ______, extent: ______
- [ ] I understand and can explain every line; the description and my review
      replies are written by me, not generated. This PR is not fully or
      predominantly AI-generated.

## Checklist

- [ ] One logical change (no bundled refactor/perf/correctness).
- [ ] Commits follow Conventional Commits and are **DCO signed-off** (`git commit -s`).
- [ ] Docs/`CHANGELOG.md` updated if user-facing; reporters credited in the
      CHANGELOG (not in code).
- [ ] Public-API changes considered for semver (`semver-checks` will run).

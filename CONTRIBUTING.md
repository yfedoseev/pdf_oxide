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

- [Who depends on this](#who-depends-on-this)
- [The two rules that matter most](#the-two-rules-that-matter-most)
- [Code of Conduct](#code-of-conduct)
- [Before you open a pull request](#before-you-open-a-pull-request)
- [Communication volume](#communication-volume)
- [Development setup](#development-setup)
- [Testing and regression requirements](#testing-and-regression-requirements)
- [How review works, and when a PR is closed](#how-review-works-and-when-a-pr-is-closed)
- [Test fixture policy](#test-fixture-policy)
- [AI-assisted contributions](#ai-assisted-contributions)
- [Coding standards](#coding-standards)
- [Continuous integration gates](#continuous-integration-gates)
- [Commits, DCO, and review](#commits-dco-and-review)
- [Security reports](#security-reports)
- [License](#license)

---

## Who depends on this

Worth knowing before you read the rules, because it is where they come from.

PDFOxide is a production dependency of projects with a combined **195,000+
GitHub stars** — RAG engines, collaborative editors, AI coding agents,
document-intelligence frameworks, and arXiv.org's own submission pipeline. The
[Notable Users](README.md#notable-users) list in the README is verified against
each project's public dependency manifest.

It is also a library people can simply use. MIT/Apache-2.0, `cargo add` or
`pip install`, no account, no API key, no telemetry, no hosted service in the
path, no usage tier that expires. That is deliberate and we intend to keep it —
a PDF toolkit that phones home is not a PDF toolkit anybody should build on.

Both facts have the same consequence for contributions. A defect here does not
surface as a failed build in one place; it surfaces as wrong text inside someone
else's product, on a document neither of us has seen, with no error and nothing
to alert them. Extraction fails *silently* — that is the whole difficulty of
this domain. There is no telemetry to catch it and no server-side fix to push
afterwards; whatever ships is what runs, in every downstream product, until the
next release.

So the bar here is higher than "the tests pass", and the rules below are the
practical form of that bar rather than ceremony. They are also why a review can
take a while: the maintainer is checking the thing that CI structurally cannot.

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
participating you agree to uphold it. To report someone else's behavior, email
the maintainer at **yfedoseev@gmail.com** rather than opening a public issue.
That applies to reports about a person; it does not stop anyone — maintainer
included — from addressing something directly and civilly in the thread where it
happened.

Conduct is governed there. **Contribution volume and submission rate are governed
by [Communication volume](#communication-volume) below**, and the two are
separate: a rate limit or an intake cap reflects what one maintainer can review,
not a judgement about your conduct or your work.

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
- For a **bug fix**, open an issue too. It does not need discussion — a
  reproducer and a one-line description are enough.

**An issue must be approved *and assigned to you* before you open the PR.**
Two separate things, both required, both checked automatically:

1. **Approved by a maintainer.** Someone with write access to this repository
   comments `/approve` on the issue, or applies the `approved` label. A reply, a
   thumbs-up, or a question is *not* approval — approval is a deliberate act, so
   that "a maintainer engaged with it" cannot be mistaken for "a maintainer
   agreed to it".

   **Approval from anyone without write access does not count**, in either form.
   That includes the issue's own author and anyone with triage access who can
   apply labels: the check verifies the approver's actual repository permission,
   not their comment badge or their ability to set a label. Self-approval is not
   a thing here.

   Approval can be withdrawn later with `/unapprove`, or by removing the label;
   the most recent maintainer act is the one that counts.
2. **Assigned to you.** The maintainer assigns the issue to whoever will
   implement it. Assignment is how work is claimed here: it stops two people
   building the same fix, and it means the maintainer has agreed not just the
   approach but who is doing it. An issue approved and assigned to someone else
   is not yours to implement — say so in the issue and ask.

If you want to work on something, say so in the issue and ask to be assigned.
Opening a PR is not how you claim work.

**A pull request that references no issue at all is closed automatically.**
That is what the existing rule about drive-by PRs means in practice — work here
starts from an approved, assigned issue, and opening a PR is not how work is
claimed. Reference the issue in the **PR description** with `Closes #NNN` (not
in the commit message — it survives rebases better there).

**A PR whose linked issue is unapproved, or assigned to somebody else, fails its
checks and is not reviewed** — but is left open, because that is usually a
timing problem rather than a violation: the maintainer approving and assigning
the issue is enough to clear it, with nothing for you to redo. This is not bureaucracy and it is not about
trust — it costs a maintainer one line to say "go ahead" or "we'd fix that
differently", and it costs you nothing compared with discovering the same thing
after you have written the change and run a corpus sweep.

- **Trivially exempt** from all of the above: typo and documentation fixes, CI
  repairs, and reverts of your own merged work. Open those directly.
- Reference the issue in the **PR description** with `Closes #NNN` (not in the
  commit message — it survives rebases better there).

**2. One logical change per PR.** Do not bundle a bug fix with a refactor, or
correctness with performance. Grab-bag PRs are unreviewable and get split.
Complete the change within the PR — no "finish it later" TODOs for new features.

**3. One open PR at a time per non-maintainer contributor.** Review is the
scarcest resource on this project, not code. A second PR from the same author
while an earlier one is still open **will be closed automatically**, with a link
back to this section — reopen it once your first PR resolves (merged, closed, or
superseded). This applies regardless of how the PRs were produced or how good
the diagnosis behind them is: a burst of simultaneous PRs from one author costs
far more reviewer time in aggregate than the same fixes landed one at a time,
each actually verified before the next starts. Applies to every non-maintainer
contributor equally, human or AI-assisted.

- **Exceptions.** Documentation typos, CI fixes, and reverts of your own merged
  work do not count against the limit.
- **The maintainer may lift the limit** for a specific contributor or a specific
  piece of work. Ask in the issue.
- **Stacked or dependent work is one PR**, or opened one at a time in dependency
  order. If a change is genuinely too large to review in one piece, say so in
  the issue *first* and agree a split — do not open the split yourself as five
  simultaneous PRs.

If you have found several unrelated bugs, that is genuinely valuable — **open
issues for them**. Issues cost the maintainer far less than open PRs, they do
not rot against a moving `main`, and they let fixes be sequenced deliberately
instead of competing at once. The diagnosis work (reproducer, root cause, spec
citation) is valuable on its own, and a maintainer may pick it up directly.

**4. Keep the PR small enough to review.** For changes to extraction, layout,
rendering or font handling, a PR over roughly **400 changed lines** should be
split, or should say in its description why it cannot be. These paths are
heuristic and reviewed by reading, not by trusting the tests; past a few hundred
lines that stops being possible. Size is also the best available predictor of a
PR never landing at all.

**5. Do not re-submit a closed PR's work as a new PR.** If an approach was
declined, the way back in is an issue agreeing a different approach — not the
same change under a new number.

**6. Write the PR in your own words.** The description must explain *what
problem you're solving and why this approach*. If you can't describe the change
in your own words, it isn't ready.

**7. Run the full local gate before pushing** (see
[CI gates](#continuous-integration-gates)) — a green subset is necessary, not
sufficient.

**8. Disclose a material downstream interest.** If you maintain a fork of this
project, or a product that vendors it, say so in the PR description. Forking is
permitted — see [TRADEMARKS.md](TRADEMARKS.md) and [FORKS.md](FORKS.md) — and
downstream maintainers are often the best contributors here, because they run
this code against real documents at volume. This is a disclosure, not a
restriction, and it does not affect whether a change is accepted. It exists so
review effort can be weighed with the full picture visible, and so nobody has to
infer a relationship later.

## Communication volume

This project is maintained by one person in their free time. Reviewer attention
is the scarcest resource here — scarcer than code. These rules exist because
volume can cost more maintainer time than the contribution saves, however good
the underlying work is.

**Plainly, so nobody has to infer it: sustained spamming gets an account
blocked.** A blocked account cannot comment, cannot open issues, and cannot
submit pull requests — GitHub applies it to all three at once and to every
repository owned by the maintainer. Existing contributions stay in the history;
what ends is the ability to add more.

That is the last step, and it is reserved for two things: continuing after a
written request to stop, and programmatic submission of any kind. It is not for
being prolific, not for being wrong, and not for disagreeing with the
maintainer — people do all three here regularly and are welcome to. If you are
reading this and wondering whether it applies to you, it almost certainly does
not.

- **One topic, one thread.** Post a point once, in the most relevant issue or
  PR. If it applies to several PRs, post it once and link to it. **Do not paste
  the same comment into multiple threads** — that turns one point into one
  notification per thread, and the maintainer reads all of them.
- **No automated or scripted commenting.** Comments must be posted by a human,
  one at a time.

  **An account that posts faster than a person writes is treated as a bot**,
  whatever is behind it. Concretely: several comments within a minute, or the
  same text posted across multiple threads in one burst. Nobody composes
  nineteen comments in eighteen seconds, and content does not change that — a
  well-argued point delivered nineteen times is nineteen notifications, and the
  maintainer reads all of them.

  This is not a rule against using tools to help you write. It is a rule about
  the *rate*: whatever produced them, a burst is machine output arriving at
  machine speed into a queue served by one person.

- **The same rate limit applies to submissions, not just comments.** Issues and
  pull requests opened in bursts — several within a minute, or an issue and the
  pull request that closes it created seconds apart — are treated the same way.

  An issue filed moments before its own PR is not a proposal anyone could have
  responded to; it is paperwork generated alongside the code. The point of
  opening an issue first is that a maintainer gets a chance to say "yes",
  "not like that", or "we already fixed it upstream" *before* anyone writes the
  change. Filing both at once removes that chance while appearing to follow the
  rule, and it is why an issue now needs an explicit approval and an assignment
  before its PR is opened.

  Again this is about rate, not tooling. Generate the work however you like;
  submit it at a rate a person can respond to.

  **This one is enforced automatically.** A `Submission rate limit` workflow
  closes an issue or pull request opened less than **5 minutes** after your
  previous one, and flags a comment posted less than **60 seconds** after your
  previous one. Closed items can simply be reopened once the gap has passed —
  nothing is lost and it is not a rejection. Comments are never deleted; the
  workflow only leaves a note.

  Comments get the shorter window deliberately: answering three review threads
  in five minutes is ordinary participation, while nineteen comments in eighteen
  seconds is not, so the comment window targets bursts rather than conversation.
  Maintainers and toolchain bots (Dependabot and similar) are exempt.

  An account treated this way is **added to the auto-reject list** — its new
  issues and pull requests are closed automatically on arrival, without review.
  Removal is by asking, and by not doing it again.
- **Correct in place.** If a claim turns out to be wrong, edit the original
  comment or post one correction. Do not stack successive revisions of the same
  claim as new comments on the same thread.
- **Measure before you post.** Do not report a finding, a regression, or a
  stability claim you have not actually run. "I inferred it" costs the
  maintainer a real investigation. Say plainly what you measured, on what, and
  what you did not.
- **Scope claims to your evidence.** A result from your corpus is a result from
  your corpus. Do not present it as a property of this project's baseline, and
  do not ask for work to be resequenced on evidence the maintainer cannot
  reproduce.
- **Prefer fewer, denser messages.** A single well-organised comment is worth
  more than six partial ones. If you find yourself posting repeatedly to the
  same PR in one day, collect it into one update instead.
- **Do not use direct messages, email, or social media to chase a review.** The
  PR thread is the channel. Out-of-band pressure to prioritise your work is not
  acceptable.

The same effort test from the previous section applies to conversation, not just
code: **if reading your messages costs more than the problem they describe, they
are a net loss to the project.**

### When these are not followed

In order, and proportionate to what actually happened:

1. Duplicate or cross-posted comments are hidden as off-topic.
2. A single written request to consolidate.
3. **Auto-reject.** The account is added to the auto-reject list: new issues and
   pull requests are closed on arrival with a pointer to this section. Existing
   open work is not touched, and commenting still works — this caps intake
   without ending the conversation. It is not permanent: ask, and it comes off.
4. Interaction limits on the repository.
5. **Blocking.** A blocked account cannot comment, open issues, or submit pull
   requests — all three surfaces at once — and open contributions from it may be
   closed unmerged. This is the end of the ladder, not a first response: it is
   for sustained volume after a written request, or for programmatic submission,
   and it is the only step that is not trivially reversible.

Steps 1 and 2 are skipped for automated posting: a burst goes straight to step 3,
because there is no point asking a script to slow down.

Good-faith contributors who overdo it will simply be asked once. Blocking is for
sustained volume after a request, or for automated posting — not for being
enthusiastic, and not for disagreeing with the maintainer.

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
Add the test in a commit **before** the fix commit, so that checking out the
fix commit's parent and running the new test shows it red. Reviewers verify
this, and CI checks it. A test that passes even when the feature is a no-op is
worse than no test — it gives false confidence.

Ordering the other way round ("the test is in there, revert the fix and you'll
see") moves the work to the reviewer, who then performs the revert-and-rerun by
hand on every PR. One commit ordering on your side removes that entirely.

Three ways a test can look like it proves the fix and not prove it. Each of
these has shipped here, so they are named rather than left to judgement:

- **A fixture may not carry data the code under test never reads.** A form-field
  test once built an `/AP /N` dictionary the code never consults — it looked like
  it validated a specification rule and validated a name remap. Grep your
  fixture's distinctive keys against the code path you are fixing.
- **A test may not compute its expected value by calling the code under test.**
  If the expectation comes from the function being tested, the test cannot detect
  a change to it. Derive the expected value from the specification, or hard-code
  one you worked out by hand.
- **Changing what an existing test asserts needs a written reason.** Editing an
  assertion is a claim that the old expectation was wrong. Say so in the PR
  description and why. A test being in the way of your change is the case where
  it is most likely to be right.

CI enforces the first rule mechanically: for a `fix(...)` PR it checks out your
test commit, runs the tests it adds, and requires them to go red.

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

### 4. If your change *broadens* something, a corpus sweep cannot validate it
A sweep compares your branch against `main` and reports what moved. That
measures **change**, not correctness — and for one specific class of change it is
actively misleading.

If your change causes more data to be collected, matched, or retained than
before — first-match becomes all-matches, a filter is removed, a `break` is
deleted, a guard is loosened, a lookup drops part of a composite key — then
everything it *wrongly* picks up appears in the sweep as **more output**. More
extracted text reads as recovered content. A word-count or Jaccard comparison
cannot tell "recovered the run we were wrongly dropping" from "absorbed a run
that belongs to something else". Both are gains.

So for any broadening change you must **additionally** supply a fixture in which
the broadened operation would pick up something it should not, and show that it
does not:

- a **colliding identifier** — the same id legitimately used by two different
  scopes (a page and a Form XObject may each define marked-content id 0; a
  cross-reference stream and an object stream may each define object 4)
- a **duplicate key** — the same key present twice with different values
- an **out-of-scope match** — data that satisfies the loosened predicate but
  belongs to a different structure, page, or content stream

Build it synthetically per §2, and state in the PR what the fixture would have
collected without the guard. A clean sweep plus "no test for the collision case"
is not evidence; it is the absence of evidence in exactly the place the defect
would be.

### 5. Green across the whole feature matrix, not just `cargo test`
Default `cargo test` does **not** compile the rendering, FIPS, OCR, or binding
tiers, and PRs regularly break exactly those. Run the tiers your change touches:
```bash
cargo test --features rendering        # tiny-skia / shaping / fonts
cargo test --no-default-features --features fips,icc
cargo test --features ml               # OCR / table detection
# plus the relevant binding when you touch the C ABI (python / wasm / go / …)
```

### 6. Prefer semantic, tolerant comparison over byte-exact goldens
Compare extracted text/structure with whitespace/newline **normalization**.
For any rendered-pixel check, use a bounded per-channel tolerance at a fixed DPI
— never byte-exact; rendering is font- and platform-fragile. New parsing/decoding
paths should add a property-based or fuzz test where practical.

### 7. A claim in a comment, changelog or name must be true of the code

Four doc comments and two `CHANGELOG.md` entries in recent releases asserted
properties the merged code does not have. A changelog entry saying a bug is fixed
when it is not is the most costly of these: a reader removes a workaround that is
still needed.

- If a doc comment says "snapped to a quadrant", the function snaps.
- If a changelog entry says "every binding", every binding has it.
- If a test's name says what it proves, it proves that.

**If you cite a specification clause, quote the sentence you are relying on.**
A bare section number next to code is checked by readers *against each other*
rather than against the source — one inverted image mask survived review here for
exactly that reason: the comment cited the clause, the code matched the comment,
and both were wrong. A quoted sentence cannot agree with code that contradicts
it.

## How review works, and when a PR is closed

Review here is one person reading a change carefully. It is not a loop in which
the maintainer finds the problems and the contributor fixes them until the PR is
acceptable — that inverts the cost, and it is how a single PR consumes a week.

**What you send is expected to be complete.** CI checks the most mechanical
part of this for you — a `fix:` or `feat:` PR that adds no test fails before a
human looks at it — but the rest is on you. The requirements above are
published in advance: a regression test that fails without the fix, evidence of
no corpus regression, and — for a change that broadens what is collected — a
fixture proving it does not over-collect. A PR missing any of these has not been
reviewed and then rejected; it was **incomplete on arrival**, and may be closed
without a detailed review. Reopen it when it is complete. This is the "effort
test" from the AI-assisted section applied to review time.

**Repeated findings of the same kind end the review.** Review rounds are for
things neither of us could have anticipated. If the *same class* of problem has
to be raised twice — still no regression test, a claim still not measured,
unrelated changes still bundled in, a gate still failing — the PR is closed. A
*different* problem found in a later round is normal and is not counted.

This is deliberately about *repetition*, not about being wrong. Getting a hard
change wrong is expected and is what review is for. Having to be told the same
thing twice means the checklist was not applied, and the second telling costs
the maintainer as much as the first.

**Other reasons a PR is closed**, so none of these are a surprise:

- It references no issue, or the issue is not approved and assigned to you.
- It is a second concurrently-open PR from the same author.
- It sits in "changes requested" without a response.
- Its work is a re-submission of an approach already declined.

None of these is a judgement about you or about the underlying diagnosis, and
none is permanent. Closing a pull request costs nothing to undo; a review cycle
cannot be undone at all, which is the asymmetry all of this exists to manage.

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
  own words — this covers *everything you write to a person*, not just the PR
  body: issue text, PR descriptions, review comments, and replies in a thread.
  Do not paste AI-generated prose into a conversation.

  This is not a style preference. Generated prose is fast to produce and slow to
  read, and it is usually longer, more confident, and less specific than what
  the same person would have written. The maintainer then has to read all of it
  and work out which parts are load-bearing. A three-line answer in your own
  words is worth more here than six paragraphs, and is far more likely to be
  acted on.
- **Disclose AI assistance** with a commit trailer naming the tool:

  ```
  Assisted-by: claude-code
  ```

  The model is not interesting and you do not need to name it — any model can
  produce work that is not ready, and naming one neither excuses nor condemns
  the change. What matters is that a person is answerable for it. Say in the PR
  description how much of the change it produced. You — the human
  author — remain fully responsible for the code's correctness, licensing, and
  provenance regardless of how it was produced.
- **An AI agent may not sign off a commit.** `Signed-off-by:` is the Developer
  Certificate of Origin: a statement by a person that they have the right to
  contribute the code. A tool cannot make that certification, so the sign-off is
  yours and the responsibility that comes with it is yours. Use `Assisted-by:`
  for the tool, never `Signed-off-by:` or `Co-authored-by:`.
- **AI-generated code must be verified on real input you actually ran.** For this
  project that means: include the synthetic reproducer and the corpus-sweep
  result. Do not submit hypothetically-correct code you haven't executed.
- The effort test: **if the effort you put in is less than the effort we'd spend
  reviewing it, please don't open the PR.**

Good-faith first-timers who slip up will simply be pointed back here. Repeated,
time-wasting submissions lead to being blocked — see
[Communication volume](#communication-volume) for what blocking means and the
steps that come before it.

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

Two of these are **intake gates**, run by `PR quality gates` on every pull
request from a non-maintainer:

- **One open PR per contributor** — a second concurrently-open PR is closed
  automatically with a link to the rule.
- **Linked issue is approved and assigned to you** — **closes** the PR if it
  references no issue at all; **fails** it (leaving it open) if the issue exists
  but nobody with write access has approved it, or if it is assigned to somebody
  else. A PR that mentions an issue without a closing keyword is failed, not
  closed, with a note to use `Closes #NNN`. Approval is resolved by replaying `/approve`, `/unapprove` and
  `approved`-label events in order and keeping the last one performed by an
  account that actually holds write access.

A separate workflow, **Submission rate limit**, closes an issue or pull request
opened within 5 minutes of the same account's previous one, and flags a comment
posted within 60 seconds of their previous one. Thresholds are repository
variables (`RATE_MIN_GAP_ISSUES`, `RATE_MIN_GAP_PRS`, `RATE_MIN_GAP_COMMENTS`);
setting one to `0` disables that surface. Write-access accounts and toolchain
bots are exempt.

A further workflow, **Auto-reject list**, closes new issues and pull requests
from accounts whose intake has been capped under
[Communication volume](#communication-volume). It does not touch comments or
existing open work, and it never applies to anyone with write access.

A third gate applies to everyone, maintainers included:

- **Change ships a test** — a `fix:` or `feat:` PR must add at least one new
  `#[test]`. Inline `#[cfg(test)]` tests count exactly as much as files under
  `tests/`. If a change genuinely cannot be tested — a pure refactor, or a defect
  only reproducible on a document that cannot be shared — say so in the
  description and a maintainer will waive it.

Drafts are exempt from all three. The two intake gates additionally exempt
`docs:`, `ci:`, `chore:` and `revert:` titles, and anyone with write access.

## Commits, DCO, and review

- **Conventional Commits**: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`,
  `perf:`, `chore:`. Each commit should build and pass tests on its own.
- **DCO sign-off is required** on every commit — certify you wrote the code and
  may contribute it under the project's licenses:
  ```bash
  git commit -s     # adds: Signed-off-by: Your Name <you@example.com>
  ```
  The `dco` CI job enforces this.
- **CLA** — non-trivial contributions are accepted under the project's
  [Contributor License Agreement](CLA.md). It is a *licence, not an assignment*:
  you keep ownership of your work and grant the project a broad licence to use
  and sub-licence it. Trivial changes (typos, formatting, docs) are exempt. Once
  the CLA bot is enabled it records your one-click sign-off on your first PR;
  until then the DCO sign-off above is the operative requirement. See
  [CLA.md](CLA.md) and the licence/trademark note in the README.
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

By contributing you agree that the **outbound licence for released code is
MIT OR Apache-2.0** (see [LICENSE-MIT](LICENSE-MIT) and
[LICENSE-APACHE](LICENSE-APACHE)) — inbound = outbound for what ships to users. In
addition, **non-trivial contributions are made under the project's
[Contributor License Agreement](CLA.md)**, which grants the Maintainer a broader,
sub-licensable copyright and patent licence so the project can relicense *future*
versions if it ever needs to. The CLA does not change the licence of any
already-published release. Trivial changes (typos, formatting, docs) are exempt
from the CLA and remain inbound = outbound only.

---

Questions? Check `docs/spec/` and `AGENTS.md`, or open an issue. Thank you for
contributing.

# Forking PDFOxide

PDFOxide is dual-licensed `MIT OR Apache-2.0`. **Forking is allowed, and this
page is not an argument against it.** Forks exist for good reasons — a different
release cadence, a smaller build, a feature we declined, a product that needs the
engine on its own terms. This page says what the licence requires, what the
trademark policy requires, and what you can and cannot expect from us afterwards.

This project does **not** maintain a register of forks, and does not rank,
endorse or comment on them.

## What the licence requires

You may fork, modify, relicense under either arm of the dual offer, and ship
commercially, with no obligation to contribute anything back.

- **Keep the copyright notice.** Both licences require it. Retain the existing
  `Copyright (c) … Yury Fedoseev` line in your `LICENSE` and add your own
  alongside it — do not replace it.
- **If you elect Apache-2.0**, §4(d) additionally requires you to carry forward
  the attribution notices in [NOTICE](NOTICE). Electing MIT instead is permitted
  and removes that requirement; either arm is a valid choice.
- **Vendored third-party code carries its own terms.** If you keep our vendored
  components and bundled fonts, keep their licences with them.

## What the trademark policy requires

The code is open source; the name is not. See [TRADEMARKS.md](TRADEMARKS.md) for
the full policy. In short:

- **Give it a different name** — one that does not contain "PDFOxide",
  "pdf_oxide" or "pdf-oxide" as a component, prefix and suffix included.
- **Say it is not the official project.** A line in your README is enough.
- You may accurately describe it as *"a fork of PDFOxide"* or *"derived from
  PDFOxide"*. That is explicitly permitted and is not something you need to ask
  about.

None of this is adversarial. The policy exists so a user who types the official
package name gets the official package — the same protection your own fork's name
will give your users.

## What to expect from us

- **We support only the official distributions**: the `pdf_oxide` crate and the
  first-party bindings published from this repository. If you are running a fork
  and something misbehaves, its maintainer is the right place to start — we
  cannot reproduce against code we do not ship.
- **Bug reports must reproduce against an official release.** A report against a
  fork's build tells us nothing actionable about ours.
- **Security reports are the exception.** If you find something in forked code
  that plausibly also affects us, report it under [SECURITY.md](SECURITY.md) and
  we will treat it as ours until shown otherwise.

## Sending fixes back

Fixes are welcome and are the cheapest way to stop carrying a patch set. Normal
rules apply — see [CONTRIBUTING.md](CONTRIBUTING.md) — with one addition: say in
the PR that you maintain a fork. That is a disclosure, not a restriction, and it
does not affect whether a change is accepted.

If you would rather not open pull requests, an issue with a reproducer and a root
cause is genuinely valuable on its own, and often more so than a patch.

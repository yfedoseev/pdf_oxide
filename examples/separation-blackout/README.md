# Separation / DeviceN tint-transform black-out reproducer

These PDFs reproduce a colour-rendering bug: a **`Separation`/`DeviceN` spot
colour set at full tint with `scn` rendered as solid black** instead of being
resolved through its tint transform. It blacks out tinted callout boxes and
headings on InDesign-exported PDFs.

## The reproducer

Each PDF is a single 100×100 pt page whose entire area is filled, via `1 scn`,
with a `Separation` colour over a `DeviceCMYK` alternate. The tint transform
maps tint `1.0 → CMYK(0.1, 0, 0.15, 0)` — a light green, **RGB ≈ (230, 255, 216)**.
The two files differ only in the tint-transform function type:

| File | Colour space | Tint transform | Alternate | Centre pixel |
| --- | --- | --- | --- | --- |
| [`separation-type2.pdf`](separation-type2.pdf) | Separation | `FunctionType 2` (exponential) | DeviceCMYK | light green ≈ (230,255,217) |
| [`separation-type0.pdf`](separation-type0.pdf) | Separation | `FunctionType 0` (sampled) | DeviceCMYK | light green ≈ (229,255,217) |
| [`separation-type4.pdf`](separation-type4.pdf) | Separation | `FunctionType 4` (PostScript) | DeviceCMYK | light green ≈ (230,255,217) |
| [`devicen-type4.pdf`](devicen-type4.pdf) | DeviceN (2 colorants) | `FunctionType 4` (2→4) | DeviceCMYK | green ≈ (51,255,102) |
| [`separation-lab.pdf`](separation-lab.pdf) | Separation | `FunctionType 2` | **Lab** | red ≈ (255,0,0) |

The page content stream is just (two operands for the 2-colorant DeviceN):

```
/CsTest cs 1 scn 0 0 100 100 re f
```

## Expected vs. buggy

| | FunctionType 2 | FunctionType 0 | FunctionType 4 |
| --- | --- | --- | --- |
| **Fixed** (light green) | ![](separation-type2-fixed.png) | ![](separation-type0-fixed.png) | ![](separation-type4-fixed.png) |
| **Buggy** (solid black) | ![](separation-type2-buggy.png) | ![](separation-type0-buggy.png) | ![](separation-type4-buggy.png) |

The multi-colorant DeviceN and Lab cases render as below. Their "buggy" images
are the state at the parent commit (`dbc832a`), before this change: Type 4 was
unevaluated (DeviceN and Type 4 Separation fell back to grey → black) and the
Lab alternate's `[L*, a*, b*]` output was mistaken for out-of-range RGB → black.

| | DeviceN Type 4 (green) | Separation + Lab (red) |
| --- | --- | --- |
| **Fixed** | ![](devicen-type4-fixed.png) | ![](separation-lab-fixed.png) |
| **Buggy** (solid black) | ![](devicen-type4-buggy.png) | ![](separation-lab-buggy.png) |

## Root cause

In `src/rendering/page_renderer.rs`, the `"Separation" | "DeviceN"` arm of
`SetFillColorN` (`scn`) never evaluated the tint transform — it always used the
bare `grey = 1 - tint` fallback, so a full tint (`1.0`) became black. The
`SetFillColor` (`sc`) arm only handled `FunctionType 2` with a `DeviceCMYK`
alternate, so sampled (`FunctionType 0`) transforms and RGB/Gray alternates hit
the same fallback.

The fix evaluates the tint transform and maps the alternate-space output to RGB:

- **`FunctionType 0` and `2`** are evaluated for the single-input Separation case.
- **`FunctionType 4`** (PostScript calculator, §7.10.5) is wired in for any input
  arity, which also makes true **multi-colorant DeviceN** (n inputs → m outputs,
  §8.6.6.5) render correctly. Multi-input `FunctionType 0`/`2` (unsupported) and
  any unevaluable transform still degrade safely to grey.
- The **alternate colour space** (the arr[2] slot) is inspected rather than
  guessed from component count, so a `Lab` (§8.6.5.4) or non-RGB `ICCBased`
  alternate is converted correctly (CIELAB→sRGB for Lab) instead of being read
  as `DeviceRGB`.

## Reproduce

Render the PDFs to PNG (writes `*-rendered.png` next to them and prints the
centre pixel):

```sh
cargo run --example render_separation_demo --features rendering
```

Or run the regression test, which renders both fixtures and asserts the centre
pixel is light and green-dominant rather than black:

```sh
cargo test --features rendering --test separation_color
```

## Regenerating the PDFs

The PDFs are committed artifacts. They were generated with a small throwaway
[`lopdf`](https://crates.io/crates/lopdf) program (not a dependency of this
crate). The `FunctionType 0` sample stream is two CMYK samples — `00 00 00 00`
(tint 0) and `1A 00 26 00` (tint 1, i.e. `0.1, 0, 0.15, 0` at 8 bits/sample) —
with `Size [2]`, `BitsPerSample 8`, `Range [0 1 0 1 0 1 0 1]`.

The later fixtures use these tint transforms (also via the throwaway generator):

- `separation-type4.pdf` — `FunctionType 4` program
  `{ dup 0.1 mul 0 3 -1 roll 0.15 mul 0 }`: tint `t → CMYK(0.1t, 0, 0.15t, 0)`.
- `devicen-type4.pdf` — 2-colorant DeviceN, `FunctionType 4` program
  `{ exch 0.8 mul 0 3 -1 roll 0.6 mul 0 }`: tints `[a, b] → CMYK(0.8a, 0, 0.6b, 0)`.
- `separation-lab.pdf` — `FunctionType 2` over a `Lab` alternate
  (`/WhitePoint [0.9505 1.0 1.089]`, D65): tint `1.0 → Lab(53.24, 80.09, 67.2)`,
  i.e. sRGB red.

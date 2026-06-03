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

| File | Tint transform |
| --- | --- |
| [`separation-type2.pdf`](separation-type2.pdf) | `FunctionType 2` (exponential interpolation) |
| [`separation-type0.pdf`](separation-type0.pdf) | `FunctionType 0` (sampled) |

The page content stream is just:

```
/CsTest cs 1 scn 0 0 100 100 re f
```

## Expected vs. buggy

| | FunctionType 2 | FunctionType 0 |
| --- | --- | --- |
| **Fixed** (light green ≈ 230,255,216) | ![](separation-type2-fixed.png) | ![](separation-type0-fixed.png) |
| **Buggy** (solid black 0,0,0) | ![](separation-type2-buggy.png) | ![](separation-type0-buggy.png) |

## Root cause

In `src/rendering/page_renderer.rs`, the `"Separation" | "DeviceN"` arm of
`SetFillColorN` (`scn`) never evaluated the tint transform — it always used the
bare `grey = 1 - tint` fallback, so a full tint (`1.0`) became black. The
`SetFillColor` (`sc`) arm only handled `FunctionType 2` with a `DeviceCMYK`
alternate, so sampled (`FunctionType 0`) transforms and RGB/Gray alternates hit
the same fallback.

The fix evaluates the tint transform (`FunctionType 0` and `2`) and maps the
alternate-space output to RGB by component count (1 → Gray, 3 → RGB, 4 → CMYK).

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

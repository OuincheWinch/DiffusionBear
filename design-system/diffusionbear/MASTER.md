# Design System — DiffusionBear

Source of truth for the interface. This file **overrides** anything the
`ui-ux-pro-max` skill generates, and the reason is recorded under
"Why this file exists instead of the generated one".

Read this before building a new surface. Page-specific overrides go in
`pages/<name>.md` and take precedence over this file.

---

## What DiffusionBear is

A single-window macOS desktop app for local image generation. Not a website, not a
landing page, not a dashboard. The person using it is generating an image right now,
often waiting on a 30–280 s render, with a queue running behind it.

That drives every decision below: the canvas is the subject, the queue must never be
ambiguous about what is happening, and nothing may compete with the image for
attention.

## Global Rules

### Colour — measured, not chosen

Every value below is in `frontend/src/App.css` `:root` and is verified by
`backend/test_contrast.py`. The ratio column is measured against the surface named.
**Do not change a token without running that test**; it will tell you if the change is
legal.

| Token | Hex | Role | Measured | Required |
|---|---|---|---|---|
| `--bg` | `#12121a` | App background | — | — |
| `--panel` | `#1c1c28` | Cards, sections, inputs | — | — |
| `--text` | `#e6e6f0` | Body text | 15.03:1 on bg, 13.60:1 on panel | 4.5:1 |
| `--muted` | `#8a8aa0` | Hints, labels, units | 5.52:1 on bg, 4.99:1 on panel | 4.5:1 |
| `--accent` | `#7c5cff` | **Borders and fills only** | 4.29:1 on bg, 3.88:1 on panel | non-text |
| `--accent-text` | `#a78bfa` | Accent **as text**: links, active preset, badges | 6.85:1 on bg, 6.19:1 on panel | 4.5:1 |
| `--primary` | `#2563eb` | Primary button fill | 5.17:1 with white label | 4.5:1 |
| `--primary-hover` | `#1d4ed8` | Primary hover | 6.70:1 with white label | 4.5:1 |
| `--focus-ring` | `#93c5fd` | Keyboard focus ring | 10.33:1 on bg, 9.35:1 on panel | 3:1 |
| `--checkbox-border` | `#6b7280` | Control edge | 3.49:1 on panel | 3:1 |
| `--switch-knob` | `#9ca3af` | Switch knob, off | 6.64:1 on panel | 3:1 |
| `--danger` | `#f87171` | Error text | 6.09:1 on panel | 4.5:1 |
| `--success` | `#4ade80` | Success text | 9.67:1 on panel | 4.5:1 |

**The `--accent` / `--accent-text` split is load-bearing.** `--accent` at 4.29:1
fails 4.5:1 as text. It was being used for text in 19 places, which is 19
WCAG AA failures. All of them now use `--accent-text`; `--accent` remains correct for
borders, fills and the tint behind an active preset.

A test asserts that `--accent` *stays* below 4.5:1. If it ever passes, the split can
be retired — that is the signal, not a reason to remove it early.

### Typography

System UI stack (`system-ui, -apple-system`). No web font, deliberately: the app runs
offline, a web font is a network dependency and a licence to track, and the
generative tool's whole premise is that nothing leaves the machine.

- Base: browser default 16px, untouched. Every `rem` in the stylesheet assumes this.
- Body: 1rem / 1.5 line-height.
- Labels and hints: 0.72–0.82rem. These are the small end of the scale and are
  checked by `test_contrast.py` rather than trusted.
- Nothing below 0.68rem.

### Spacing

4px base. Section padding 1rem, field gap 0.2–0.6rem. The settings sections are
compact by design — the tab holds a lot of controls and a user should not scroll to
find one.

### Components

The settings form has real primitives in `frontend/src/settings-form.css`, because
the per-component rules that preceded them had drifted apart and produced four
visible defects:

- `input[type="checkbox"].settings-checkbox` — sized control. There was no
  checkbox rule in the app at all before this; the native 13px control was the
  visible bug.
- `.settings-switch` — for boolean engine behaviours, `role="switch"`.
- `.settings-field` / `.settings-grid` / `.settings-inline--split` — controls in a
  field share the cell width; a row mixing a control and a boolean is two explicit
  columns aligned on the control's box.
- `.settings-actions` — a full-width row beneath the fields, so a save button can
  never be read as belonging to one control.

### Focus

One global rule, `:where(a, button, input, select, textarea, [tabindex]):focus-visible`,
2px ring at `--focus-ring` with 2px offset. Drawn outside the control so an
`overflow: hidden` ancestor cannot clip it.

Six `outline: none` rules once replaced the ring with a 1px border-colour change,
which is not a focus indicator: it disappears against a busy background and entirely
under `forced-colors`. There are now **zero** `outline: none` declarations in the
stylesheet.

### Motion

Transitions are 150–300ms on colour properties only. Never on width, height or
position — those are layout. Everything collapses under
`@media (prefers-reduced-motion: reduce)`.

`forced-colors: active` is handled: the switch and checkbox change **shape** as well
as colour, so on/off survives when the backgrounds are discarded.

### Icons

**Emoji are in use as section icons** (🧠 📦 💾 🔐 🖥 🗂 ⏳ ⚙ 🌐, ~27 glyphs). The
generated checklist says "no emojis as icons, use SVG". This is a known, deliberate
deviation, not an oversight: they render identically on every Mac with no asset
pipeline, and they carry meaning alongside a text label in every current use.

**If you add an icon-only button, it needs an SVG or an `aria-label`.** The emoji
exception applies to labelled section headers, not to unlabelled controls. That is
where the guideline is right and it does bind.

### Responsive

Only one breakpoint exists (`max-width: 900px`, collapsing the studio layout to a
single column) plus the language picker's `min-width: 720px`.

This is intentional for a desktop app with a resizable window, but it is a real
limitation: the Parameters tab was not verified below ~700px. The generated checklist
asks for 375/768/1024/1440 testing. **768 and 1024 are the widths that matter** here
and neither has been checked in a browser. Treat that as open.

---

## Why this file exists instead of the generated one

`ui-ux-pro-max --design-system` was run as instructed. Its output was rejected for
this product, and the reasons are recorded so nobody re-runs it and assumes the
result is authoritative:

1. **It proposed a marketing landing page.** Pattern: "Hero + Testimonials + CTA",
   with a testimonial carousel. DiffusionBear has no marketing surface, no
   testimonials and no conversion goal. The matcher is biased toward landing pages.
2. **It produced a light palette** (`#FAF5FF` background) for a dark image editor.
   A light chrome around a dark canvas also breaks the app's one real advantage,
   which is judging an image against neutral surroundings.
3. **Its colour numbers were not measured against this app's surfaces.** They are a
   generic palette, and importing them would have replaced a palette that passes
   16/16 with one of unknown contrast.
4. **Its primary `#1E293B` / accent `#22C55E`** is a slate-and-green developer-tool
   identity. The existing violet is fine and the ratios now pass.

The skill's *guideline* data was genuinely useful and is what this file is built on:
the WCAG contrast requirements, the focus-indicator rules, the reduced-motion and
forced-colors handling, and the pre-delivery checklist. The *generated design system*
was not. Those are different things, and the skill's own workflow says to treat
search results as recommendations.

## Pre-delivery checklist

Run before shipping UI. Each line is either verified by a test or a measurement, or
explicitly marked unverified.

- [x] Text contrast 4.5:1 — `backend/test_contrast.py`, 16/16 pairs
- [x] Control boundaries and focus ring 3:1 — same test
- [x] Visible focus ring on every interactive control — one global rule, zero `outline: none`
- [x] `prefers-reduced-motion` respected — settings form, verified in bundle
- [x] `forced-colors` handled — shape change, not colour alone
- [x] Labels on all inputs, no placeholder-only fields
- [x] `cursor: pointer` on clickable elements — base `button` rule
- [x] Hover states with 150–300ms transitions
- [ ] **No emoji as icons** — DEVIATION, documented above. Applies to icon-only controls
- [ ] **Responsive 375/768/1024/1440** — NOT VERIFIED. Only 900px exists
- [ ] Visual confirmation in the running app — I cannot see the screen; ask the user

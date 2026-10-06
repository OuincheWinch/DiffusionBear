# DiffusionBear — Engineering Handoff

**Read this before changing anything.** This document exists because most of the
expensive knowledge in this project is not in the code. Several things here look
like bugs, work locally, and only fail on someone else's machine. Others look
fine and are silently dead.

Written at the v0.3.5 release. Private repo `65181a2`, public `35c05ab`.

---

## 1. What this is

A local, offline-capable image generator for Apple Silicon. Five model families
(FLUX.2, Krea 2, Qwen-Image 2.1, Z-Image, SDXL) running through MLX/mflux, shipped
as a single ad-hoc-signed macOS `.app`. No Apple Developer ID, not notarised, no
model weights bundled.

It is **not** a hosted service, not cross-platform, and not a general MLX
front-end. Everything here is tuned for one machine class (16 GB Apple Silicon,
macOS 15+).

### Scale

| | |
|---|---|
backend Python | 48 modules, 20,573 lines |
backend routers | 9 modules, 3,711 lines |
backend tests | 29 modules, 8,138 lines, 514 tests |
frontend | 66 modules, 18,594 lines + 5,128 lines of CSS |
packaging | 2,166 lines of shell + Swift |
shipped bundle | 36,151 files, 1.6 GB, zero writable state |

---

## 2. Repository topology — this confuses people

There are **three** trees. They are not copies and they are not in sync.

| tree | path | role |
|---|---|---|
**private DEV** | `/Volumes/Externe/IA/MLX-Diffusion-DEV` | source of truth, origin |
**public** | `github.com/OuincheWinch/DiffusionBear` | filtered source for readers |
**legacy checkout** | `/Volumes/Externe/IA/MLX-DIFFUSION OpenCode` | superseded, only its `AGENTS.md` is still read |

Private → public is a **filtered export**, not a push. Use the committed script:

```bash
cd packaging
./export_public.sh          # build the tree, audit it, print a summary
CONFIRM_PUSH=1 ./export_public.sh --push
```

The exclusion list has only ever lived in terminals until now, which is how
0.3.4 nearly published `data/lora_files` (425 MB) and `data/hf_token.txt` — a real
credential. The script audits every export for this machine's identifiers, for
credential-shaped **content**, for weights, and for the store itself, and refuses
to proceed on a hit.

Two details worth knowing before you touch the audit:

- It matches **credentials by content signature** (`hf_` + 30 chars), not by
  filename. Grepping for `hf_token` flags every module that legitimately
  constructs `DATA_DIR/hf_token.txt`.
- It matches **`/Volumes/Externe`**, the volume, not the repo path. Naming only the
  repo missed the sibling store path that also shipped. Capital `E` keeps it clear
  of upstream mflux docstrings, which write lowercase `/Volumes/flux2-...`.

Consequences you must internalise:

- **Public and private commit hashes differ** for identical content. Public `35c05ab`
  and private `65181a2` are the same tree minus four files. This is not drift.
- **Public `main` must be force-pushed**, because the filtered history has different
  parent hashes than the private history.
- **Tests and `packaging/vendor/mflux-src` stay public.** Do not strip them; the
  public CI runs the backend suite.
- Anything you add that is private-only must be added to `EXCLUDES` in
  `packaging/export_public.sh`, or it silently ships.

## 2b. Read `docs/HANDOFF.md`

A full engineering handoff — architecture, engine quirks, the storage contract,
every "never do this", dead code, open issues, and the testing architecture — is in
`docs/HANDOFF.md` in this repo. It is excluded from the public export because it
names this machine's paths. Start there.

---

## 3. Architecture

### 3.1 Process topology

```
DiffusionBear.app/Contents/MacOS/DiffusionBear     Swift/AppKit shell
  └── spawns (Process, inherits env)               uvicorn main:app  → 127.0.0.1:8001
        └── spawns (subprocess.Popen, no env=)      venv-sdxl/bin/python sdxl_engine.py --serve
        └── spawns (subprocess.Popen, no env=)      venv/bin/python qwen_engine.py
```

Both engines inherit the parent's environment, which is why
`DIFFUSIONBEAR_ASSET_DIR` reaches them. Neither re-derives the store root from
scratch. **If you add an engine, it must not invent its own default store path** —
that exact bug made SDXL look in the wrong directory on a test machine.

### 3.2 Resources layout is a contract

```
Contents/Resources/
  backend/      application .py, incl. sdxl_engine.py
  venv/         main runtime (Python 3.10, mflux/torch/mlx)
  venv-sdxl/    Python 3.14, mlx_diffuser/diffusers
  python/       bare python-build-standalone (venv's base)
  python-sdxl/  bare pbs for venv-sdxl
  frontend/dist production SPA
```

The names are load-bearing: `generator.py:3245` finds SDXL at
`<root>/venv-sdxl/bin/python`, `generator.py:3533` finds qwen at
`<root>/venv/bin/python`, `main.py:63` serves the SPA from `<root>/frontend/dist`.
Renaming any of them breaks the app with no env override available.

### 3.3 The storage contract — read this twice

Everything mutable lives in `DATA_DIR`. Nothing may be written inside the bundle.

```python
DATA_DIR  = env DIFFUSIONBEAR_DATA_DIR  or  <backend>/data      # fallback is INSIDE the bundle
ASSET_DIR = env DIFFUSIONBEAR_ASSET_DIR or  DATA_DIR
```

`~/Library/Application Support/DiffusionBear/` holds either the store itself or a
`store_path` file containing one absolute line. The Swift shell reads it
(`main.swift:71-95`) and exports the result; the backend never reads it. So:

| situation | `DATA_DIR` |
|---|---|
normal launch from the app | correct store, env set by the launcher |
backend started by hand without env | **inside the signed bundle** |
`store_path` missing | default `~/Library/Application Support/DiffusionBear/data` |

Store contents on the reference machine: `models/` (82 GB), `SDXL/` (11 GB),
`generated/` (1.6 GB), `lora_files/` (1.5 GB), `uploads/`, `settings.json`,
`loras.json`, `hf_token.txt`, `civitai_token.txt`, `pending_queue.json`,
`queue_recovery.json`, `Logs/`.

**Never delete `generated/`.** It is the user's gallery.

### 3.4 Engine quirks that will bite you

| engine | notes |
|---|---|
**FLUX.2** | guidance-distilled. **No negative prompt** — the mflux API has no such parameter. 4 steps. LoRAs must be FLUX.2 architecture. |
**Krea 2** | 8 steps. Its published native buckets are multiples of **32**, not 64. |
**Qwen-Image 2.1** | 25 steps, **no LoRAs**. Native is 2K (~4.2 MP) which is 7× the largest size the picker offers. **1024×1024 is a confirmed OOM on a 16 GB M1.** Advisory-warned, not blocked. |
**Z-Image** | 6 steps. Any aspect ratio by *area*, 512²–2048². |
**SDXL** | 4 engines. Two are Lightning (4–6 steps). `quantize_unet=4` is mandatory: ~5× faster than fp16 and half the memory. **8-bit is 7× slower — never enable it.** |

Model registry: `backend/generator.py:161` (`MODELS`). Nine entries across five
ecosystems, all defaulting to 512×768.

SDXL runs through `venv-sdxl` + `mlx_diffuser` and requires **diffusers layout**
(`model_index.json` + a populated `unet/`). Almost every SDXL checkpoint on Hugging
Face is a single file and cannot work. `backend/test_sdxl_layout.py` enforces this
at adopt time.

---

## 4. Commands

```bash
# backend tests (514; must stay green)
cd backend && "/Volumes/EXterne/IA/MLX-DIFFUSION OpenCode/venv/bin/python" -m unittest discover -q

# frontend
cd frontend && npm run lint          # oxlint, must be 0 errors
cd frontend && npm run build         # vite
cd frontend && npm run i18n:build    # regenerate es/zh/ja/pt/ko from _parts/*.json

# component render harness (catches throw-on-render; see §6)
cd frontend && node tools/run-size-selector-render.mjs .

# build + install to /Applications, with gates
cd packaging && INSTALL_APP=1 ./build_app.sh

# runtime venvs, only when changing dependencies
./build_runtime_venv.sh && ./build_sdxl_venv.sh
```

`build_app.sh` runs ~5 minutes and refuses to ship on a failed gate. It installs
only when `INSTALL_APP=1`.

---

## 5. Good practices — established here, keep them

### 5.1 Data layer separate from view

`frontend/src/components/sizePresets.js` holds every resolution, the orientation
logic, the model gating and the arithmetic. `SizeSelector.jsx` only renders. The
module imports no React, no i18n and touches no DOM, so `backend/test_size_selector.py`
executes **the shipped file** under node rather than a Python re-implementation.

That matters because the original size bug survived review precisely this way: the
Python copy had the arithmetic, the component had a different line, and only the
component runs.

### 5.2 Test the code that runs, not a copy

Same principle in three places: node-executes-the-component, AST-walks-the-source
(`test_taesd_path.py`), and `vite build` bundles the real component.

### 5.3 Verify claims before acting on them

Several plausible-sounding claims in this project were wrong and were disproved by
measurement. Notable corrections, all now in the code's comments:

- mflux does **not** reject a non-multiple-of-16 dimension. It logs *"Width and
  height should be multiples of 16. Rounding down."* and floors it. So a bad
  dimension is a **silently different size**, not a failed generation.
- `mx.get_peak_memory()` is **not** a memory metric. It reported 24.29 GB on a
  16 GB machine. The only trustworthy signals are *did the process survive* and
  *is the output a real image*.
- The per-model pixel caps were removed by decision on 2026-09-22. `max_pixels`
  plumbing still exists; do not assume it is enforced.

### 5.4 A gate must fail loudly, and must be tested

Every build gate added this session was itself wrong once. Each fix was verified by
deliberately reintroducing the defect and confirming the gate fires. Do that for
new gates too — an untested gate is a gate that may be silently inverted.

### 5.5 Owner tests the compiled app before publication

`Principles.md` rule 7, in the legacy checkout. Non-negotiable, and it exists
because v0.3.4 shipped a crash that lint, 504 tests and green CI all passed.

---

## 6. Never do these

### Packaging

- **Never run the interpreter from inside `packaging/dist/`.** It writes `.pyc`
  into a sealed resource, invalidating the signature; macOS then re-validates 36k
  files before the child may exec, which looks exactly like a hung backend with an
  empty log. This was done by accident and cost real time. `PYTHONDONTWRITEBYTECODE=1`
  is set by the launcher precisely to prevent it.
- **Never install by hand.** `cp -R` produced a bundle 1,973 files short and reported
  success. `ditto` on APFS-over-USB exits 0, copies nothing, and *merges* into an
  existing bundle, leaving a broken seal. Use `INSTALL_APP=1 ./build_app.sh`, which
  compares file counts and re-verifies the signature.
- **Never bake an absolute path into the bundle.** `build_app.sh` uses `$APP` for
  the in-repo *build output*; using it where the code means "the install location"
  put this machine's volume path into 88 files. Shebangs are `#!/bin/sh`
  trampolines and `pyvenv.cfg` carries a **relative** `home` for this reason.
- **Never mutate the bundle after signing.** Signing is `build_app.sh:490`; every
  content edit must precede it. Rewriting a sealed file afterwards yields
  *"a sealed resource is missing or invalid"*.
- **Never set `PYTHONHOME`** when spawning an engine. It leaks into every
  grandchild and kills interpreters that are not ours.
- Sign only real Mach-O files. Never sign a symlink.

### Frontend

- **Never reintroduce base × ratio.** The old control multiplied a base by a ratio
  and produced sizes no model was trained on — 16:9 at base 1024 is 1024×576. The
  ratio is now a property of the resolution.
- **Never re-add a size filter.** One was built and removed: it cost a line of
  height to hide exactly one button, did nothing on two of three tabs, and silently
  resized the image when ticked. If you think a filter is warranted, ask first.
- **Never add a `useState` after a `useMemo`/function that reads it.** That TDZ
  ordering is what blanked v0.3.4's interface. Run
  `node tools/run-size-selector-render.mjs .` — it catches this class in seconds.
- Derived paths (`sys.base_prefix`, the true aspect ratio) must be **computed**,
  never stored next to the number they describe. `1536×640` is labelled `12:5`,
  not `21:9`.

### Data

- **Never write inside the bundle**, including a fallback path derived from
  `__file__`. That is how TAESD died: `taesd_mlx.py` looked for its weights at
  `<bundle>/backend/data/...`, missed, and fell back to the full VAE inside an
  `except` that printed one line to stderr. It was dead in every shipped build.
- Never persist a credential anywhere but `DATA_DIR`. `build_app.sh:505` audits the
  bundle for credential-shaped files and fails the build.
- Never hardcode a username, volume name or home directory — not even in a
  comment. Two such leaks shipped in 0.3.4.

---

## 7. Open issues — unresolved, with what is known

### 7.1 The launcher sometimes fails to start the backend

**Symptom:** the app launches, the child process exists, and it hangs forever —
0% CPU, ~3 s of CPU total, empty `backend.log`, nothing listening on 8001.

**What is ruled out:** the bundle. `build_app.sh`'s relocation rehearsal copies the
finished bundle elsewhere and asserts both interpreters start, `mlx` and `torch`
import, and a trampolined script executes. Running the launcher's exact argv by
hand from the same installed bundle binds in **6 seconds**. The child also has the
correct env vars.

**Leading hypothesis:** macOS TCC consent. `main.swift:113-121` already documents
that launching from an external volume gates the backend's first `open()` behind a
consent prompt, blocking at 0% CPU with an empty log — an identical symptom. The
launcher holds a `.userInitiated` activity token for App Nap, so that is handled.

**Next step:** reproduce after a clean reboot. If it survives a reboot, the next
place to look is why `launch.log` and `backend.log` stay 0 bytes — the launcher's
own retry/timeout logic is invisible, which is the real reason this is hard.

**Workaround:** start it by hand.
```bash
cd /Applications/DiffusionBear.app/Contents/Resources/backend
env DIFFUSIONBEAR_ASSET_DIR="$(cat ~/Library/Application\ Support/DiffusionBear/store_path)" \
    DIFFUSIONBEAR_DATA_DIR="$same" PYTHONDONTWRITEBYTECODE=1 \
    ../venv/bin/python -u -m uvicorn main:app --host 127.0.0.1 --port 8001 --no-access-log
```

### 7.2 GitHub Actions runners are not scheduling

Two consecutive runs were cancelled after exactly 15 minutes with **zero steps
executed**. Not a code failure. v0.3.5 shipped on locally-replicated equivalents:
514 tests with `mlx` blocked, oxlint, `vite build`, licences parse. The one genuine
CI failure along the way was real and is fixed: `test_taesd_path` imported a module
needing `mlx`, so the guard *errored* on Linux instead of running.

### 7.3 qwen-image-2.1 at 1024×1024 OOMs on a 16 GB M1

Metal bf16 VAE decode. Surfaced as a non-blocking advisory in the size picker.
Hard-blocking would encode one machine's RAM into shared data.

### 7.4 1024×688 plus a reference image is a dead end

The failing frame is the *reference* VAE encode, not the output size. Do not add
sizes near it and do not "fix" it by resizing — the reference is pre-scaled to the
output dimensions.

---

## 8. Dead and vestigial code

| what | status |
|---|---|
`app_settings.py:39` `__file__/data` fallback | inert (launcher always sets the env) but points **inside the bundle**. Left as a last resort; consider failing loudly. |
`hf_service.py:53`, `civitai_service.py:264` `__file__` fallbacks | exception-branch only, effectively safe |
`generator.py:1188` and `:2223` `max_pixels` plumbing | caps were removed 2026-09-22; still honoured if set |
| `FLOOR`, `SIZE_PRESETS`, `LEGACY_PRESETS`, `ECOSYSTEMS`, `ADVISORY_PIXELS`, `isModern`, `ratioValue` in `sizePresets.js` | exported and consumed by the **tests**, not the component. Keep them -- `SIZE_PRESETS` is the documented extension point for adding a resolution |
`frontend/src/i18n/lang/_parts/*.json` → `es.js` etc. | **generated**, edit the `_parts` and run `npm run i18n:build` |
`frontend/src/version.js` | **generated** from `backend/app_version.py` by the build |
`packaging/vendor/mflux-src` | vendored upstream, kept public deliberately |
`frontend/tools/.render-smoke/` | transient, gitignored |

Fifteen `params.size.*` translation keys (`baseLabel`, `baseTitle`, `ratioLabel`,
`ratioTitle`, `customShort`, `customTitle`, `backToGrid`, `backToGridTitle`,
`budgetWarning`, `budgetHint`, `lockRatioTitle`, `unlockRatioTitle`, `lockRatioHint`,
`customOption`, `shapePreviewTitle`) are residue of the old base x ratio control and
are referenced nowhere. `test_i18n.py` has **no unused-key check**, which is why they
survived three refactors. Adding one is cheap and would have caught them.

---

## 9. Testing architecture

514 tests, and the distribution matters more than the count.

| mechanism | why |
|---|---|
**node-executes-the-component** (`test_size_selector.py`) | tests shipped code, not a copy |
**vite SSR render** (`test_size_selector_render.py`) | catches throw-on-render, the class that shipped |
**AST source guards** (`test_taesd_path.py`) | runs on Linux CI where `mlx` is absent |
**substring-vs-AST discipline** | a plain grep flags the docstring explaining the bug; both docstrings and code must be distinguished |
**dangerous-pattern allowlist** (`test_repo_security.py`) | `exec` is allow-listed per (file, label) with a count, and the test fails if an allowance goes unused — a dead allowance is a bug |
**i18n exact-zero bound** (`test_i18n.py`) | was "no more than 23 missing"; now exactly 0 |

### The render harness is the important one

Lint does not do control-flow analysis across a component body. The backend tests
never import React. CI never evaluates a component. So a temporal dead zone passed
all three and shipped a blank interface.

A **static** TDZ scanner was tried first and **rejected**: 63 false positives,
because it cannot tell a function parameter from a declaration in another scope.
Rendering has no such blind spot.

```bash
cd frontend && node tools/run-size-selector-render.mjs .
# SUMMARY 9/9
```

It is verified against the real pre-fix revision: pointing it at the v0.3.4
`ModelInstaller` reproduces `Cannot access 'showBar' before initialization`.

---

## 10. Git and release

- Private `origin` = `MLX-Diffusion-DEV`, public `upstream` = `DiffusionBear`.
- Public `main` is force-pushed from a filtered export.
- Release order that must not change: **bump version → build → install to
  `/Applications` → owner tests → CI green → tag → push public → release.**
- Version lives in three places and `test_version_consistency.py` enforces it:
  `backend/app_version.py`, `frontend/package.json`, `frontend/src/version.js`.
- The build regenerates `version.js` and the language overlays; commit the result.
- Artifacts: app zip, source zip, source tarball, `ALL_SHA256SUMS`. Verify sizes
  byte-for-byte against local before publishing — the install step prints both
  counts.

---

## 11. Machine-specific facts

Not portable, but they will save you a day.

- Internal boot volume has ~14 GB free. **The model store must stay on Externe.**
- The store is relocated via
  `~/Library/Application Support/DiffusionBear/store_path` →
  `/Volumes/Externe/IA/DiffusionBear/backend/data`. 99 GB was copied there once;
  never symlink it, macOS TCC treats the link target as a different volume.
- `/Volumes/Externe` is an **external USB volume**. Copying a 36k-file bundle off it
  is slow and is where the `ditto` warning in `build_app.sh` came from.
- `ditto` and `cp -R` both misbehave on this filesystem. The build's install step
  exists because of it.
- `AGENTS.md` in the legacy checkout is a **historical** copy; the maintained one is
  in the private DEV repo. Read the DEV one.

---

## 12. First week checklist

1. Read `packaging/build_app.sh` end to end. It encodes more hard-won constraint
   than any other file here.
2. Run `cd backend && "/Volumes/EXterne/IA/MLX-DIFFUSION OpenCode/venv/bin/python" -m unittest discover -q`.
   Understand why each of the 514 exists before adding more.
3. Reproduce the launcher hang after a clean reboot (§7.1). Highest-value open bug.
4. Read the comments in `backend/generator.py` around the FLUX.2 model-repo mapping
   and the krea2 step presets — they record measurements, not opinions.
5. Before any change to the size picker or the engines, run the render harness and
   the full backend suite. Both are fast; both have caught real regressions.
6. Do not publish. The owner tests the compiled app in `/Applications` first.
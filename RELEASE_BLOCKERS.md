# Release Readiness — Bug Tracker
Generated 2026-10-10 · post QIT repair session · target: ship to /Applications

Status legend: `[ ]` open · `[~]` in progress · `[x]` fixed & verified

---

## BLOCKERS — must fix before any build

### B1 · CRITICAL · `LSMinimumSystemVersion` lies by 11 macOS majors
`packaging/build_app.sh:385` writes `15.0`. The payload's own load commands say otherwise:

| binary | minos |
|---|---|
| `mlx/lib/libmlx.dylib` | **26.2** |
| `mlx/core.cpython-310-darwin.so` | **26.2** |
| `torch/lib/libtorch.dylib` | 14.0 |

Verified in `packaging/build/venv`, `packaging/build/sdxl/venv` and the installed
`/Applications/DiffusionBear.app`.

**Failure mode:** on macOS 15.0–26.1 the Swift shell launches (its own minos is 15.0),
spawns the backend, and dyld kills the child. The user sees a ~5-minute hang then
*"The backend exited during startup"* — not a version error.

**Fix:** derive the floor — scan `*.so`/`*.dylib` under `Resources`, take max `minos`,
write that. Hardcode `26.2` if deriving is too invasive.
*Verified working:*
```
find "$APP/Contents/Resources" \( -name '*.so' -o -name '*.dylib' \) -print0 \
 | xargs -0 -n1 otool -l | awk '/LC_BUILD_VERSION/{f=1} f&&/minos/{print $2;f=0}' \
 | sort -V | tail -1
```

### B2 · CRITICAL · The mflux tested tonight is not the mflux that ships
`build_runtime_venv.sh:73-82` installs from `packaging/vendor/mflux-src`.
Dev `venv/` is a **Sep 22** install; vendor is **Oct 9**. Six files differ.

| file | vendor (ships) | dev venv (tested) |
|---|---|---|
| `flux2/variants/txt2img/flux2_klein.py` | `use_kv_cache` kwarg + full KV-cache ref path; M1/M2 = 2 eager steps then `mx.compile(shapeless=True)` | **no `use_kv_cache` kwarg**; plain loop; **never compiled on M1/M2** |
| `flux2/variants/edit/flux2_klein_edit.py` | warmup + shapeless compile | plain, never compiled |
| `krea2/weights/krea2_weight_definition.py` | `group_size=32`, `skip_quantization=False` (TE q4 g32) | `skip_quantization=True` (**bf16 TE**) |
| `krea2/variants/txt2img/krea2.py` | warmup + shapeless | plain |
| `z_image/variants/z_image.py` | warmup + shapeless | plain |
| `common/config/model_config.py` | no turbo entry | has turbo entry |

**Consequences:**
- `generator.py:2892-2893` passes `use_kv_cache=True` for flux2-klein + reference image.
  Against the venv that is a **`TypeError`** (param doesn't exist); against vendor it is
  a reference-token-extraction + cached-steps-2..N path **that has never run in dev** —
  and it is the flow AGENTS.md records as OOM-prone at 16 GB.
- Every FLUX / Krea 2 / Z-Image timing measured tonight is from the wrong build.
  The 48.5 s FLUX 4B figure in particular: vendor would compile + use KV cache.
- `generator.py:2262-2263` comment describes the vendor (g32); the machine that wrote it
  ran the venv (bf16 TE). AGENTS.md's `te-q4-g64` cache-key scheme is g64 — wrong for vendor.

**Fix (pick one):**
- (a) Re-verify FLUX/Krea2/Z-Image against `packaging/build/venv`, not `venv/`.
- (b) Revert vendor's five files to match the venv, re-pin digest in
  `test_repo_security.py:233-238`.

**NEEDS USER DECISION** — see bottom.

### B3 · CRITICAL · Qwen text-encoder selector returns `None` → opaque crash
`qwen_engine.py:63` — `return orig_selector(root_path) if orig_selector else None`.
The stock `text_encoder` ComponentDefinition has **no** `variant_selector` (it is `None`),
so this always returns `None`, and `weight_loader.py:123` dereferences it.

Layouts verified against the real mflux loader:

| layout | on disk? | result |
|---|---|---|
| A `model.safetensors`, no index | Sawfwair Turbo | works |
| B `model.safetensors` + index naming it | `mlx-community/Qwen-Image-2.1-MLX-4bit` | works |
| C genuinely sharded + index | common upstream layout | **AttributeError** |
| D `0.safetensors`, no index | `backend/data/models/Qwen-Image-2.1-mflux-q4` | **AttributeError** |

**This is the exact error the user saw tonight.** We fixed case B (Tonight's model);
cases C/D remain latent.

**Fix:** return the base component, never `None`:
```python
def _te_variant_selector(root_path, _base=c, _orig_selector=c.variant_selector):
    te = root_path / _base.hf_subdir
    single, index = te / "model.safetensors", te / "model.safetensors.index.json"
    if single.is_file() and (not index.is_file() or _is_single_file_index(index)):
        return replace(_base, loading_mode="single")   # inherit precision/skip_quantization
    return _orig_selector(root_path) if _orig_selector else _base
```
`dataclasses.replace` matters: the hand-built `ComponentDefinition` at `:57-62` silently
drops `precision`, `skip_quantization`, `key_transform`, `weight_files`.

Add a regression test: `selector(root) is not None` for all five layouts.

### B4 · HIGH · Hand-edit to a gitignored dependency, invisible to guards
`venv/.../mflux/models/common/config/model_config.py:801-815` adds a
`qwen-image-2.1-turbo` entry. **Will not ship** — vendor ends at line 801.

Currently *inert*: `Qwen21Initializer` uses `model_path` when truthy (always), never
`model_config.model_name`; `AVAILABLE_MODELS` is only read by the mflux CLI.
But it proves the trees disagree, and `test_repo_security.py` only pins `packaging/vendor/`,
so a `venv/` edit is completely unguarded.

**Fix:** delete it. The repo id is already app-side data at `generator.py:527`.

### B5 · HIGH · `_generate_sdxl` cancels the Qwen watchdog but never re-arms it
`generator.py:3575` cancels; the `finally` at `:3765` only calls `_arm_sdxl_watchdog()`.
Run SDXL while a warm Qwen daemon exists → the ~10.5 GB daemon has **no idle timer** and
survives until the next Qwen job or app quit. On a 16 GB machine that is the documented
OOM-adjacent state.

**Fix:** add `_arm_qwen_watchdog()` to that `finally` (it self-guards on a dead process).

### B6 · HIGH · Engine errors still reach the user as a bare Python exception
`generator.py:3891` does `raise RuntimeError(str(result["error"]))` — verbatim, no model,
no path, no component. `_qwen_stderr_tail` **holds the full traceback and is discarded**
on this path (only surfaced on engine-death / no-result / timeout).

**Fix (two lines, high leverage):**
1. B3's `return _base` converts the opaque AttributeError into mflux's own
   `FileNotFoundError` naming the missing path.
2. At `generator.py:3891`, append `''.join(_qwen_stderr_tail)[-2000:]` to the message.

### B7 · HIGH · Unconditional debug prints on hot paths
- `generator.py:2177` — `[DEBUG] _get_pipeline called…` **every** generation
- `generator.py:2286-2301` — dead branch, still prints
- `generator.py:38/41/63/66/97`, `qwen_engine.py:21/69/71` — every process start
- `qwen_engine.py:107-123` — verification block on every cold load

**`qwen_engine.py` prints go to stdout, which is the JSON-lines transport.** They survive
only because `generator.py:3870` swallows `JSONDecodeError`. Any parser tightening turns
these into a protocol desync. **All engine stdout must go through `emit()`.**

### B8 · HIGH · Dead, broken diffusers engine ships in the bundle
`backend/qwen_diffusers_engine.py` (untracked, 184 lines) is copied by
`build_app.sh:109-112` (which never consults `git ls-files`). It is unreachable
(`_generate_qwen_diffusers_subprocess` has no callers) **and** non-functional:
`phase_cb` undefined at `:115`, tuple-unpacked as a pipeline at `:118`,
`with mx.eval():` as a context manager at `:133`, `pipe.to("mps")` at `:49`,
writes into the sealed bundle at `:148`.

**Fix:** delete it and `generator.py:3974-4060`.

### B9 · HIGH · Duplicate divergent monkey-patch
`generator.py:58-97` patches `Qwen21WeightDefinition.get_components`. That branch
(`:2286-2310`) is **unreachable** — both Qwen models return from `generate()` at `:2701`.
It also lacks tonight's case-B fix. Two copies of a load-bearing patch, one stale, no test
asserting they agree. That is exactly how the fix landed in one and not the other.

**Fix:** delete `generator.py:58-97` and `:2286-2310`. Keep the patch only in `qwen_engine.py`.

---

## HIGH

- [ ] **H1** · `backend/data/` not gitignored (only individual children are). `git status`
  already shows `?? backend/data/queue_recovery.json.invalid.*`. Add `backend/data/` and assert it.
- [ ] **H2** · `test_repo_security` is **RED**: `GenerationParams.jsx:47` `clipboard read`.
  `build_app.sh` runs no tests, so this would ship. Add to `_ALLOWED_DANGEROUS`, and make
  `build_app.sh` refuse to build while the suite is red.
- [ ] **H3** · `packaging/build/venv` is **stale** — still Sep-29 mflux. Always
  `rm -rf packaging/build` before building; `build_runtime_venv.sh` has no clean step.
- [ ] **H4** · Ad-hoc signing only. `codesign -dv` → `Signature=adhoc`, `TeamIdentifier=not set`.
  Runs on this Mac (no quarantine). **The 505 MB zip will be quarantined and refused
  anywhere else.** Real distribution needs a Developer ID + `notarytool`.
- [ ] **H5** · `/halloween` + `/halloween-report` (`main.py:191-204`) read
  `Path("/tmp/halloween_comparison_report.html")`. Not under `/api/`, so **no loopback
  check and no token**; the launcher never sets `DIFFUSIONBEAR_API_TOKEN`, so every
  `/api/*` route is unauthenticated in the shipped app. `/tmp` is 1777 — any local process
  can plant that file and have it served as `text/html` from the app's own origin, which
  holds `window.webkit.messageHandlers.native`. **Delete these debug routes.**
- [ ] **H6** · `QWEN_ENGINE_DIAG` block (`qwen_engine.py:397-429`) keeps `_emb`, `_msk`,
  `_lat`, `_tok_emb` and a **full transformer forward output** alive for the entire
  real generation — on the 16 GB machine AGENTS.md says dies on allocations. Off by
  default; OOM generator when on. Move to its own function scope or delete.

---

## MEDIUM

- [ ] **M1** · `generator.py:1391-1398` — engine inference is a hardcoded 6-tuple +
  `repo.startswith("mlx-community/")`. **No MODELS entry has `engine == "mflux"`**, so the
  first clause is dead. `Sawfwair/…` survives only via the tuple. Add a real `engine` field.
- [ ] **M2** · `generator.py:1411-1413` — HF fallback ignores `_model_path_has_incomplete`,
  so an interrupted download with one complete shard is accepted as the local path.
- [ ] **M3** · `generator.py:2201` — flux2-klein-**4b edit variant** still passes
  `model_path=local_arg` with no `or info["repo"]`. Same 22 GB bug as txt2img, still live.
- [ ] **M4** · `qwen_engine.py:28/42` — `_orig_get_components` is a module global looked up
  at call time. Rebinding it (exec/runpy/copy-paste) → mutual recursion → `RecursionError`.
  Bind as a default arg. Also delete the duplicate assignment at `:42`.
- [ ] **M5** · `qwen_engine.py:50` — `orig_selector` is late-bound in a loop. Correct today
  (one match), fragile. Add `_orig_selector=orig_selector` default + `break`.
- [ ] **M6** · `routers/loras.py:198` — destructive `force` rmtree of a shared asset store
  via a **DELETE query parameter**. A stray `?force=true` permanently deletes a model.
- [ ] **M7** · `build_app.sh:41-68` mutates the working tree every run (rewrites
  `frontend/src/version.js` + `package.json`). `git checkout --` them after building.
- [ ] **M8** · `build_runtime_venv.sh:28/41` and `build_sdxl_venv.sh:18/31` reuse
  `/tmp/pbs*.tar.gz` with no checksum. A truncated file fails much later.
- [ ] **M9** · Rehearsal (`build_app.sh:591-640`) proves relocation but never exercises the
  port ladder, `reapStrayBackends()`, or the TCC probe. First launch stays untested.

---

## LOW / cleanup

- [ ] **L1** · Untracked junk to delete or gitignore: `SUPERPOWERS_GUIDE.html`,
  `packaging/DiffusionBear-0.3.7-arm64.zip` (505 MB build artifact),
  `backend/data/`, `backend/qwen_diffusers_engine.py`.
- [ ] **L2** · Stray dev backend on :8001 (PID 14598) — kill before any first-launch test.
- [ ] **L3** · `_is_single_file_index` parses index.json twice per cold load (uncached).
  Delete the second call with B7's debug block.
- [ ] **L4** · `_fingerprint` (`qwen_engine.py:216-222`) does numpy on a lazy graph.
  Tested safe today (`np.asarray` not `np.array`), latent SIGABRT hazard per AGENTS.md.
- [ ] **L5** · 4 pre-existing `WiredBudgetTests` failures — unrelated, but red.

---

## Verified-clean tonight

- `_generate_sdxl` indentation repair — AST-verified, body correctly scoped, `finally` runs.
- Qwen dispatch removal — both models route to `_generate_qwen_subprocess`; no frontend
  or status consumer expects the diffusers path.
- `resolve_local_model_path` HF fallback does **not** shadow an explicit registered path
  (`app_settings.model_local_path` returns `""` for a missing dir).
- `uninstall_model` and `model_disk_usage` agree with the new resolver (both delete/report
  the HF snapshot). No orphaned-after-uninstall case.
- `resolve_local_model_path` HF scan costs ~0.2 ms — `rglob` short-circuits on first hit.
- `test_i18n` 42/42, `test_model_download_path` 19/19, `test_progress_animation`,
  `test_dimensions`, `test_sdxl_convert`, `test_np_array_eval_guard` — all pass.
- Swift launcher compiles clean; Vite build clean; bundle audit finds no credential-shaped
  files, no `.safetensors`, no >50 MB strays; `codesign --verify --deep --strict` passes.

---

## Resolutions since the review

Re-verified by live generation on 2026-10-10, after syncing the dev venv to the vendored
mflux (so the tested tree is the shipped tree).

- **B1 FIXED** — `build_app.sh` derives `LSMinimumSystemVersion` from the payload's own
  `minos` (26.2). *(Rebuild still required for the shipped plist to change.)*
- **B2 FIXED (option a, per user)** — dev venv synced to `packaging/vendor/mflux-src`;
  `diff -rq` over the source trees is now clean. Consequence measured, not assumed:
  - FLUX 4B **47.8s → 42.9s** at 512x768/4 steps (vendor warmup+compile path).
  - `use_kv_cache=` is now a real kwarg; previously img2img paths raised `TypeError`.
  - krea2 text encoder is now quantised (`skip_quantization=False`), the OOM fix.
- **B2 follow-on FIXED** — krea2 crashed at denoise step 3/8 with
  `[Primitive::output_shapes] Split cannot infer output shapes.` Cause: the vendored
  `_predict` swapped to `mx.compile(..., shapeless=True)` after 2 warmup steps, and
  shapeless compilation cannot infer shapes for Krea 2's `Split`. Switched to
  shape-specialised `mx.compile` (keeps the fusion, drops the hazard). Verified: krea2
  512x512/4 steps → 174.3s, 65,615 distinct colours. Vendored-tree digest re-pinned.
- **B3 FIXED** — selector returns the base component, never `None`; 9 layout tests cover
  single-no-index, single-with-index, and sharded.
- **B4, B7, B8, B9 FIXED** — hand-edit, debug prints, dead diffusers engine, duplicate patch.
- **B5 FIXED** — `_arm_qwen_watchdog()` in the `_generate_sdxl` `finally`.
- **M1, M2, M3, H1, H2, H5, H6, L1 FIXED.**

### FLUX edit root cause — found and fixed properly

Disabling compile was a **workaround, not a fix**, and it cost the speedup. The real
cause was in our own code, not mflux:

`generator.py`'s rope memo cache keys on `ids.reshape(-1).tolist()`. `.tolist()` forces
an evaluation, and the FLUX *edit* reference-image graph reaches the rope embedder while
`mx.compile` is tracing. MLX refuses that: `ValueError: [eval] Attempting to eval an array
during function transformations like compile or vmap is not allowed.`

Proof, not inference: with `MLX_DISABLE_ROPE_CACHE=1` and compile re-enabled, FLUX edit
completed. Fix: the cache catches that specific `ValueError` and falls through to the
uncached call. Nothing is lost — a compiled graph traces once and reuses its own fused
result, so the memo would never have been hit on those steps anyway.

FLUX edit compiles again. Guarded by `backend/test_rope_memo_cache.py` (4 tests), which
**reproduces the exact production error** when the fix is reverted.

### Live re-verification (every engine in the registry, one backend)

| engine | config | gen | result |
|---|---|---|---|
| flux2-klein-4b | 512x768 / 4 | 49.2s | ok, 72,520 colours |
| flux2-klein-9b | 512x768 / 4 | 136.4s | ok, 69,534 colours |
| flux2-klein-4b **edit** | 512x768 / 4 + ref | 88.4s | ok, 781 KB, 211k colours |
| qwen-image-2.1 | 512x512 / 2 | 30.0s | ok |
| qwen-image-2.1-turbo | 512x512 / 4 | 343.0s | ok, 393 KB |
| krea2-turbo | 512x512 / 4 | 174.3s | ok, 65,615 colours |
| z-image-turbo | 512x768 / 8 | 113.1s | ok, 83,231 colours |

Suite: **559 tests, 3 failures** — the pre-existing wired-budget trio only.

### `shapeless=True` removed from every model we ship

The vendored warmup-swap optimisation turned out to have **two** distinct failure modes,
both only visible by actually running the models:

1. **Shape inference** — `mx.compile(shapeless=True)` cannot infer output shapes for a
   `Split` whose axis length is only known at runtime. Krea 2 died at denoise step 3/8.
2. **`mx.eval` inside a trace** — the FLUX *edit* reference-image graph reaches an
   `mx.eval` while the trace is open. MLX refuses it: *"Attempting to eval an array
   during function transformations like compile or vmap is not allowed."* Raised at
   denoise step 3/4. **This one is independent of `shapeless`** — shape-specialising
   does not help, which is why FLUX edit is now simply not compiled.

Resolution: `flux2_klein.py`, `flux2_klein_edit.py`, `krea2.py` and `z_image.py` no longer
compile shapeless. The two remaining `shapeless` uses are upstream helpers we did not fork
(`flow_match_euler_discrete_scheduler.py`, `gpt_oss`) — untouched.

Cost: FLUX 4B 42.9s → 49.2s on the first pass. Once the rope-cache bug below was fixed at
the source, FLUX edit kept its compile.

---

## Open risks carried into the build

1. **B1 not effective until rebuilt** — the installed/staged plist still says `15.0`.
2. **QIT Metal timeout** — one `512x768`/4-step run hit a GPU CommandBuffer timeout;
   subsequent runs of the same config succeeded. Intermittent, not root-caused.
3. **`QWEN_IDLE_KILL_S=60`** in settings vs the `300` default — verify a long active Qwen
   job is not cancelled mid-generation.
4. H4 (ad-hoc signing), M6 (destructive LoRA `force` delete), M7 (`build_app.sh` mutates
   version files), M8 (unchecked PBS download), M9 (first-launch port ladder / TCC) remain.

---

## `reference_strength` is dead on FLUX Edit (found while documenting it)

The API accepts `reference_strength`, records it in the job and the sidecar, and it has
**no effect** on FLUX. Proven empirically: 0.25 and 0.85 produced **byte-identical PNGs**
(md5 match).

Cause: any FLUX reference switches `variant="edit"` (`generator.py:2723`), and that call
site never passes `image_strength`. So mflux receives `None`.

**Do not "fix" this by threading the value through.** FLUX Edit is not classic img2img: it
starts from `create_noise()` (pure noise) and feeds the reference as concatenated
conditioning tokens. `image_strength` only feeds `Config.init_time_step`, so on this path
it cannot make the result more faithful — it would just skip the first
`int(steps * strength)` denoise steps. Passing the app's 0.6 default would skip 60% of the
steps and degrade every edit. This was verified in the code before being reverted.

Consequence for users: on FLUX, fidelity to the reference is controlled by the **prompt
wording and guidance**, not a slider. The other reference-capable models (Qwen 2.1, QIT,
Krea2, Z-Image) do use a real strength path.

Decision needed: either reject `reference_strength` for FLUX with a 400 so the UI stops
implying it works, or document it as FLUX-only no-op. Left as-is (silent no-op) for now.

---

## Decisions taken by user

1. **B2** — ship the vendored mflux (option a).
2. **B1** — OS floor `26.2` or newer.
3. **H4** — Apple Silicon only.

"""DEV-machine health check: every registered model, end to end.

Run on the dev machine (repo venv):

    cd backend && ../venv/bin/python check_dev_models.py

For each model in generator.MODELS this reports:
  * where its weights resolve (settings override, default models/<id>, or an
    HF-cache snapshot of the official repo),
  * validate_local_model_dir() verdict (layout mismatch is the bug class that
    shipped: a diffusers pack auto-bound to an mflux model),
  * whether an embedded manifest exists (so the repair/reinstall route works),
  * for mflux engines, that the model family module imports in this venv.

Exit code is non-zero if any INSTALLED model fails validation. Models that are
simply not downloaded are reported as such, not as failures.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import app_settings  # noqa: E402
import generator  # noqa: E402
import model_integrity  # noqa: E402


def _hf_cache_candidates(repo: str):
    from huggingface_hub import scan_cache_dir

    wanted = f"models--{repo.replace('/', '--')}"
    try:
        for r in scan_cache_dir().repos:
            if r.repo_id.replace("/", "--") == wanted[7:]:
                yield r.repo_path / "snapshots" / r.revision
    except Exception:
        return


def _family_module_ok(minfo) -> str:
    import importlib.util

    if minfo.get("engine") not in (None, "mflux"):
        return "n/a (subprocess engine)"
    family = minfo.get("family") or (minfo.get("id") or "").split("-")[0]
    # Registry ids whose module name differs from the id prefix.
    family = {"flux1": "flux", "z": "z_image"}.get(family, family)
    if importlib.util.find_spec(f"mflux.models.{family}") is None:
        return f"MISSING mflux.models.{family}"
    return f"ok (mflux.models.{family})"


def main() -> int:
    manifests = model_integrity.load_manifests().get("models") or {}
    rows = []
    bad_installed = 0
    for model_id, minfo in sorted(generator.MODELS.items()):
        if minfo.get("is_utility"):
            kind = "utility"
        else:
            kind = minfo.get("engine") or "mflux"
        repo = generator.model_download_repo(model_id, minfo) or minfo.get("repo") or ""
        # Resolution order mirrors the generation path.
        candidates = []
        override = app_settings.configured_model_local_path(model_id)
        if override:
            candidates.append(("override", Path(override)))
        default = app_settings.ASSET_DIR / "models" / model_id
        candidates.append(("default", default))
        for snap in _hf_cache_candidates(repo):
            candidates.append(("hf-cache", snap))
        installed = None
        verdict = "not installed"
        for where, path in candidates:
            ok, why = generator.validate_local_model_dir(model_id, path)
            if ok:
                installed = (where, path)
                verdict = "ok"
                break
            if path.is_dir():
                verdict = f"INSTALLED-BUT-INVALID ({why})"
        manifest = "manifest" if model_id in manifests else "NO-MANIFEST"
        fam = _family_module_ok(minfo)
        if verdict != "ok" and verdict != "not installed":
            bad_installed += 1
        elif fam.startswith("MISSING"):
            bad_installed += 1
        rows.append(
            f"{model_id:28s} {kind:8s} {verdict:60s} {manifest:12s} {fam}  repo={repo}"
        )
        if installed:
            rows.append(f"{'':28s} -> {installed[0]}: {installed[1]}")
    print("\n".join(rows))
    total = len(generator.MODELS)
    n_installed = sum(1 for m in rows if "-> " in m)
    print(f"\n{total} registered models, {n_installed} with installed weights on this machine")
    if bad_installed:
        print(f"FAIL: {bad_installed} installed model(s) failed validation (layout mismatch?)")
        return 1
    print("OK: every installed model validates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

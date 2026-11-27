# Handoff implémentation — DreamLite (instantané) + Anima Turbo q4 + ControlNet Qwen-Image-2.1-Turbo

**Date :** 2026-11-19 · **De :** session veille/rebase · **Pour :** agent d'implémentation
**Périmètre : items 1/2/3 uniquement.** La vidéo LTX-2.5 (item 4) est volontairement hors scope — ses tests seront faits après.

## 0. Contexte — ce qui vient d'être fait (commits sur `main`)

- `aaa8141` — **rebase du mflux vendored** : `v0.23.0` → mflux-community `main@014252f` (tag v0.24.0). 18 conflits résolus vers nos patches, arbre `flux3/` gardé nôtre. Suite backend **2255 tests OK** (vérifiée avant ET après `pip install` dans le venv). Pin `mflux==0.24.0`, digest `test_repo_security` re-pinné (1206 fichiers).
- `489f337` — docs de veille + résultats de bench.
- Nouveaux modèles disponibles dans le mflux rebasé (classes importables, testé) :
  - `mflux.models.dreamlite.variants.txt2img.dreamlite.DreamLite` — `DreamLite(quantize=8, model_config=ModelConfig.dreamlite_mobile()|ModelConfig.dreamlite())`, repos `carlofkl/DreamLite-mobile` / `carlofkl/DreamLite-base` (5.07 GB, **CC BY-NC 4.0**). `generate_image(seed, prompt, num_inference_steps, height, width, guidance, negative_prompt)`.
  - `mflux.models.anima.variants.txt2img.anima.Anima` — `Anima(quantize=None, model_path="mflux-community/anima-turbo-mflux-q4", model_config=ModelConfig.anima_turbo(), lora_paths=..., lora_scales=..., bake_lora=False)` (3.11 GB, **licence NC circlestone-labs**). LoRA mapping upstream : `anima_lora_mapping.py`. `generate_image(..., guidance, negative_prompt, scheduler)`.
  - `mflux.models.qwen21.variants.controlnet.qwen_image_21_controlnet.QwenImage21Controlnet` — usage de référence dans `packaging/vendor/mflux-src/mflux/models/qwen21/cli/qwen21_controlnet_generate.py` (`controlnet_image_path`, `controlnet_strength`, inpaint via `image_path`+`mask_image`, guidance 1.0, LoRA via `lora_init_kwargs_from_args`).

## 1. Où sont les packages / artefacts

| Quoi | Où |
|---|---|
| mflux vendored (source de vérité) | `packaging/vendor/mflux-src/` (layout plat `mflux/`) — voir `VENDORED_FROM.md` pour la base upstream et la liste des patches à préserver |
| mflux installé (ce que le backend importe) | `venv/lib/python3.10/site-packages/mflux` — installé par `./venv/bin/python -m pip install --no-deps --no-build-isolation --force-reinstall ./packaging/vendor/mflux-src` (version 0.24.0). NE PAS polluer `packaging/vendor/mflux-src` avec `build/`/`egg-info` (le digest test casse) — `packaging/build_runtime_venv.sh` construit depuis une copie jetable, suis ce pattern |
| Cache modèles HF (téléchargements mflux/HF) | `~/.cache/huggingface/hub/models--<org>--<repo>/` — DreamLite-mobile/base et anima-turbo-q4 déjà téléchargés et chauds pendant la session du 19/11 ; **les caches Ideogram ont été supprimés (14 GB libérés, df 27→37 Gi free)** |
| Store de l'app (galerie, settings, store modèles) | `~/Library/Application Support/DiffusionBear` (symlink — ne pas "corriger") |
| Bench reproductible | `scripts/bench_mflux_new_models.py` (cas: `dreamlite-mobile`, `dreamlite-base`, `anima-turbo`) — résultats dans `scripts/bench-results.jsonl` |
| Résultats de référence (M1 16 GB, seed 42) | `docs/veille/etude-rebase-mflux-2026-11-19.md` |
| Rapport de veille vivant | `docs/veille/etat-de-lart.md` (cron lundi 09:00) |

## 2. Comment J'AI testé (méthodologie à reproduire)

1. **Imports** : `PYTHONPATH=packaging/vendor/mflux-src ./venv/bin/python -c "import mflux; ..."` puis classes des nouveaux modèles → OK.
2. **Suite backend avec le nouvel arbre** : `cd backend && PYTHONPATH=../packaging/vendor/mflux-src ../venv/bin/python -m unittest discover -q` → **2255 tests OK (72 skipped)** ; puis après install venv, idem sans PYTHONPATH → OK (68 s).
3. **Bench génération** : un process par modèle (isolation mémoire 16 GB), `mx.set_wired_limit(int(0.68*16*2**30))`, seed 42, prompt fixe, 512×768, chrono `perf_counter` autour de `generate_image`, PNG validé via PIL (`getcolors` — **jamais `np.array()` sur un `mx.array` non évalué**, abort process garanti, cf. AGENTS.md).
4. **Résultats obtenus** : DreamLite-mobile 4 steps = **8.25 s** @512×768 (17.7 s @1024²) ; DreamLite-base 8 steps = 18.5 s, 28 steps = 66.4 s ; Anima Turbo q4 4 steps = 32.9 s, 10 steps = 58.0 s. Images valides (couleurs distinctes > 60k).

## 3. Ce qu'il y a à faire (items 1/2/3)

### 1) DreamLite-mobile en mode « Instantané »
- `backend/generator.py` : entrée `MODELS["dreamlite-mobile"]` (engine `"mflux"`, label, `repo: "carlofkl/DreamLite-mobile"`, `default_steps: 4`, `default_width/height: 512/768`, guidance off — mobile est distillé sans guidance) + chaîne `_get_pipeline` (miroir de `z-image-turbo` ~ligne 4678) : `DreamLite(quantize=quantization, model_path=local_arg or info["repo"], model_config=ModelConfig.dreamlite_mobile())`.
- `backend/model_families.py` : nouvelle famille `dreamlite` (pas de LoRA au début).
- `backend/model_manifests.json` : entrée repo (miroir Z-Image).
- Frontend : `frontend/src/components/settings/ModelsSection.jsx` / `GenerationParams.jsx` (le picker lit les modèles du backend), `frontend/src/data/licences.js` (ajouter DreamLite, **CC BY-NC 4.0**, url `https://huggingface.co/carlofkl/DreamLite-mobile`).
- i18n : clés dans `frontend/src/i18n/lang/_parts/*.json` (**FR = référence, 9 langues obligatoires**), puis `cd frontend && npm run i18n:build && npm run lint && npm run build` ; `test_i18n.py` doit passer.
- **Synchro licences obligatoire** (règle repo) : `frontend/src/data/licences.js` + `README.md` + `site/licenses.html` doivent lister le même modèle/URL/licence.
- Bench : nouveau run réel via l'app (sidecar JSON dans `backend/data/generated/`) et ligne ajoutée au tableau **Benchmarks** du README (jamais de chiffre inventé — sidecars uniquement).

### 2) Anima Turbo q4 (moteur anime léger, 3.1 GB)
- Famille `anima` dans `model_families.py` (LoRA oui — `LORA_FAMILIES`, mapping upstream `anima_lora_mapping.py` ; vérifier `backend/lora_compat.py`/`lora_convert.py` pour le mapping de clés).
- `MODELS["anima-turbo"]` : `repo: "mflux-community/anima-turbo-mflux-q4"`, steps 4 (rapide) et 10 (défaut upstream), guidance 1.0 (turbo = distillé), quantize=None (repo déjà q4).
- `_get_pipeline` : `Anima(quantize=None, model_path=..., model_config=ModelConfig.anima_turbo(), lora_paths=[...], lora_scales=..., bake_lora=False)`.
- Licence **NC (circlestone-labs-non-commercial-license)** → synchro licences 3 endroits + `licences.js`.
- Tokenizers : Anima charge ses tokenizers depuis `Qwen/Qwen3-0.6B-Base` et `google/t5-v1_1-xxl` (fichiers texte seuls) — vérifier que le chemin de téléchargement fonctionne via le store app.

### 3) ControlNet + LoRA pour Qwen-Image-2.1-Turbo
- Upstream l'a livré dans le rebase (#859) : classe `QwenImage21Controlnet` + CLI `qwen21_controlnet_generate.py` (voir §0). LoRA ComfyUI turbo déjà supportés upstream (#965).
- Côté app : exposer une image de contrôle (upload galerie ou chemin) + `controlnet_strength` dans Generate → Advanced pour `qwen-image-2.1-turbo`, et brancher `QwenImage21Controlnet` dans `_get_pipeline`/le moteur qwen (`backend/qwen_engine.py`, chaîne `model_id == "qwen-image-2.1-turbo"` ~ligne 5348).
- Inpaint (image_path + mask_image) possible via la même classe — à scoping avec l'utilisateur si doute.

## 4. Règles et pièges (AGENTS.md — les relire avant de toucher)

- Backend : `cd backend && ../venv/bin/python -m uvicorn main:app --port 8001` (JAMAIS `./venv/bin/uvicorn`, shebangs cassés). Un seul process sur le port 8001 (`lsof -nP -iTCP:8001 -sTCP:LISTEN`).
- Tests : `cd backend && ../venv/bin/python -m unittest discover -q` + `test_i18n` + `test_licence_center` + `test_review_bugs`. Frontend : `npm run i18n:build && npm run lint && npm run build`.
- Si tu modifies `packaging/vendor/mflux-src` : re-pinner count+digest dans `backend/test_repo_security.py` (boucle de hash dans le test) et documenter dans `VENDORED_FROM.md`.
- Ne PAS toucher aux changements non commités pré-existants du checkout (`backend/generator.py`, `qwen_engine.py`, `staged_mflux.py`, `security_pro.py`, `model_manifests.json`, `test_*.py`, `frontend/index.html`, `packaging/shell/main.swift` ont des modifs locales) — commite séparément ou préserve-les.
- Mémoire 16 GB : **ne jamais lancer Ideogram 4** (poids 14.9 GB → a planté la machine le 19/11) ; une génération à la fois ; 1024×688 + image de référence = dead end sur M1 16 GB (512×768 OK) ; `MFLUX_IDLE_KILL_S=300`, éviction pipeline gratuite.
- Galerie : ne jamais supprimer/dédoublonner d'images. Barre de progreso : `transform: scaleX()` uniquement. `np.array(mx_array)` = abort process (le guard process-wide de `generator.py` couvre l'app, pas les scripts ad hoc).
- Ne pas push sur `upstream`. Commit avec le heredoc « Generated by Mistral Vibe. / Co-Authored-By: Mistral Vibe <vibe@mistral.ai> ».
- Pas de routeur : tabs = `useState` dans `App.jsx`.

## 5. Définition de done (items 1/2/3)

- [ ] Les 2 modèles (dreamlite-mobile, anima-turbo) visibles et sélectionnables dans l'UI, téléchargeables depuis l'onglet Models, génération OK via l'app avec sidecar JSON + image valide.
- [ ] ControlNet Qwen-Image-2.1-Turbo fonctionnel (image de contrôle + strength) ; LoRA Qwen-Turbo charge.
- [ ] `unittest discover` OK, `test_i18n` OK, `test_licence_center` OK, `npm run i18n:build && npm run lint && npm run build` OK.
- [ ] Synchro licences : `licences.js` = README = `site/licenses.html` (DreamLite CC BY-NC 4.0, Anima NC).
- [ ] README Benchmarks : nouvelles lignes avec des sidecars réels (512×768) — objectif attendu ~8 s (dreamlite-mobile 4s), ~33–58 s (anima-turbo 4–10s).
- [ ] Bench reproductible rejoué via `scripts/bench_mflux_new_models.py` et résultats cohérents avec `docs/veille/etude-rebase-mflux-2026-11-19.md`.
- [ ] Commit(s) sur `main` (pas `upstream`).

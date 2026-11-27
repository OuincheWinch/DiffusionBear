# Rebase mflux vendored + étude de temps DreamLite / Anima / Ideogram 4 — 2026-11-19

Machine de mesure : **MacBook Pro M1 16 GB** (même machine que les benchmarks README), wired limit 10.88 GB (politique app : 68 % mémoire unifiée), seed 42, prompt fixe, PNG validés non blancs (couleurs distinctes > 60k).

## 1. Rebase du mflux vendored — fait le 2026-11-19

- **Avant** : `packaging/vendor/mflux-src` = upstream `v.0.23.0` (63cec2b) + patches DiffusionBear (918 fichiers + pyproject).
- **Après** : upstream **mflux-community/mflux `main` @ `014252f`** (2026-11-18 ; dernier tag `v.0.24.0`) + mêmes patches. Nouveau pin : `mflux==0.24.0`.
- **Méthode** : merge 3-way par fichier (`git merge-file`, base `v.0.23.0`) ; 18 fichiers en conflit résolus à la main vers la sémantique de nos patches ; `models/flux3/` conservé intégralement nôtre (le backend y est câblé). Provenance et liste des patches : `packaging/vendor/mflux-src/VENDORED_FROM.md`.
- **Ce que le rebase apporte** : ports **DreamLite / DreamLite-mobile**, **Anima Turbo/Aesthetic/Base**, **Ideogram 4 FP8**, ERNIE Image, Iris-3B, SenseNova-U1.5 ; **ControlNet + LoRA Qwen-Image-2.1-Turbo** (#859) ; LoRA training (`mflux-train`) Qwen-2.1/Anima ; small text encoders FLUX.2/Z-Image/Qwen-2.1 ; `--enhance-prompt` en t2i ; patterns mflux-save (#869).
- **Garde-fous** : `test_repo_security.py` re-pinné (1206 fichiers, digest `8c97e448…`). Suite complète backend : **2255 tests OK (72 skipped)** d'abord via `PYTHONPATH` sur l'arbre vendored, puis **à nouveau après `pip install` dans le venv** (mflux 0.24.0 installé, 68 s) — re-vérifiée après le crash machine du 19/11. `import generator` OK, `MFLUX_VENDOR_VERSION = 0.24.0`, classes `DreamLite`/`Anima`/`Ideogram4` importables. Commit `aaa8141`.
- **Patches conservés** (voir VENDORED_FROM.md) : `mx.eval` avant `np.array` (image_util, vae_tiler), loaders 100 % sans torch, TE q4 16 GB (krea2/qwen/qwen21), warmup-then-compile M1/M2 (jamais shapeless), KV-cache FLUX.2, mappings LoRA/quant (scales/biases), layout single-file à la racine, arbre flux3.

## 2. Temps de génération mesurés (M1 16 GB, 512×768 sauf mention)

| Modèle (repo HF) | Steps | Résolution | Temps gen | s/step | Load (chaud) | Licence |
|---|---|---|---|---|---|---|
| **DreamLite-mobile** 0.39B (`carlofkl/DreamLite-mobile`, q8) | 4 | 512×768 | **8.25 s** | 2.06 | 0.7 s | CC BY-NC 4.0 |
| DreamLite-mobile | 4 | 1024×1024 | **17.74 s** | 4.43 | 0.7 s | CC BY-NC 4.0 |
| **DreamLite-base** 0.39B (`carlofkl/DreamLite-base`, q8, guidance 3.5) | 8 | 512×768 | **18.50 s** | 2.31 | 0.8 s | CC BY-NC 4.0 |
| DreamLite-base | 28 (défaut) | 512×768 | **66.43 s** | 2.37 | 0.8 s | CC BY-NC 4.0 |
| **Anima Turbo q4** (`mflux-community/anima-turbo-mflux-q4`, guidance 1.0) | 4 | 512×768 | **32.89 s** | 8.22 | 4.5 s | NC (circlestone-labs) |
| Anima Turbo q4 | 10 (défaut) | 512×768 | **58.03 s** | 5.80 | 4.5 s | NC |
| **Ideogram 4 q4** (`OsaurusAI/ideogram-4-mflux-4bit`) | 20 | 512×768 | **⚠️ non mesuré — crash machine** | | | NC (Ideogram Non-Commercial Model Agreement) |

Référence app (mêmes conditions, sidecars 18–19/11) : FLUX.2-klein 4B 4 steps ≈ 42–48 s ; Qwen-Image-2.1 2 steps DMD2 ≈ 47–61 s ; Z-Image-Turbo 8 steps ≈ 108–117 s ; SDXL Lightning 4 steps = 8.8 s.

**Lecture** : DreamLite-mobile à 4 steps (**8.25 s** à 512×768, 17.7 s à 1024²) est **~5× plus rapide que FLUX.2-klein 4B** et du même ordre que le SDXL Lightning — c'est un vrai mode « instantané » pour itérer le prompt. DreamLite-base à 8 steps (18.5 s) reste 2.3× plus rapide que FLUX.2 4B avec guidance 3.5. Anima Turbo q4 (3.1 GB) tient en 16 GB : 4 steps = 33 s, 10 steps = 58 s — très raisonnable pour un modèle anime dédié.

**Ideogram 4 — verdict 16 GB : hors périmètre.** Le repo q4 pèse 14.88 GB sur disque (transformer + unconditional_transformer + TE, déjà packés q4) : les poids seuls approchent la totalité de la mémoire unifiée. Le run du 19/11 a **planté la machine** (load > 30, reboot) pendant la génération. Même en q3 (`mflux-community/ideogram-4-mflux-q3`, 11.6 GB, public) les poids ~11.6 GB laissent < 4 GB d'activations — non viable sur M1 16 GB. **Cible : Mac 32–64 GB+.** Variante distillée (`VisualInference/ideogram-4-mlx-distilled`) existe en diffusers (gated) mais pas au format mflux. À retenir pour les machines 64 GB, pas pour la cible 16 GB de l'app.

## 3. Tailles de téléchargement (HF)

- DreamLite-base/mobile : 5.07 GB (text_encoder 4.26 GB + unet 0.78 GB) — quantisé q8 en RAM via `-q 8`
- Anima Turbo q4 : 3.11 GB (+ tokenizers Qwen3-0.6B / T5-XXL, fichiers texte seuls)
- Ideogram 4 q4 (OsaurusAI 4bit) : 14.88 GB (transformer + unconditional_transformer ~8.6 GB chacun en fp8, TE, VAE) — **gros** ; à réserver aux machines 16 GB avec de la marge, ou q3 (`mflux-community/ideogram-4-mflux-q3`)

## 4. Licences — point bloquant pour l'intégration UI

| Modèle | Licence | Conséquence |
|---|---|---|
| DreamLite base/mobile | **CC BY-NC 4.0** | lister NC dans `licences.js` / README / `site/licenses.html` ; pas de monétisation directe |
| Anima (circlestone-labs) | **Non-Commercial License** | idem |
| Ideogram 4 | **Ideogram Non-Commercial Model Agreement** (juin 2026) ; repos mflux-community gated, OsaurusAI/MLXBits publics en `license:other` | idem + vérifier la licence exacte du repo choisi |
| FLUX / Qwen-Image / Z-Image / FIBO | Apache-2.0 / autres permissives | cœur commercial-safe existant |

## 5. Prochaines étapes

1. Intégrer `dreamlite-mobile` comme modèle « Instantané » (dispatch `model_id`, presets 4 steps) — le rebase le rend disponible sans nouveau moteur.
2. Anima Turbo q4 (3.1 GB) comme moteur anime léger ; Ideogram 4 q4 pour design/typographie (gros téléchargement, preset 20 steps).
3. ControlNet Qwen-Image-2.1-Turbo (#859) et LoRA Qwen-Turbo à évaluer ensuite.
4. Vidéo LTX-2.5 : prochain chantier (pas couvert par ce rebase).

## 6. Rejeu du bench après intégration app (2026-11-19, commit de livraison)

`scripts/bench_mflux_new_models.py` rejoué **un process par cas** (isolation mémoire M1 16 GB, wired 68 %, seed 42, prompt fixe, PNG validé par couleurs distinctes) ; résultats dans `scripts/bench-results.jsonl` :

| Cas | Taille | Steps | gen_s (rejeu) | gen_s (étude §2) |
|---|---|---|---|---|
| dreamlite-mobile q8 | 512×768 | 4 | **5.92** (161 792 couleurs) | 8.25 |
| dreamlite-mobile q8 | 1024×1024 | 4 | **14.29** (233 745 couleurs) | 17.74 |
| dreamlite-base q8 | 512×768 | 8 | **18.29** (69 512 couleurs) | 18.5 |
| dreamlite-base q8 | 512×768 | 28 | **58.53** (103 193 couleurs) | 66.4 |
| anima-turbo-q4 | 512×768 | 4 | **26.51** (83 682 couleurs) | 32.89 |
| anima-turbo-q4 | 512×768 | 10 | **52.90** (70 697 couleurs) | 58.03 |

Écarts < 25 % sur tous les cas (charge machine variable) : le rejeu **confirme** les ordres de grandeur de l'étude. Côté app (sidecars `512×768` seed 42 q4, échantillon de 10 prompts via `/api/generate`, 2026-11-19) : DreamLite-mobile 4 steps médiane **4.61 s** (4.43–6.38), Anima Turbo 4 steps **25.00 s** (24.11–26.51), 10 steps **54.31 s** (53.18–57.31) — lignes ajoutées au tableau Benchmarks du README.

# État de l'art — génération locale sur Apple Silicon & position de DiffusionBear

- **Dernière mise à jour :** 2026-11-26 (jeudi) — édition 2
- **Prochaine mise à jour automatique :** tous les lundis à 09:00 (Europe/Zurich), via cron de session
- **Méthode :** veille web (`web_search`, fenêtre 2026-11-18 → 2026-11-26) + **sources primaires** (API GitHub pour les releases et commits mflux, PyPI, API Hugging Face) + inspection du repo (README, `backend/generator.py`, `backend/model_families.py`, `packaging/vendor/mflux-src`, sidecars `backend/data/generated/*.json`) + versions installées dans `venv/`
- **Règle d'or :** aucun chiffre de performance inventé — les perfs citées viennent des sidecars locaux ou sont marquées comme rapportées par des tiers
- **Historique :** snapshots dans `docs/veille/snapshots/YYYY-MM-DD.md`

---

## 0. Nouveautés depuis la dernière édition (2026-11-19 → 2026-11-26)

### mflux : aucune release, mais 12 commits — la semaine de la mémoire « out-of-core »

- **Release :** toujours `v.0.24.0` — publiée **2026-10-27T02:53Z** ; PyPI `mflux==0.24.0`. Sources : `api.github.com/repos/mflux-community/mflux/releases/latest` et `pypi.org/pypi/mflux/json`, consultés le 2026-11-26.
- **Notre base vendored :** `014252f` (2026-11-18). Upstream `main` est à **`6aebc2d6` (2026-11-26)** → **12 commits d'écart** (`api.github.com/.../commits?since=2026-11-18T00:00:00Z`, 13 commits listés dont notre base).

| Date | Commit | Changement upstream | Impact pour un Mac 16 GB |
|---|---|---|---|
| 2026-11-19 | `3aeed050` | Qwen-Image-2.1 **W8A8 DiT** (checkpoints int8_convrot ComfyUI, TensorOps M5) | Quantification alternative ; gain pleinement réservé aux M5 |
| 2026-11-21 | `4e74cc5c` | Qwen-Image-2.1 : **VAE décodé en bandes de lignes — pic VAE 35,6 → 8,8 GB à 1536×1536** (#1000) | 🔴 le plus utile : remet le 1536² dans le domaine du possible |
| 2026-11-21 | `d39c9639` | MOSS-Image : `--ssd-stream` (couches lues depuis le SSD) (#1008) | Modèles proches de la RAM |
| 2026-11-25 | `b6f4b12f` | Qwen-Image-2.1 : `--ssd-stream` blocs DiT + couches TE (#1022) | idem |
| 2026-11-25 | `6bb4cdea` | **Z-Image Turbo : `--ssd-stream`** (#1021) | idem, sur un modèle rapide |
| 2026-11-25 | `3d1d17a9` | MOSS-Image : `--stream-lora` (facteurs LoRA par bloc depuis le SSD) (#1010) | LoRA sans tout charger |
| 2026-11-25 | `e7e1b1c3` | FLUX.2-klein-9B : `--small-text-encoder` (TE élagué 5,1B) (#1016) | Plusieurs GB de moins pour klein-9B |
| 2026-11-25 | `ed62ae3d` | Qwen-Image-2.1 edit : résidus de blocs prévus dans le cache d'étapes (#1007) | Vitesse édition |
| 2026-11-25 | `32cb223f` | **SenseNova** : chargement d'un checkpoint upstream local (#1017) | Nouvelle famille ? **à vérifier** |
| 2026-11-26 | `f13b4d6c` | Pipelines edit : évaluer K/V et latents de référence à la construction (#1023) | Mémoire des pipelines edit |
| 2026-11-26 | `6aebc2d6` | Qwen-Image-2.1 W8A8 : rejeter une largeur d'entrée fausse / des poids int8 sans métadonnées convrot (#1018) | Robustesse |

**Lecture :** la semaine marque un virage « out-of-core » (streaming SSD, LoRA par bloc, VAE en bandes) — c'est précisément l'axe qui débloque les gros modèles sur 16 GB. Aucune release ne l'embarque encore, donc rien à installer, mais **le prochain rebase doit viser `6aebc2d6`** plutôt que `014252f`.

### Hugging Face / mflux-community : aucun nouveau dépôt dans la fenêtre

Dépôt le plus récemment modifié de l'org : `flux2-klein-4b-mflux-q8`, `lastModified` **2026-11-01** (`huggingface.co/api/models?author=mflux-community&sort=lastModified&direction=-1`). Rien entre le 18 et le 26 novembre.

### Signaux tiers (à vérifier avant d'en tirer une conclusion)

| Signal | Source | Statut |
|---|---|---|
| Portage **Qwen-Image-2.1 en EXL3 sur Metal** | `github.com/0xZKnw/mlxl3/issues/182` | Discussion de faisabilité, pas de livraison |
| **FLUX.3 Image Klein** dans un runtime MLX tiers : 1–10 images de référence, flow map distillé, 16 steps, guidance 1 | `github.com/lablup/mlxcel/issues/3230` | Issue, **à vérifier** |
| ComfyUI : vitesse Anima/ERNIE, LTX audio VAE, SAM 3.1, nœud HappyHorse | `docs.comfy.org/changelog` | Tiers ; aucun portage MLX annoncé |
| Nouveaux dépôts image « MLX » sur HF : `Nemotron-Labs-Image-30B-A3B-MLX-5bit`, `MiniMax-H3-Image-GGUF` | API HF `filter=text-to-image`, 2026-11-26 | 30B MoE **hors 16 GB** ; à surveiller |
| Comparatif tiers : mflux « 1–6 » pour Z-Image/Klein/Qwen-Image-2.1, « 45–230 » pour ERNIE/Krea 2/Qwen-Image-2512 (unité non précisée) | page non datée | **à vérifier**, à ne pas citer comme mesure |

### Côté DiffusionBear

- Page d'erreur 404 localisée (9 langues) + illustration générée en local : prête, **non déployée**.
- Aucun modèle ajouté cette semaine. Galerie utilisée pour produire 30 candidats d'illustration : Kromoss 5B q4 (8 steps, batch 5) ≈ **162–192 s/image** et FIBO 2.0 Turbo (4 steps, batch 5) ≈ **126–128 s/image** à 768², le 2026-11-20 (script de bench, pas des médianes sur 10 prompts).

---

## 1. Résumé exécutif

L'écosystème local Apple Silicon a bougé sur **trois axes** depuis notre dernière veille structurée :

1. **mflux upstream (mflux-community) a explosé** : 813 commits, nouveaux ports chaque semaine — **DreamLite 0.39B** (18 nov, modèle on-device unifié t2i+édition, ECCV 2026), **Anima Turbo/Aesthetic/Base** (27 oct, anime), **Ideogram 4 FP8** (design/typographie), **ControlNet + LoRA pour Qwen-Image-2.1-Turbo** (13 oct), support **vidéo** dans le repo. Notre mflux vendored (0.23.0 patché) est en retard structural sur tout ça.
2. **La vidéo locale sur Mac est devenue réelle en 2026** : LTX-2.5 (août 2026, modèle le plus « liked » de HF mi-novembre), Wan 2.1/2.2/2.7, MiniMax H3, HunyuanVideo 1.5 — avec des runtimes MLX natifs (mlx-video, Rapid-MLX, FastMetal-QAD) et des apps Mac gratuites (Phosphene, Draw Things v1.20260228.0 avec LTX-2 + audio). **DiffusionBear n'a aucune vidéo : c'est l'écart le plus visible.**
3. **Le fine-tuning LoRA on-device existe maintenant** : Draw Things entraîne des LoRA FLUX.2-klein 4B/9B, Z-Image, Qwen-Image et ERNIE directement sur Mac/iPhone (avril 2026). Nous savons importer/convertir des LoRA (Draw Things → MLX), mais pas en entraîner.

À périmètre constant, les intégrations les plus rentables : **rebase mflux (DreamLite/Anima/Ideogram/ControlNet)**, **mode « instantané » DreamLite 0.39B**, **upscaler neural q4**, **Qwen-Image-Edit-2511**, puis **vidéo LTX-2.5 en moteur subprocess** (pattern SDXL existant) et **LoRA training** sur FLUX.2-klein 4B.

---

## 2. Écosystème MLX / mflux — versions et mouvements

| Composant | Chez nous (venv) | État de l'art (19 nov 2026) | Commentaire |
|---|---|---|---|
| mlx | 0.32.4 (pin `mlx==0.32.1`) | 0.32.x courant, releases régulières ml-explore/mlx ; discussions « MLX Community Projects » actives | À re-pinner périodiquement |
| mlx-lm | 0.31.3 | mlx-lm actif ; **Rapid-MLX** revendique jusqu'à 4× plus rapide que mlx-lm (serveur LLM Apple Silicon) | Pertinent pour notre Prompt Enhancer (Qwen2.5-0.5B) |
| mlx-vlm | 0.7.10 | actif | — |
| transformers / tokenizers | 5.15.1 / 0.22.2 | 5.x courant | SDXL TE, pas de torch |
| **mflux** | **0.24.0 vendored** (`main@014252f`, 2026-11-18) + patches locaux (TE q4 cache, arbre flux3, no-torch) | **mflux-community/mflux** : dernière release `v.0.24.0` (2026-10-27), `main` à `6aebc2d6` (2026-11-26) ; 12 commits depuis notre base, dominés par la mémoire out-of-core (`--ssd-stream`, `--stream-lora`, VAE en bandes 35,6 → 8,8 GB, `--small-text-encoder`) | **Décalage de 12 commits** (voir §0) — plus de retard structurel |
| mlx-teacache | 0.14.1 | IonDen/mlx-teacache ajoute du support modèle (Anima #102, Ideogram 4 #69) | Bump à tester |
| mlx-taef | 0.8.1 | actif | fast-VAE SDXL |

**Matériel :** M5 (Neural Accelerators intégrés), Mac Studio M5 Ultra 512 GB, M6 évoqué dans les guides 2026 ; nouveau CEO Apple (John Ternus, sept 2026). La diffusion profite surtout de la bande passante mémoire et de l'ANE (voir §6).

---

## 3. Modèles image marquants (novembre 2026)

| Modèle | Params / nature | Licence | Dispo MLX | Intérêt pour DiffusionBear |
|---|---|---|---|---|
| **DreamLite** | 0.39B, unifié t2i+édition, « on-device » (arXiv 2603.28713, ECCV 2026) | open | **porté dans mflux le 18/11/2026 (#997)** | **Très fort** : mode « instantané » ~1 s pour itérer le prompt avant le rendu final |
| **Ideogram 4** | modèle image orienté design/typographie | open (ideogram-oss/ideogram4) | FP8/NF4 HF + **MLXBits q4/q8** + port mflux FP8 | **Fort** : typographie d'affiche (complète nos Virtual Boxes) |
| **Anima Turbo / Aesthetic / Base** | ~2–3.8B, anime, LoRAs Civitai existants | checkpoint Civitai (à vérifier) | **port mflux 27/10/2026**, teacache supporté | Fort pour la communauté anime/LoRA |
| **FLUX.2-dev** (full) | ~30B | open weights BFL 2026 | Comfy-Org le rend local ; q4 ≈ 64 GB | 16 GB : non ; Mac Studio 64 GB+ : oui |
| **Qwen-Image-2.1 / Turbo** | 7B | Apache-2.0 | déjà chez nous (DMD2 2 steps ~47 s) | maintenu ; upstream ajoute ControlNet+LoRA au Turbo |
| **Qwen-Image-Edit-2511 / 2512** | édition « consistency-first », multi-images | Apache-2.0 | ComfyUI/HF FP8 | **Upgrade prioritaire** de notre Qwen-Image-Edit |
| **Z-Image Base / Edit** | compléments de Z-Image-Turbo (déjà supporté, 8 s ~115 s) | Apache-2.0 | Draw Things v1.20260207.0 les supporte | Base = génération non-turbo ; Edit = édition instruction |
| **ERNIE Image** (Baidu) | nouveau venu 2026 | à vérifier | Draw Things v1.20260418.1 : import + LoRA + **training** | À évaluer (licence !) |
| **HunyuanImage-3.0 / 3.5** | 80B MoE (sept 2025), 3.5 preview API $0.024 | open (3.0) | surtout cloud/distillations | Surveiller les distillations q4 16 GB |
| **SD4 (2026)** | Stability AI « survit », SD4 Ultra photoréaliste | open | ComfyUI/SDNext | À évaluer vs notre ligne SDXL Lightning (8.8 s) |
| **Chroma1-HD** | 8.9B Apache-2.0 dérivé FLUX.1-schnell | Apache-2.0 | ComfyUI ; déjà étudié (`docs/prospective/chroma1_hd_study.md`) | Uncensored/CFG — niche, licence OK |
| **Kroma v0.2 / Chroma-H3** | fine-tunes Chroma+Krea2 ; Chroma-H3 = DiT vidéo MiniMax H3 en stills + LoRA DMD 8 steps | — | ComfyUI | Intéressant pour perfs (DMD 8 steps) |
| Leaderboards | Arena t2i (snapshot 12/11/2026), pirouter (9/11/2026, 80 modèles, 127k votes) | — | — | Référence externe de qualité perçue |

---

## 4. Vidéo locale sur Mac — le grand mouvement 2026

- **LTX-2.5** (Lightricks, août 2026) : modèle **le plus « liked » de HF à la mi-novembre 2026** ; 10 s de vidéo en < 7 s (GPU haut de gamme, chiffre constructeur) ; multishot natif, audio unifié, 4K/50 fps, text encoder Gemma 4 12B ; LTX-2.3 (mars 2026) avait déjà livré un éditeur desktop ; test publié sur **M5 Max** (note.com).
- **Wan 2.1 / 2.2 / 2.7, MiniMax H3 (Hailuo), HunyuanVideo 1.5, LongCat, Kandinsky** : écosystème ouvert riche.
- **Runtimes MLX natifs** : `mlx-video`, **Rapid-MLX** (familles vidéo servies sur Apple Silicon, rapidmlx.com), **FastMetal-QAD** (Hao AI Lab UCSD, vidéo Metal optimisée), WanGP (faible VRAM, surtout NVIDIA mais tiling intelligent).
- **Apps Mac gratuites** : **Phosphene** (mrbizarro) — LTX-2.5 + MiniMax H3 locaux, son, lip-sync, **entraînement LoRA de personnages**, accès MCP ; **Draw Things v1.20260228.0** — LTX-2 avec audio.
- **Position DiffusionBear :** aucune génération vidéo. C'est le **plus gros écart fonctionnel** vs l'état de l'art. Un moteur subprocess LTX-2.5 i2v 480p/4–5 s tient dans 16 GB (pattern SDXL existant : `mlx_sdxl_engine.py` → `mlx_video_engine.py`).

---

## 5. LoRA & fine-tuning local

- **Draw Things** : entraînement LoRA **on-device** depuis v1.20260410.1 (avril 2026) pour **FLUX.2-klein 4B/9B, Z-Image, Qwen-Image, ERNIE Image** ; wiki « LoRa Training Notes » ; LoRA Storage porté à 20 GiB (Draw Things+).
- **Phosphene** : LoRA de personnages pour la vidéo, sur Mac.
- **Civitai 2026** : Creator Program (Buzz « bankable » à partir du 1er nov 2026, licensing fees, tips), **ToS modifiés en juin et août 2026**, « remix provenance », Muse Image dans le générateur, nouveaux outils d'édition image/vidéo. → Notre intégration Civitai (browse/téléchargement) doit suivre ; **vérifier notre allow-list de licences** avec les ToS 2026.
- **Nous :** import + conversion LoRA Draw Things→MLX (`lora_convert.py`), registre multi-familles (`model_families.py`) — **en avance côté import, en retard côté training**.

---

## 6. Performance & quantification

- **Quantification diffusion :** q4/q8 MLX reste la norme ; MXFP4/NVFP4 concernent surtout les LLM (GGUF vs MLX : 37 % plus lent en GGUF pour les LLM). Pour la diffusion sur M1 16 GB, q4 + distillation (DMD2/DMD4) reste le meilleur rapport qualité/vitesse.
- **Accélération cache :** TeaCache / **mlx-teacache** (chez nous 0.14.1 ; upstream ajoute Anima, Ideogram 4) ; speculative decoding pour diffusion en recherche active.
- **Apple Neural Engine (ANE)** : Draw Things v1.20260410.1 accélère ses modèles « 8-bit S » via l'ANE **sur M4 (hors M4 Max)** ; ANEForge (Python direct sur ANE, arXiv 2606.17090) ; « real-time diffusion on M3 Ultra » (arXiv 2605.16259) ; ONNX Runtime CoreML 6.8× plus rapide ; ModelPiper fait de l'upscale sur ANE. → **Nous n'exploitons pas l'ANE** : piste de gain sur M4/M5.
- **Nos perfs mesurées (M1 16 GB, 512×768, sidecars + README v0.5.0)** :

| Modèle | Steps | Médian / runs récents (18–19 nov, sidecars) |
|---|---|---|
| Juggernaut XL Lightning (SDXL) | 4 | **8.80 s** (README, 10 prompts) |
| FLUX.2-klein 4B | 4 | 44.09 s (README) ; **41.8–47.8 s** (sidecars 18/11) |
| Qwen-Image-2.1 + DMD2 | 2 | 46.77 s (README) ; **46.6–60.9 s** (sidecars) |
| Qwen-Image-Turbo 2.1 | 2 | **42.2–165 s** (sidecars, variance forte) |
| Z-Image-Turbo 6B | 8 | 114.97 s (README) ; **108–117 s** (sidecars) |
| Krea 2 Turbo 13B | 4 | 156.81 s (README) ; **161–309 s** (sidecars, variance) |
| FIBO 2.0 Edit Turbo | 4 | 93.68 s (README) |
| MOSS-Image SFT DMD | 4 | 83.57 s (README) |
| RMBG-3.0 (1024²) | — | **150–600 s** (sidecars, variance) |

Wired limit auto = 10.88 GB sur cette machine (68 % de 16 GB). Éviction pipeline : aucun coût mesurable (MFLUX_IDLE_KILL_S=300).

---

## 7. Apps concurrentes (nov 2026)

| App | Nouveautés 2026 marquantes |
|---|---|
| **Draw Things** | LTX-2 vidéo+audio (fév 2026) ; Z-Image Base + TAESD (fév) ; ANE pour 8-bit S sur M4 ; **LoRA training FLUX.2/Z-Image/Qwen/ERNIE** (avr) ; ERNIE Image (avr) ; revendique « 40 % plus rapide que ComfyUI » |
| **ComfyUI** | v0.40.4 (12 oct 2026) ; v0.32.6 (4 août 2026) a retiré temporairement une feature (non réintégrée à date selon vdf.ai) |
| **Phosphene** | app Mac OSS gratuite : vidéo LTX-2.5/MiniMax H3 locale + son + LoRA perso + MCP |
| **MLXBits image-studio** | app Mac MLX (Ideogram 4 etc.) |
| **DiffusionBee** | toujours actif, orienté grand public |
| Guides de référence | modelfit.io (guidés par tier de RAM, mis à jour 24/10/2026), vdf.ai (oct 2026) |

---

## 8. Apple on-device & CoreML

Apple Intelligence / Image Playground 2026 progresse en qualité mais reste fermé ; côté recherche, CoreML + ANE rendent la diffusion **temps réel** possible sur M3 Ultra/iOS (arxiv, MVP Factory, Draw Things engineering blog « Making Apple Neural Engine work in a custom inference stack »). Pour nous : piste **ANE pour modèles 8-bit sur M4/M5**, et éventuellement export CoreML d'un petit modèle (DreamLite ?) pour un futur companion iOS.

---

## 9. Tendances adjacentes

- **Upscale local :** SUPIR / OSEDiff / Real-ESRGAN ; app `local-upscale` (moteur PyTorch) ; ModelPiper sur ANE. **Nous : Lanczos + UnsharpMask uniquement** (`upscale.py`) → écart.
- **3D :** Gaussian splatting 100 % local Mac (RadianceKit, Apple Photogrammetry, OpenSplat/Brush) ; TRELLIS.2, Hi3DGen en tendance HF. Hors périmètre court terme.
- **Segmentation :** nous avons RMBG-3.0 Q4 (à jour).
- **Vidéo post-traitement :** FlashVSR, SeedVR2, RIFE (interpolation) — pertinent si on ajoute la vidéo.

---

## 10. Position de DiffusionBear

**Forces (vérifiées dans le repo) :**
- 9 familles / 12+ checkpoints, **pur MLX, zéro torch** (v0.5.0), app `.app` standalone signée ad-hoc
- LoRA multi-familles + **conversion Draw Things→MLX** + registre `loras.json`
- Virtual Boxes (FLUX.3, MOSS), RMBG-3.0 Q4 intégré galerie, import images externes
- SDXL Lightning **8.80 s** à 512×768 sur M1 16 GB — meilleur temps de notre gamme
- Benchmarks réels : 220 générations chronométrées (README), sidecars JSON systématiques
- i18n 9 langues, 464 tests backend, test_i18n de garde

**Écarts vs l'état de l'art (19 nov 2026) :**
1. Pas de **vidéo** (LTX-2.5, Wan, MiniMax H3 tournent localement sur Mac)
2. ~~mflux vendored en retard~~ → **résorbé le 2026-11-19** (base `014252f`, release 0.24.0, DreamLite/Anima/ControlNet intégrés). **Nouveau décalage : 12 commits** upstream (§0), dominés par la mémoire out-of-core — à reprendre au prochain rebase.
3. Pas de **LoRA training** on-device (Draw Things le fait depuis avril 2026)
4. Pas d'**upscaler neural** (SUPIR/OSEDiff q4)
5. Pas d'exploitation de l'**ANE** (M4/M5)
6. Qwen-Image-Edit **2511/2512** pas encore intégrés ; Z-Image Base/Edit, ERNIE Image absents

---

## 11. Recommandations à périmètre constant

### 🔴 Absolument nécessaire (prochaine release)
1. **Rebase du mflux vendored** sur mflux-community récent → ✅ **FAIT le 2026-11-19** : rebase sur `main` @ `014252f` (tag `v0.24.0`), 18 conflits résolus vers nos patches, `models/flux3/` gardé nôtre, 2255 tests OK, digest re-pinné (1206 fichiers). Détails + étude de temps : `docs/veille/etude-rebase-mflux-2026-11-19.md`. Débloque DreamLite, Anima, Ideogram 4, ControlNet Qwen-Turbo **sans nouveau moteur**.
2. **Mode « Instantané » DreamLite 0.39B** : ✅ **mesuré puis intégré** (commit `1acc603`) — app : médiane **4.61 s** @512×768/4 steps (10 prompts, vrais sidecars) ; bench script : 5.92 s. DreamLite-base 8 steps = 18.3–18.5 s, 28 steps = 58.5–66.4 s. ⚠️ licence **CC BY-NC 4.0** (sync licences OK). Anima Turbo q4 ✅ **intégré** : app 25.0 s/4 steps, 54.3 s/10 steps (bench 26.5/52.9 s), LoRA Anima supportés, licence NC (sync OK). Ideogram 4 : **hors 16 GB** (poids q4 = 14.9 GB → crash machine constaté le 19/11 ; q3 = 11.6 GB, non viable non plus) → réservé aux Mac 32–64 GB+, licence NC.
3. **Garder le banc de bench vivant** : sidecars systématiques + ce rapport hebdo (cron en place).

### 🟠 Pertinent (roadmap 1–2 releases)
4. **Vidéo LTX-2.5 i2v 480p / 4–5 s** en subprocess dédié (`mlx_video_engine.py`, pattern SDXL) — plus gros morceau, mais c'est la feature la plus demandée de l'état de l'art.
5. **LoRA training on-device**, FLUX.2-klein 4B d'abord (kohya-style MLX ; Draw Things prouve la faisabilité sur Mac).
6. **Upscaler neural q4** (OSEDiff ou SUPIR) dans `upscale.py`, à côté du Lanczos.
7. **Qwen-Image-Edit-2511/2512**, **Z-Image Base/Edit**, **ERNIE Image** (licence à vérifier) via le mflux rebased. ✅ ControlNet + LoRA Qwen-Image-2.1-Turbo **intégré** (`1acc603`) : branche `qwen-image-2-1-controlnet-union-mflux-q4`, champ UI image+force, sidecars conformes.
8. **ANE pour modèles 8-bit sur M4/M5** (gating hardware, fallback MLX pur sur M1).

### 🟡 Possible / à chiffrer avant
9. ControlNet FLUX.2 via mflux upstream ; TeaCache pour plus de familles ; speculative decoding.
10. Wan 2.7 720p sur machines 64 GB+ ; export CoreML (DreamLite) pour companion iOS ; 3D (hors périmètre).

### 👁️ À surveiller (veille hebdo)
Civitai ToS/licensing 2026 ; SD4 en local 16 GB ; distillations HunyuanImage q4 ; Anima derivatives (2.9B/3.8B merges) ; M6 ; LTX-2.5 distilled LoRAs (4 steps).

---

## 12. Étude de vitesse — où on gagne

| Levier | Gain attendu | Base |
|---|---|---|
| DMD2/DMD4 distillation (déjà fait : Qwen 2 s ≈ 47 s, MOSS DMD4 ≈ 84 s vs 25 s stock) | ×2–6 | nos sidecars |
| TeaCache sur familles compatibles | ~30–50 % de steps équivalents | mlx-teacache 0.14.1 chez nous |
| DreamLite 0.39B en preview | **mesuré : 8.25 s @512×768 / 4 steps (mobile), 18.5 s / 8 steps (base)** | bench 2026-11-19, M1 16 GB |
| ANE 8-bit S sur M4/M5 | significatif (Draw Things, avr 2026) | à benchmarker sur M4 |
| q4 vs bf16 text encoder (déjà fait : cache TE q4 persisté, 2.26 GB) | ~5 GB économisés | AGENTS.md 2026-10-02 |
| Vidéo LTX-2.5 480p 5 s sur M1 16 GB | **à mesurer** (subprocess dédié) | modelfit.io « LTX-2.5 on Mac » 2026 |

**Mesures locales de la semaine** — sidecars 512×768 de `backend/data/generated/` (1 sidecar = 1 génération ; la dispersion vient de la charge machine et du cache froid, ce ne sont pas des médianes) :

| Modèle (sidecar) | Steps | Temps |
|---|---|---|
| `mflux-community/anima-turbo-mflux-q4` | 4 | 47,8 s · 50,1 s |
| `mflux-community/anima-turbo-mflux-q4` | 10 | 68,8 s · 115,4 s |
| `Ouinche/FIBO2-Edit-Turbo-MLX-4bit` | 4 | 97,8 s → 255,3 s |
| `mlx-community/flux2-klein-9b-4bit` | 4 | 206,4 s · 216,5 s |
| `mflux-community/flux-1-schnell-mflux-q4` | 4 | 86,6 s → 197,7 s |
| `mlx-community/Qwen-Image-2.1-mflux-q4` | 2 | 47,0 s · 81,3 s |
| `Ouinche/krea2-turbo-mflux-q4-full` | 4 | 266,8 s |
| `zimageapp/FLUX-3-Klein-MLX-Q4` | 8 | 92,2 s · 92,6 s |

Les médianes de référence restent celles du README (10 prompts, seed fixé) — ces valeurs ne servent qu'à détecter une dérive.

---

## 13. Risques & licences à vérifier avant intégration

- **Licences** : Ideogram 4, Anima (checkpoint Civitai — possiblement NC), ERNIE Image (Baidu), LTX-2.5, DreamLite — la règle repo : synchroniser `site/licenses.html` + README + `frontend/src/data/licences.js` à chaque ajout.
- **Rebase mflux** : nos patches (cache TE q4 `<model>-te-q4-g64-<sha>.safetensors`, arbre flux3, torch_guard) doivent être re-portés ; gate : `test_review_bugs.py` + génération end-to-end 512×512 4 steps par famille.
- **Vidéo 16 GB** : VAE vidéo + frames → limiter 480p / 4–5 s / tiling ; dest = raw RGB (cf. note AGENTS.md SDXL).
- **Civitai** : ToS changés juin/août 2026 → revoir l'allow-list de téléchargement.

---

## 14. Sources principales (édition 1 : 2026-11-19 · édition 2 : 2026-11-26)

- mflux upstream : github.com/mflux-community/mflux (commits #997 DreamLite 18/11, #931 Anima 27/10, #859 ControlNet+LoRA Qwen-Turbo 13/10 ; « image & video models »), newreleases.io mflux-community, pypi.org/project/mflux
- DreamLite : arxiv.org/abs/2603.28713 (ECCV 2026 poster), reddit r/StableDiffusion
- Vidéo : wikipedia LTX ; ltx.io blog 2026 landscape ; github.com/deepbeepmeep/WanGP ; github.com/mrbizarro/phosphene + discuss.huggingface.co/t/phosphene…/191949 ; rapidmlx.com/docs/models/families/video ; haoailab.com/blogs/fastmetal ; modelfit.io/blog/ltx-2-5-on-mac ; modelfit.io/guides/local-video-generation-mac ; vdf.ai/blog/best-local-video-generation-models
- Draw Things : x.com/drawthingsapp (v1.20260207.0, v1.20260228.0, v1.20260410.1, v1.20260418.1), wiki.drawthings.ai/wiki/LoRa_Training_Notes, engineering.drawthings.ai (ANE), heyuan110.com guide 2026
- Modèles : ideogram-oss/ideogram4 + huggingface.co/ideogram-ai/ideogram-4-{fp8,nf4} + MLXBits/ideogram-4-mlx-{q4,q8} ; alphasignal.ai (FLUX.2-dev local Comfy-Org) ; bfl.ai/blog/flux-2 ; docs.bfl.ml/release-notes ; blog.comfy.org Qwen-Image-Edit-2511 ; diffusiondoodles.substack.com model rundown (Z-Image Turbo, Qwen 2512/Edit-2511, Flux.2 Dev) ; arxiv 2509.23951 (HunyuanImage-3.0) ; localaimaster.com Chroma guide ; x.com/Machinedelusion (Chroma-H3/Kroma) ; reddit Kroma v0.2 ; arena.ai/leaderboard/image (12/11/2026) ; pirouter.ai/rankings/t2i (9/11/2026)
- Perf/quant : IonDen/mlx-teacache (#69 Ideogram, #102 Anima) ; arxiv 2605.16259 (real-time diffusion M3 Ultra) ; arxiv 2606.17090 (ANEForge) ; xybrid.ai ONNX CoreML 6.8× ; hoxtonmacs.co.uk ANE M1–M5 ; famstack.dev GGUF vs MLX ; raullenchai/Rapid-MLX (4× vs mlx-lm)
- Civitai : civitai.com/articles/33294 (Creator Program 2026), articles ToS 20260615/20260618/20260813, releasebot.io/updates/civitai (nov 2026)
- Concurrence/apps : modelfit.io/guides/local-image-generation-mac (24/10/2026) ; vdf.ai best local image models (oct 2026) ; github.com/MLXBits/image-studio ; HerRei/local-upscale ; modelpiper.com (upscale ANE) ; radiancekit.de (Gaussian splatting Mac)
- Apple : news AP/AFP nov 2026 (CEO Ternus, AI tools China 4/11/2026) ; guides M5/M6 local AI 2026 (llmcheck.net, promptquorum, codersera)
- **Veille 2026-11-26 — sources primaires** : `api.github.com/repos/mflux-community/mflux/releases/latest` (`v.0.24.0`, 2026-10-27) ; `api.github.com/.../commits?since=2026-11-18` (13 commits, head `6aebc2d6`) ; `pypi.org/pypi/mflux/json` (0.24.0) ; `huggingface.co/api/models?author=mflux-community&sort=lastModified` (dernier 2026-11-01) ; `huggingface.co/api/models?filter=text-to-image` (2026-11-26) ; `github.com/0xZKnw/mlxl3/issues/182` ; `github.com/lablup/mlxcel/issues/3230` ; `docs.comfy.org/changelog` ; `aitools.flocci.in/tool/hunyuan-image-tencent` (HunyuanImage-3.0-Instruct 2026-01-26, **à vérifier**)

---

## Historique des mises à jour

| Date | Édition | Faits marquants |
|---|---|---|
| 2026-11-19 | v1 (snapshot `snapshots/2026-11-19.md`) | Première édition ; cron hebdo lundi 09:00 posé |
| 2026-11-19 | v1.1 | Rebase mflux vendored 0.23.0 → main@014252f (v0.24.0) effectué + testé (2255 tests OK) ; étude de temps DreamLite/Anima/Ideogram 4 lancée (`etude-rebase-mflux-2026-11-19.md`) |
| 2026-11-19 | v1.2 | Items 1/2/3 livrés (commit `1acc603`) : DreamLite-mobile Instantané (4.61 s médians app), Anima Turbo q4 (25.0/54.3 s), ControlNet Qwen-Image-2.1-Turbo + LoRA ; 2276 tests OK ; licences NC synchronisées ; caches Ideogram purgés (14 GB) |
| 2026-11-26 | v2 (snapshot `snapshots/2026-11-26.md`) | Veille 8 jours : aucune release mflux (0.24.0 du 27/10), aucun nouveau dépôt HF mflux-community ; **12 commits upstream** dominés par la mémoire out-of-core (VAE Qwen-Image-2.1 35,6 → 8,8 GB, `--ssd-stream` MOSS/Qwen/Z-Image, `--stream-lora`, `--small-text-encoder` klein-9B) → prochain rebase vers `6aebc2d6` ; mesures locales ajoutées |

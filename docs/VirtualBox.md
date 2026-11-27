# Spatial Bounding Boxes Guide (VirtualBox) — FLUX.3-klein

> **Documentation bilingue / Bilingual Documentation**  
> - [🇫🇷 Version Française](#-version-française)
> - [🇬🇧 English Version](#-english-version)

---

## 🇫🇷 Version Française

### 1. Introduction & Principe

Dans les modèles de génération d'images traditionnels (SD 1.5, SDXL, FLUX.1), contrôler précisément l'emplacement des objets dans l'image est un défi historique :
- **Par le texte seul** : Écrire *"un chat rouge à gauche et un chien bleu à droite"* provoque souvent du *concept bleeding* (mélange d'attributs où le chien devient rouge ou le chat bleu, ou placement aléatoire).
- **Par des outils externes** : L'utilisation de ControlNet, d'inpainting ou de masques régionaux impose des modèles secondaires lourds en mémoire VRAM/RAM et ralentit l'inférence.

**FLUX.3 Image Klein (5.6B)** intègre pour la première fois une capacité **native** de localisation spatiale (*Spatial Grounding / Bounding Box Conditioning*). Le modèle a été pré-entraîné pour interpréter directement des coordonnées géométriques insérées au cœur du prompt texte, sans aucun adaptateur externe ni surcoût de calcul !

---

### 2. Comment ça marche sous le capot ?

L'architecture de FLUX.3 repose sur deux composants majeurs qui rendent cette fonctionnalité possible :

1. **L'encodeur multimodal Qwen3-VL (4B)** :
   Contrairement aux anciens encodeurs de texte (CLIP ou T5) qui ne comprennent que le vocabulaire générique, Qwen3-VL est un modèle vision-langage entraîné sur la détection d'objets, le sous-titrage dense et l'analyse spatiale. Il tokenize et interprète les coordonnées numériques `[ymin, xmin, ymax, xmax]` comme de véritables coordonnées cartésiennes.

2. **Le Transformer de diffusion (DiT 5.6B) avec RoPE 2D** :
   Les tokens de description associés à leurs boîtes de coordonnées sont mis en correspondance via l'attention croisée/jointe avec les patchs latents 2D de l'image. Les embeddings de position rotative (RoPE 2D) guident le débruitage de chaque étape pour confiner mathématiquement l'objet dans la région demandée.

3. **Validation automatique dans DiffusionBear** :
   Le backend (`backend/flux3_spatial.py`) analyse le prompt, valide les coordonnées, les transmet au pipeline et enregistre les boîtes dans les métadonnées de l'image (`sidecar .json`) pour permettre leur réutilisation ultérieure en un clic.

---

### 3. Système de coordonnées : La grille normalisée 0 à 1000

Quelle que soit la résolution choisie pour votre rendu (512×512, 768×512, 1024×1024, etc.), l'image est toujours cartographiée sur une grille relative de **0 à 1000** :

```text
(0, 0) [Haut-Gauche] ------------------------ (0, 1000) [Haut-Droite]
  |                                                  |
  |             ymin (Bord supérieur)                |
  |                v                                 |
  |     xmin > +-------+                             |
  |            | OBJET |                             |
  |            +-------+ < xmax                      |
  |                ^                                 |
  |             ymax (Bord inférieur)                |
  |                                                  |
(1000, 0) [Bas-Gauche] --------------------- (1000, 1000) [Bas-Droite]
```

La syntaxe standard universelle est :
```text
[ymin, xmin, ymax, xmax]
```

- **`ymin`** : Bord supérieur (Top) — distance depuis le haut de l'image (0 à 1000).
- **`xmin`** : Bord gauche (Left) — distance depuis la gauche de l'image (0 à 1000).
- **`ymax`** : Bord inférieur (Bottom) — distance depuis le haut de l'image (0 à 1000).
- **`xmax`** : Bord droit (Right) — distance depuis la gauche de l'image (0 à 1000).

> **Règle impérative** :
> `0 <= ymin <= ymax <= 1000` et `0 <= xmin <= xmax <= 1000`.

---

### 4. Syntaxe dans les prompts & Exemples concrets

La règle d'or est d'écrire la boîte de coordonnées **immédiatement avant** la description de l'élément à positionner. Vous pouvez spécifier jusqu'à 10 boîtes distinctes dans un même prompt.

#### Exemple A : Séparation stricte de deux sujets (évite la contamination)
```text
[150, 50, 850, 480] a fluffy ginger kitten sitting, [150, 520, 850, 950] a black labrador puppy sitting, soft daylight, studio photography, 8k
```
- Le chaton roux est verrouillé dans la moitié gauche (`xmin=50` à `xmax=480`).
- Le chiot labrador noir est verrouillé dans la moitié droite (`xmin=520` à `xmax=950`).
- **Résultat** : Zéro mélange de pelage ou d'attributs entre les deux animaux.

#### Exemple B : Sujet centré et élément de décor précis
```text
A serene Japanese garden with blooming cherry trees, [350, 300, 850, 700] a red traditional arched wooden bridge over a stream, [100, 750, 400, 950] an ancient weathered stone lantern
```
- Le pont rouge arqué occupe le centre-bas.
- La lanterne de pierre est positionnée dans le quart supérieur droit.

#### Exemple C : Cadrage cinématographique (Premier plan / Arrière-plan)
```text
Cinematic cyberpunk street in the rain, [600, 200, 980, 800] a sleek futuristic sports car with glowing blue headlights, [50, 350, 450, 650] a neon billboard advertising ramen
```

---

### 5. Raccourcis d'interface dans DiffusionBear

Lorsque le modèle **FLUX.3-klein** est sélectionné dans DiffusionBear, une barre dédiée **Spatial Box** s'affiche sous le champ de prompt :

| Bouton | Coordonnées insérées | Position résultante |
| :--- | :--- | :--- |
| **`[+] Box`** ou **`Centre`** | `[250, 250, 750, 750]` | Carré central occupant 50 % de l'image |
| **`Gauche`** | `[100, 50, 900, 500]` | Moitié gauche verticale (portrait/sujet à gauche) |
| **`Droite`** | `[100, 500, 900, 950]` | Moitié droite verticale (portrait/sujet à droite) |
| **`Haut`** | `[50, 100, 500, 900]` | Moitié supérieure horizontale (ciel, décor haut) |
| **`Bas`** | `[500, 100, 950, 900]` | Moitié inférieure horizontale (sol, premier plan) |

Un simple clic sur l'un de ces boutons injecte instantanément les coordonnées à l'emplacement du curseur dans votre prompt.

---

### 6. Bonnes pratiques & Astuces

1. **Laisser une marge de respiration** :  
   Évitez les boîtes trop étriquées (ex: `[10, 10, 30, 30]`). Laissez 50 à 100 unités de marge autour du sujet pour que le modèle puisse générer les contours, les ombres portées et les reflets naturels.
2. **Combiner description globale et locale** :  
   Commencez par poser l'atmosphère générale, la lumière et le style (ex: *"A sunny morning in Venice, oil painting style"*), puis détaillez les sujets localisés avec leurs boîtes.
3. **Respecter les proportions** :  
   Pour un personnage debout, préférez une boîte allongée verticalement (ex: `[100, 300, 900, 700]`). Pour un véhicule ou un paysage, préférez une boîte allongée horizontalement.

---

## 🇬🇧 English Version

### 1. Introduction & Overview

In conventional text-to-image diffusion models (SD 1.5, SDXL, FLUX.1), precisely positioning multiple objects across a canvas has always been notoriously difficult:
- **Text-Only Prompts**: Writing *"a red ball on the left and a blue cube on the right"* frequently causes *attribute bleeding* (the ball turns blue or the cube turns red, or placement is ignored).
- **External Frameworks**: Using ControlNet, inpainting masks, or regional adapters demands heavy extra VRAM and adds significant latency.

**FLUX.3 Image Klein (5.6B)** introduces built-in, **native** spatial grounding (*Spatial Bounding Box Conditioning*). The model is natively trained to comprehend geometric coordinates directly inside natural language prompts—requiring zero external adapters and zero performance penalty!

---

### 2. How it Works Under the Hood

The FLUX.3 architecture combines two key technological breakthroughs:

1. **Qwen3-VL (4B) Multimodal Text Encoder**:  
   Unlike traditional text encoders (CLIP or T5) that only understand prose, Qwen3-VL is a vision-language model trained on object detection, dense captioning, and spatial grounding. It tokenizes numeric coordinates formatted as `[ymin, xmin, ymax, xmax]` as true Cartesian positions.

2. **Diffusion Transformer (DiT 5.6B) with 2D RoPE**:  
   Tokens containing spatial coordinates are cross-attended against the 2D image latent tokens. The 2D Rotary Position Embeddings (RoPE) provide spatial awareness, forcing the attention mechanism to denoise each object strictly within its assigned bounding box.

3. **Backend Parsing in DiffusionBear**:  
   The DiffusionBear backend (`backend/flux3_spatial.py`) automatically parses and validates up to 10 bounding boxes per prompt, feeds them to the generation pipeline, and persists them into the image sidecar metadata for instant reuse.

---

### 3. Coordinate System: 0–1000 Normalized Grid

Regardless of the generated image resolution (512×512, 768×512, 1024×1024, etc.), the canvas is always mapped onto a relative **0 to 1000** integer grid:

```text
(0, 0) [Top-Left] ---------------------------------- (0, 1000) [Top-Right]
  |                                                        |
  |                ymin (Top edge)                         |
  |                   v                                    |
  |        xmin > +-------+                                |
  |               | OBJECT|                                |
  |               +-------+ < xmax                         |
  |                   ^                                    |
  |                ymax (Bottom edge)                      |
  |                                                        |
(1000, 0) [Bottom-Left] --------------------------- (1000, 1000) [Bottom-Right]
```

The universal computer-vision coordinate order is:
```text
[ymin, xmin, ymax, xmax]
```

- **`ymin`**: Top edge — vertical distance from the top of the canvas (0 to 1000).
- **`xmin`**: Left edge — horizontal distance from the left of the canvas (0 to 1000).
- **`ymax`**: Bottom edge — vertical distance from the top of the canvas (0 to 1000).
- **`xmax`**: Right edge — horizontal distance from the left of the canvas (0 to 1000).

> **Validation Rule**:
> `0 <= ymin <= ymax <= 1000` and `0 <= xmin <= xmax <= 1000`.

---

### 4. Prompt Syntax & Practical Examples

The golden rule is to place the coordinate box **directly before** the description of the entity you want to position. You can include up to 10 boxes in a single prompt.

#### Example A: Clean separation of two subjects (No attribute bleeding)
```text
[150, 50, 850, 480] a fluffy ginger kitten sitting, [150, 520, 850, 950] a black labrador puppy sitting, soft daylight, studio photography, 8k
```
- The ginger kitten is bounded to the left half (`xmin=50` to `xmax=480`).
- The black labrador puppy is bounded to the right half (`xmin=520` to `xmax=950`).
- **Result**: Zero color spill or anatomy overlap between the two animals.

#### Example B: Centered subject with environmental accessory
```text
A serene Japanese garden with blooming cherry trees, [350, 300, 850, 700] a red traditional arched wooden bridge over a stream, [100, 750, 400, 950] an ancient weathered stone lantern
```
- The arched red bridge is anchored in the center-bottom.
- The ancient stone lantern is positioned in the upper-right quadrant.

#### Example C: Cinematic Foreground & Background
```text
Cinematic cyberpunk street in the rain, [600, 200, 980, 800] a sleek futuristic sports car with glowing blue headlights, [50, 350, 450, 650] a neon billboard advertising ramen
```

---

### 5. UI Quick Shortcuts in DiffusionBear

When **FLUX.3-klein** is selected in DiffusionBear, a **Spatial Box** assistant toolbar appears under the prompt field:

| Preset Button | Inserted Coordinates | Placement Target |
| :--- | :--- | :--- |
| **`[+] Box`** / **`Center`** | `[250, 250, 750, 750]` | Central box occupying 50% of the canvas |
| **`Left`** | `[100, 50, 900, 500]` | Left vertical half (portrait / left-aligned subject) |
| **`Right`** | `[100, 500, 900, 950]` | Right vertical half (portrait / right-aligned subject) |
| **`Top`** | `[50, 100, 500, 900]` | Upper horizontal band (sky, ceiling, mountains) |
| **`Bottom`** | `[500, 100, 950, 900]` | Lower horizontal band (ground, vehicle, foreground) |

Clicking any preset chip inserts the coordinates directly at your text cursor.

---

### 6. Best Practices & Pro Tips

1. **Provide Breathing Room**:  
   Avoid extremely tight boxes (e.g., `[10, 10, 30, 30]`). Allow a 50–100 unit margin around the subject so the model can render natural silhouettes, cast shadows, and environmental reflections.
2. **Combine Global and Local Descriptions**:  
   Establish the overall setting, lighting, and style first (e.g., *"A foggy morning in a Scandinavian forest, 35mm film photography"*), then specify each localized element with its coordinate tag.
3. **Match Aspect Ratios to Subjects**:  
   Use tall boxes for standing figures or towers (e.g., `[100, 300, 900, 700]`), and wide boxes for vehicles, horizons, or reclining animals.

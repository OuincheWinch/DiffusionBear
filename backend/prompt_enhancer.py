"""Engine-Adaptive Prompt Enhancer with Active LoRA Trigger Preservation.

Powered by a local LLM (mlx-community/Qwen2.5-1.5B-Instruct-4bit)
running natively on Apple Silicon via MLX.

ENGINE_PROFILES here are a distilled compilation of official vendor prompt
guidance (verified against primary sources):
- FLUX.2 Klein : BFL Prompting Guide (docs.bfl.ai/guides/prompting_guide_flux2)
  — Subject+Action+Style+Context, word-order priority, no negatives, camera
  refs, object-bound HEX colors, quoted text, JSON-native, upsampling support
- SDXL Lightning: RunDiffusion/Juggernaut XL + Stability guidance
  (rundiffusion.com/juggernaut-xl-rundiffusion-guide) — dual-CLIP 77-token
  chunks, compact comma tags, background mandatory, decisive token weight
- Krea 2 Turbo  : Krea official prompting guidelines
  (github.com/krea-ai/krea-2/blob/main/docs/prompting.md) — natural language,
  long detailed prompts best, five structural layers, quoted in-image text
- Z-Image Turbo : Alibaba Tongyi-MAI model card + community best-practice
  guides (huggingface.co/Tongyi-MAI/Z-Image-Turbo) — long factual narrative,
  guidance 0 / no negatives, 6-part camera-first formula, bilingual text
- MOSS-Image   : OpenMOSS MOSS-Image USERGUIDE + official prompt-enhancer skill
  (github.com/OpenMOSS/MOSS-Image) — detailed English captions, exact quoted
  text with explicit line breaks, tags in the official order, JSON layout
  coordinates only when a layout is actually requested

Two output formats are supported:
- output_format="text" (default): engine-shaped narrative/tagged prose prompt.
- output_format="json": prompts filled per a structured JSON prompt schema
  ({"subject", "appearance", "action", "setting", "lighting", "atmosphere",
    "composition", "details", "text_elements", "technical", "trigger_word"}).
  Returns the JSON as `enhanced` plus a natural-language flattening for engines
  that prefer prose.

Guarantees 100% preservation and natural integration of active LoRA trigger words.
"""

import json
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import app_settings


class EnhancementCancelled(Exception):
    """Raised when a queued/cancelled prompt-enhancement should stop early."""


_model = None
_tokenizer = None
_loaded_repo: str | None = None
_model_lock = threading.Lock()
_lora_registry_cache: Optional[Tuple[int, int, List[Dict[str, Any]]]] = None

DATA_DIR = app_settings.DATA_DIR
ASSET_DIR = app_settings.ASSET_DIR
LOCAL_MODEL_DIR = ASSET_DIR / "models" / "qwen2.5-1.5b-instruct-4bit"

if LOCAL_MODEL_DIR.exists():
    DEFAULT_MODEL_REPO = str(LOCAL_MODEL_DIR)
else:
    DEFAULT_MODEL_REPO = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

# The enhancer backend is a user choice, not a bundled default. Weights live
# in the HF cache or the asset dir and are never shipped with the app; first
# use downloads. Unknown stored values fall back to qwen15 rather than
# raising, so a hand-edited settings.json cannot break Enhance.
ENHANCER_MODELS = {
    "qwen15": {
        "label": "Qwen2.5 1.5B",
        "repo": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "local_dirname": "qwen2.5-1.5b-instruct-4bit",
        "size": "839 MB",
    },
    "qwen05": {
        "label": "Qwen2.5 0.5B",
        "repo": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "local_dirname": "qwen2.5-0.5b-instruct-4bit",
        "size": "276 MB",
    },
}
DEFAULT_ENHANCER_MODEL_ID = "qwen15"


def selected_model_id() -> str:
    """The active enhancer backend id, falling back on anything unknown."""
    stored = app_settings.get_setting("enhancer_model")
    return stored if stored in ENHANCER_MODELS else DEFAULT_ENHANCER_MODEL_ID


def resolve_model_repo(model_id: str | None = None) -> str:
    """Local asset dir when present, else the HF repo id. Per selection."""
    info = ENHANCER_MODELS.get(model_id or selected_model_id())
    if info is None:
        info = ENHANCER_MODELS[DEFAULT_ENHANCER_MODEL_ID]
    local_dir = ASSET_DIR / "models" / info["local_dirname"]
    if local_dir.exists():
        return str(local_dir)
    return info["repo"]


def model_cached(model_id: str) -> bool:
    """Whether the backend's weights are on disk. Local scan only, no network.

    A model counts as cached when the asset dir exists or the HF cache holds
    the repo (probed with local_files_only, which never touches the network).
    """
    info = ENHANCER_MODELS.get(model_id)
    if info is None:
        return False
    if (ASSET_DIR / "models" / info["local_dirname"]).exists():
        return True
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(info["repo"], local_files_only=True)
        return True
    except Exception:
        return False


def list_models() -> list[dict]:
    """Both backends with live cached state, for the picker UI."""
    active = selected_model_id()
    return [
        {
            "id": model_id,
            "label": info["label"],
            "size": info["size"],
            "cached": model_cached(model_id),
            "active": model_id == active,
        }
        for model_id, info in ENHANCER_MODELS.items()
    ]


def download_model(model_id: str) -> str:
    """Fetch a backend's weights into the HF cache. Blocking; the UI shows a
    spinner. Raises ValueError on an unknown id before attempting anything."""
    info = ENHANCER_MODELS.get(model_id)
    if info is None:
        raise ValueError(f"unknown enhancer model: {model_id}")
    from huggingface_hub import snapshot_download

    return str(snapshot_download(info["repo"]))

# Structured JSON prompt schema (used for output_format="json").
JSON_PROMPT_SCHEMA = {
    "subject": "Primary subject using fictional identity (name, age, background) OR specific object/scene",
    "appearance": "Detailed physical description (skin tone, hair, facial structure, clothing, materials)",
    "action": "What the subject is doing or their pose",
    "setting": "Environment and location details with geographic anchors",
    "lighting": "Specific lighting conditions (soft daylight, overcast sky, sharp shadows)",
    "atmosphere": "Environmental qualities (foggy, humid, dusty)",
    "composition": "Camera angle and framing (close-up, wide shot, overhead view)",
    "details": "Additional elements (background objects, secondary subjects, textures)",
    "text_elements": 'Any text to appear in image (use double quotes: "Morning Brew", specify font and placement)',
    "technical": "Optional camera specs (Shot on Leica M6, shallow depth of field, visible film grain)",
    "trigger_word": "Mandatory trigger words to activate LoRA",
}

MING_JSON_INSTRUCTIONS = """{target_line}
Official inclusionAI Ming-Image structured-prompt style. Expand the user's request into one precise, high-resolution Figma-style caption. Return only one JSON object.

Use exactly two top-level keys. `canvas_settings` contains exactly `aspect_ratio`, `ambient_lighting`, and `image_style`. `layers` lists visible groups from background to topmost overlay. Every layer contains exactly `description`, `coordinates`, `hierarchy_and_relation`, and `color_specs`; `color_specs` is an array of hex colors.

`coordinates` MUST be one string, never an object or array, in exactly this form: "cx: 0.500, cy: 0.500, w: 1.000, h: 1.000". Values are normalized; each bbox encloses its complete owned object and stays inside the canvas.

Set `canvas_settings.aspect_ratio` to {aspect_ratio}.

For UI, web, app, poster, dashboard, or page-design requests, describe the target device/viewport and aspect ratio. For a scrollable page, keep above-fold UI at natural scale and let later sections continue below it instead of compressing the complete page. Never draw browser, device, operating-system, or host-container chrome unless the user explicitly asks for a device mockup.

A layer is one selectable visible semantic group: background, full person, coherent object, panel, card, row, or text block. Prefer the fewest groups that preserve the layout. Keep people and objects intact. Never create invisible parents, guides, placeholders, empty layers, duplicate summaries, or multiple owners for one element.

Preserve every user-supplied rendered string character-for-character and as one contiguous string. Unless multiple visible copies are requested, it must occur exactly once across all `description` fields and zero times in `hierarchy_and_relation`. Quote it only where describing its visible rendering; refer to the related subject elsewhere with unquoted semantic wording. Enumerate intended copy, invent extra copy sparingly, and never hide content behind "other text", "remaining labels", or "etc."

Describe concrete composition, typography, materials, texture, lighting, pose, and camera treatment without literary filler. Use `hierarchy_and_relation` only for ownership, alignment, containment, stacking, and occlusion.

Infer structured layouts first. Use one complete layer per card and state its row and column. A compact secondary table may be one layer only if every header and cell is listed; otherwise use a visible shared frame when present, one complete header, and one complete layer per body row, binding values to columns and stating blanks. Enumerate sequences, schedules, spans, gaps, and vacant tracks in visual order. Do not mistake ordinary alignment for a table.

Silently verify schema, string coordinates, Z-order, exact-text counts, geometry, bbox validity, and completeness."""

ENGINE_PROFILES = {
    "flux1": {
        "label": "FLUX.1-schnell / dev (CLIP + T5)",
        "length": "30 to 60 words",
        "max_words": 60,
        "instructions": (
            "{target_line}\n"
            "Official prompting style (BFL FLUX.1 guide) — concise natural-language scene description:\n"
            "1. Write one or two flowing sentences, not a tag list. Order: main subject -> action -> "
            "distinctive style or medium -> setting and light -> secondary details.\n"
            "2. FLUX.1-schnell is guidance-distilled and runs without a negative prompt. Describe only "
            "what should appear, in positive terms ('sharp focus throughout', 'empty street').\n"
            "3. Keep the prompt under ~60 words. Precise nouns and one clear light direction beat "
            "stacked quality adjectives.\n"
            "4. Name a concrete medium or photographic finish when style matters (macro photograph, "
            "soft graphite drawing, 35mm film grain).\n"
            "5. In-image text: put the exact words in double quotes and state placement.\n"
            "STRICTLY FORBIDDEN: tag soup, generic quality buzzwords (masterpiece, 8k, photorealistic), "
            "and negative-phrased instructions."
        ),
    },
    "flux2": {
        "label": "FLUX.2 Klein (T5 Narrative)",
        "length": "40 to 75 words",
        "max_words": 75,
        "instructions": (
            "{target_line}\n"
            "Official prompting style (BFL Prompting Guide) — NARRATIVE PROSE built on Subject + Action + Style + Context:\n"
            "1. Write continuous natural-language sentences like a novelist setting a scene — NOT a comma-separated tag list.\n"
            "2. Word order is critical: FLUX.2 attends most to the START. Order elements Main subject -> Key action -> "
            "Critical style -> Essential context -> Secondary details.\n"
            "3. Length: 30-80 words is the sweet spot (go longer only for genuinely complex scenes).\n"
            "4. NO negative prompts exist on this model — describe only what you WANT, in positive terms "
            "('sharp focus throughout' NOT 'no blur'; 'empty, deserted street' NOT 'no people').\n"
            "5. Lighting and atmosphere carry the image — give one clear light direction and mood.\n"
            "6. For photorealism name a specific camera, lens, and film stock ('shot on Fujifilm X-T5, 35mm f/1.4, "
            "natural grain') — far stronger than 'professional photo'.\n"
            "7. Precise color: bind a HEX code to a specific object ('the car in color #FF0000'), never vaguely "
            "('use red somewhere').\n"
            "8. In-image text: put the exact words in double quotes and specify placement, font style/size, and color "
            "(hex OK), e.g. The text \"OPEN\" in red neon letters above the door.\n"
            "9. Keep character/object details consistent and repeat them VERBATIM across related prompts.\n"
            "STRICTLY FORBIDDEN: raw tag soup / comma chains, generic quality buzzwords "
            "(masterpiece, 8k, photorealistic, best quality), and any negative-phrased instruction."
        ),
    },
    "sdxl": {
        "label": "SDXL Lightning / Juggernaut XL (Dual CLIP)",
        "length": "40 to 55 words",
        "max_words": 55,
        "instructions": (
            "Target Engine: SDXL Lightning / Juggernaut XL (distilled UNet; dual OpenCLIP-G + CLIP-L text encoders, "
            "77 tokens per chunk).\n"
            "Official prompting style (Stability + RunDiffusion/Juggernaut guide) — COMPACT DESCRIPTIVE TAGS:\n"
            "1. Use short, comma-separated descriptive phrases — NOT long literary prose. Keep it UNDER ~55 words "
            "(~75 tokens): CLIP attention drops sharply past ~50-60 tokens and the tail is effectively ignored.\n"
            "2. Order matters as a visual story: Subject -> Action -> Location -> Aesthetic & Style. "
            "Everything critical goes FIRST.\n"
            "3. ALWAYS describe the background/environment — this checkpoint is GPT-4V caption-trained and produces "
            "flat, boring scenes from a bare subject ('cat with a hat in a misty forest' NOT 'cat with a hat').\n"
            "4. Use high-impact cinematic photography language: framing ('cinematic medium shot', 'close-up portrait'), "
            "lighting (moody rim light, volumetric beams), and finish ('detailed skin texture', 'sharp focus', "
            "'film grain', 'raw photo', '8k').\n"
            "5. Use explicit, specific descriptors and avoid stacked or conflicting adjectives — never mix "
            "'photorealistic' with 'anime' in one prompt; repetition dilutes token weight.\n"
            "6. In-image text: keep it short and wrap the exact words in double quotes (long sentences render poorly).\n"
            "7. Keep the positive prompt self-sufficient and free of negative phrasing — do not write 'no X'; "
            "describe the desired state instead ('clean plain background', 'perfect hands')."
        ),
    },
    "krea2": {
        "label": "Krea 2 Turbo (Deep Causal Latent)",
        "length": "50 to 90 words",
        "max_words": 90,
        "instructions": (
            "Target Engine: Krea 2 Turbo 13B (natural-language foundation model; local raw/turbo weights, no hosted expander).\n"
            "Official prompting style (Krea prompting.md) — NATURAL LANGUAGE, long and specific:\n"
            "1. Write flowing natural-language sentences, NEVER a comma-separated tag cloud — "
            "Krea states 'long detailed prompts yield best results'.\n"
            "2. Cover the five structural layers in order: (a) subject & action first; (b) medium/style EARLY "
            "('flat-color illustration', 'macro photograph', 'vintage analog collage'); (c) lighting & color; "
            "(d) composition & framing; (e) one or two grounding details (texture, environment, object).\n"
            "3. State the medium in the first sentence — one precise medium word beats ten stacked style adjectives.\n"
            "4. Put your top three requirements in the FIRST sentence; long prompts bury their priorities.\n"
            "5. Raw/Turbo local weights have no prompt expander, so write the full long-form prompt yourself "
            "(do not rely on enrichment).\n"
            "6. In-image text: wrap the exact words in double quotes and keep the string short, e.g. a neon sign "
            "reading \"OPEN ALL NIGHT\".\n"
            "7. NO negative prompts — express every constraint positively. Favor bold, high-contrast, expressive "
            "visual direction over sterile realism."
        ),
    },
    "z-image-turbo": {
        "label": "Z-Image Turbo 6B (S3-DiT)",
        "length": "80 to 200 words",
        "max_words": 200,
        "instructions": (
            "Target Engine: Z-Image Turbo 6B (single-stream S3-DiT; ~512-token default text cap; guidance_scale 0; "
            "no negative prompt).\n"
            "Official prompting style (Alibaba Tongyi-MAI + community best-practice guides) — LONG STRUCTURED FACTUAL NARRATIVE:\n"
            "1. Write 80-200 words of explicit, factual, camera-brief prose. Long and precise beats short; avoid "
            "flowery or poetic flourish.\n"
            "2. Follow the 6-part camera-first formula IN ORDER: Subject -> Environment -> Composition (shot type/angle) "
            "-> Lighting -> Finish/material/style -> Constraints.\n"
            "3. The model is an obedient art director: if you don't state it, it's allowed; if you state it vaguely, it "
            "improvises. Spell out age, appearance, expression, pose, wardrobe, and surroundings.\n"
            "4. NO negative prompts exist: convert every exclusion into a POSITIVE constraint inside the prompt "
            "('wearing a modest tailored suit, fully clothed', 'clean plain background, no text or watermark').\n"
            "5. Give one coherent key-light direction and a concrete medium/style — the model responds strongly to "
            "lighting keywords and camera terms (macro, low-angle, 45 degree angle, shallow depth of field).\n"
            "6. In-image text: it renders English and Chinese well — give the EXACT words in quotes, plus language "
            "and placement.\n"
            "7. Keep one clear visual intent; never mix contradictory styles, and repeat critical items VERBATIM "
            "for consistency."
        ),
    },
    "qwen": {
        "label": "Qwen-Image 2.1 (Qwen3-VL)",
        "length": "70 to 150 words",
        "max_words": 150,
        "instructions": (
            "Target Engine: Qwen-Image 2.1 (Qwen3-VL text encoder; experimental MLX port).\n"
            "Write 70-150 words of concrete, scene-first visual prose. State the subject, action, environment, "
            "composition, camera, lighting, materials, and important text before adding atmosphere.\n"
            "Keep relationships and spatial placement explicit. Use exact quoted text and color hex values when "
            "they matter. Prefer positive visual descriptions; only use a negative clause when the requested "
            "avoidance is essential. Preserve LoRA trigger phrases verbatim.\n"
            "Do not add unsupported camera brands, resolutions, or quality buzzwords."
        ),
    },
    "ming": {
        "label": "Ming-Image 0.2 Design 24.9B",
        "length": "90 to 220 words",
        "max_words": 220,
        "max_tokens": 1536,
        "instructions": (
            "{target_line}\n"
            "Official prompting style (inclusionAI Ming-Image Design guide) — STRUCTURED DESIGN BRIEF:\n"
            "1. Write a precise art-direction brief, 90-220 words, describing the finished designed image. "
            "Ming is strongest for UI screens, posters, infographics, branding, packaging and other layout-heavy images.\n"
            "2. Organize the brief in this order: canvas and format -> subject/product -> layout hierarchy -> "
            "exact text -> typography -> palette -> lighting/materials -> composition details -> constraints.\n"
            "3. Exact visible text is the core feature: put every string in double quotes, preserve capitalization, "
            "punctuation, line breaks and language, and state where each string appears. English and Chinese are supported.\n"
            "4. Describe typography concretely: sans/serif/monospace, weight, relative size, alignment, spacing, and whether text is curved, vertical, embossed, neon, printed or part of the object.\n"
            "5. Describe layout relationships like a design specification: grid, margins, columns, cards, hero area, badges, icons, "
            "foreground/background ordering, and which element is dominant. Do not invent numeric coordinates unless the user supplied them.\n"
            "6. Give a controlled palette with hex values when useful, bound to specific elements ('headline in #F8FAFC').\n"
            "7. No negative-prompt field exists: express exclusions positively ('clean empty background', 'uncluttered layout').\n"
            "8. Keep the brief visually concrete; do not mention prompts, generation, quality tags or invisible process notes.\n"
            "STRICTLY FORBIDDEN: tag soup, vague 'beautiful design' filler, placeholder text such as lorem ipsum unless requested, "
            "and unquoted text that must render exactly."
        ),
    },
    "flux3": {
        "label": "FLUX 3 Image Klein 5.6B",
        "length": "60 to 160 words",
        "max_words": 160,
        "max_tokens": 320,
        "instructions": (
            "{target_line}\n"
            "Official prompting style (BFL FLUX 3 / FLUX 3 Image Klein guide) — CAPTION-FIRST PROSE:\n"
            "1. Write a complete scene caption, 60-160 words. FLUX 3 Klein gives generic output from a short typed prompt; do the expansion yourself instead of relying on a prompt upsampler.\n"
            "2. Order: medium/style -> subjects and positions -> setting -> light -> framing/detail. Put what matters most first.\n"
            "3. Use connected prose and explicit relationships ('who holds what, what stands behind what'), never a keyword list.\n"
            "4. Name a concrete medium/style first. For photorealism use camera body, lens, aperture and film stock when relevant.\n"
            "5. Give one clear light source, direction, quality and temperature. Remove any sentence that describes nothing visible.\n"
            "6. Exact visible text: put it in double quotes, preserve capitalization and punctuation, use \n for intended line breaks, and describe type and placement.\n"
            "7. No negative prompt, guidance, seed or upsampling field on FLUX 3 Image. Rewrite exclusions as positive descriptions ('empty promenade', not 'no crowds').\n"
            "8. Do not add quality tags such as masterpiece, highly detailed or ultrarealistic.\n"
            "STRICTLY FORBIDDEN: keyword soup, fake camera claims the user did not ask for, negative phrasing, and invisible process notes."
        ),
    },
    "moss": {
        "label": "MOSS-Image SFT 5B (Qwen3.5 TE)",
        "length": "70 to 160 words",
        "max_words": 160,
        "max_tokens": 320,
        "instructions": (
            "{target_line}\n"
            "Official prompting style (OpenMOSS MOSS-Image USERGUIDE + prompt-enhancer skill) — DETAILED CAPTION FIRST:\n"
            "1. Write a detailed English caption of 70-160 words describing the FINISHED image, not the request. "
            "MOSS-Image was trained on detailed captions; concrete visible detail beats short prompts.\n"
            "2. Cover subject and action, appearance, setting, composition/camera, lighting, materials/textures, "
            "palette and mood. Keep relationships and spatial placement explicit.\n"
            "3. Exact visible text: put it in double quotes and preserve its original language, punctuation and "
            "capitalization. Use a real newline (\n) where a line break matters. Distinguish text to render from "
            "reference material or constraints.\n"
            "4. For designed images, compose information and imagery together: reading order, relative prominence, "
            "coherent groups, and readable text at the chosen pixels. Write the intended copy, not placeholders.\n"
            "5. If the user supplied booru-style tags, preserve the official order: Quality -> Score -> Meta -> "
            "Year -> Character -> Series -> Artist -> General. Otherwise prefer cohesive prose.\n"
            "6. Layout JSON with bounding boxes is only for explicit layout requests; do not invent coordinates "
            "in a normal caption. The app Virtual Boxes / Layout tool supplies the real geometry.\n"
            "7. Do not invent unrelated names, watermarks or facts. Keep an explicitly minimalist brief minimalist.\n"
            "8. Do not put a negative-prompt list inside the positive prompt; MOSS has its own short negative profile.\n"
            "STRICTLY FORBIDDEN: tag soup when prose was requested, generic quality buzzwords, invisible process "
            "notes, and fake bounding boxes."
        ),
    },
    "fibo2": {
        "label": "FIBO 2.0 Edit Turbo (Qwen3-VL VGL Dialect)",
        "length": "60 to 120 words",
        "max_words": 120,
        "instructions": (
            "Target Engine: FIBO 2.0 Edit Turbo (Qwen3-VL-4B Multimodal Text Encoder; 4-step distilled DiT).\n"
            "Official prompting style (Bria FIBO 2.0 Visual Grounding Language) — STRUCTURED DESCRIPTIVE SCENE BRIEF:\n"
            "1. Write clear, detailed descriptive prose covering Subject, Appearance, Lighting, Setting, and Camera/Composition.\n"
            "2. FIBO 2.0 uses Qwen3-VL multimodal understanding: it excels at precise spatial relations ('to the left of', 'in the background'), lighting dynamics (soft cinematic golden hour, deep volumetric shadows), and surface materials/textures.\n"
            "3. NO negative prompts exist on this model — specify everything in positive, concrete terms.\n"
            "4. Length: 50 to 120 words is ideal. Focus on vivid physical attributes rather than vague buzzwords.\n"
            "5. In-image text: wrap exact text in double quotes and state font/material and placement.\n"
            "STRICTLY FORBIDDEN: tag-soup lists, generic buzzwords (masterpiece, photorealistic), and negative phrasing."
        ),
    },
    "dreamlite": {
        "label": "DreamLite Mobile 0.39B (Qwen3-VL TE)",
        "length": "35 to 80 words",
        "max_words": 80,
        "max_tokens": 180,
        "instructions": (
            "{target_line}\n"
            "Official prompting style (DreamLite Mobile 0.39B 4-step guidance-free UNet) — FOCUSED & CONCISE SCENE:\n"
            "1. Write 35-80 words of clear, direct English prose describing ONE cohesive scene. "
            "A compact 0.39B 4-step UNet produces much sharper results with focused composition than overloaded multi-clause paragraphs.\n"
            "2. Lead immediately with the main subject, pose/action, and key visual traits, followed by setting, lighting, color palette, and art/photo medium.\n"
            "3. Guidance-free (no negative prompt support): describe everything in positive, concrete visual terms.\n"
            "4. Keep spatial layout simple and legible — avoid crowding too many tiny secondary props.\n"
            "5. If the user asks for visible text, wrap it in double quotes and keep it short and prominent.\n"
            "STRICTLY FORBIDDEN: tag soup, generic quality buzzwords (masterpiece, 8k, ultra-detailed), negative phrasing, and verbose filler."
        ),
    },
    "anima": {
        "label": "Anima Turbo 2B (Qwen3 + T5-XXL Anime)",
        "length": "40 to 100 words",
        "max_words": 100,
        "max_tokens": 220,
        "instructions": (
            "{target_line}\n"
            "Official prompting style (CircleStone Labs Anima 2B — anime/illustration DiT with Qwen3 + T5-XXL) — ANIME SCENE & TAG HYBRID:\n"
            "1. Anima is trained on both Danbooru-style anime tags and natural-language captions. Keep prompts focused (40-100 words) with clear visual hierarchy.\n"
            "2. If the user supplied booru-style tags (e.g. 1girl, character/series/artist tags, quality tags), preserve them verbatim in canonical order (Subject/Count -> Character -> Appearance/Outfit -> Pose/Expression -> Setting -> Lighting/Mood -> Style) and enrich with concrete anime visual details.\n"
            "3. If the user wrote natural prose, expand it into vivid anime/illustration prose covering character design, hair/eye details, expression, clothing folds, framing, cel-shading or painterly rendering, and atmospheric lighting.\n"
            "4. CFG-1 turbo distilled (no negative prompt support): state every desired detail positively.\n"
            "5. Do not force photorealistic camera/lens jargon unless the user explicitly requested photography.\n"
            "STRICTLY FORBIDDEN: negative phrasing inside the positive prompt, contradicting the user's anime/illustration style, and invisible meta-commentary."
        ),
    },
}


def _load_lora_registry() -> List[Dict[str, Any]]:
    global _lora_registry_cache
    registry_file = DATA_DIR / "loras.json"
    if not registry_file.exists():
        _lora_registry_cache = None
        return []
    try:
        st = registry_file.stat()
        if (
            _lora_registry_cache is not None
            and _lora_registry_cache[0] == st.st_mtime_ns
            and _lora_registry_cache[1] == st.st_size
        ):
            return _lora_registry_cache[2]
        data = json.loads(registry_file.read_text(encoding="utf-8"))
        if isinstance(data, list):
            _lora_registry_cache = (st.st_mtime_ns, st.st_size, data)
            return data
    except Exception as e:
        print(f"[prompt_enhancer] warning: failed reading loras.json: {e}", file=sys.stderr)
    return []


def extract_active_lora_info(loras: Optional[List[Dict[str, Any]]]) -> Tuple[List[str], List[str]]:
    """Extract mandatory trigger words and LoRA names from active LoRA list."""
    if not loras:
        return [], []

    registry = _load_lora_registry()
    triggers: List[str] = []
    names: List[str] = []

    for item in loras:
        if not isinstance(item, dict):
            continue
        path_str = str(item.get("path", "")).strip()
        name_str = str(item.get("name", "")).strip()

        # Find matching entry in registry
        match = next(
            (
                r for r in registry
                if (path_str and r.get("path") == path_str)
                or (name_str and r.get("name") == name_str)
                or (path_str and Path(path_str).name == Path(str(r.get("path", ""))).name)
            ),
            None,
        )

        # Collect name
        display_name = name_str or (match.get("name") if match else "") or (Path(path_str).stem if path_str else "")
        if display_name and display_name not in names:
            names.append(display_name)

        # Collect triggers from registry or payload
        raw_triggers = []
        if match and match.get("triggers"):
            raw_triggers.extend(match["triggers"])
        if item.get("triggers"):
            raw_triggers.extend(item["triggers"])

        for t in raw_triggers:
            if isinstance(t, str):
                # Clean and split compound trigger lines if formatted as comma lists
                sub_triggers = [s.strip() for s in t.split(",") if s.strip()]
                for st in sub_triggers:
                    # Filter out generic noise words
                    if st and st not in triggers and len(st) > 1:
                        triggers.append(st)

    return triggers, names


def _get_model(cancel_event: Optional[threading.Event] = None, allow_downloads: bool = False):
    global _model, _tokenizer, _loaded_repo
    if cancel_event is not None and cancel_event.is_set():
        raise EnhancementCancelled()
    repo = resolve_model_repo()
    if _model is None or _loaded_repo != repo:
        with _model_lock:
            if _model is None or _loaded_repo != repo:
                if not allow_downloads and not model_cached(selected_model_id()):
                    raise RuntimeError(
                        "Prompt enhancer model is not downloaded. Download it in "
                        "Parameters → Prompt Enhancer, or tick Allow required downloads in Generate → Advanced Settings."
                    )
                from mlx_lm import load
                print(f"[prompt_enhancer] loading model {repo}...", file=sys.stderr)
                _model, _tokenizer = load(repo)
                _loaded_repo = repo
                print("[prompt_enhancer] model loaded.", file=sys.stderr)
    if cancel_event is not None and cancel_event.is_set():
        raise EnhancementCancelled()
    return _model, _tokenizer


def _make_cancel_processor(cancel_event: threading.Event):
    """Logits processor that aborts generation the moment the event is set.

    mlx_lm invokes processors once per token, so raising here stops generation
    promptly and cooperatively (no need to wait for the full token budget)."""
    def _cancel_processor(_tokens, logits):
        if cancel_event.is_set():
            raise EnhancementCancelled()
        return logits
    return _cancel_processor


def _normalize_engine(engine: str) -> str:
    engine_key = "flux2"
    engine_lower = (engine or "").lower()
    if "flux3" in engine_lower or "flux.3" in engine_lower or "flux-3" in engine_lower or "flux_3" in engine_lower:
        engine_key = "flux3"
    elif (
        "flux1" in engine_lower
        or "flux.1" in engine_lower
        or "schnell" in engine_lower
    ):
        engine_key = "flux1"
    elif "sdxl" in engine_lower or "juggernaut" in engine_lower:
        engine_key = "sdxl"
    elif "krea" in engine_lower:
        engine_key = "krea2"
    elif "z-image" in engine_lower or "zit" in engine_lower:
        engine_key = "z-image-turbo"
    elif "fibo" in engine_lower:
        engine_key = "fibo2"
    elif "ming" in engine_lower:
        engine_key = "ming"
    elif "qwen" in engine_lower:
        engine_key = "qwen"
    elif "moss" in engine_lower:
        engine_key = "moss"
    elif "dreamlite" in engine_lower or "dream lite" in engine_lower:
        engine_key = "dreamlite"
    elif "anima" in engine_lower:
        engine_key = "anima"
    return engine_key


def _schema_block() -> str:
    lines = [f'    "{k}": "{v}"' for k, v in JSON_PROMPT_SCHEMA.items()]
    return "{\n" + ",\n".join(lines) + "\n}"


def _trigger_block(mandatory_triggers: List[str], lora_names: List[str]) -> str:
    trigger_bullets = "\n".join(f'- "{t}"' for t in mandatory_triggers)
    lora_context = f" (Active LoRA styles: {', '.join(lora_names)})" if lora_names else ""
    return (
        f"MANDATORY LORA TRIGGER PRESERVATION{lora_context}:\n"
        f"The following activation trigger phrase(s) MUST be included VERBATIM in your enhanced prompt:\n"
        f"{trigger_bullets}\n\n"
        "TRIGGER RULES:\n"
        "1. NEVER omit, translate, rephrase, or abbreviate any trigger phrase.\n"
        "2. Weave the trigger phrase naturally into the opening subject description.\n"
        "3. Adapt the surrounding lighting, materials, and environment to harmonize with the LoRA aesthetic."
    )


def _json_task_block() -> str:
    return (
        "TASK:\n"
        f"Rewrite and expand the user concept into a single STRICT JSON object using EXACTLY these keys "
        f"(no other keys, no nested structures):\n{_schema_block()}\n\n"
        "JSON RULES:\n"
        "1. Each key maps to ONE concise, vivid sentence (skip/shorten only when the concept gives no "
        "material; use an empty string \"\" for fields with no content).\n"
        "2. EVERY value MUST be a single plain string. NEVER use objects, arrays, lists, or nested "
        "{} for any value — even the 'details' or 'technical' keys.\n"
        "3. Fill the structured fields in a camera-first order (setting/lighting/atmosphere/composition "
        "before camera technicals) so the result reads like a scene brief.\n"
        "4. text_elements: exact in-image text in double quotes, plus font and placement when the concept "
        "implies text.\n"
        "5. trigger_word: put every mandatory trigger phrase here VERBATIM, comma-separated; also weave the "
        "main trigger naturally into 'subject'.\n"
        "6. Keep every value short (under 18 words) and the WHOLE JSON under 150 words. Output ONLY the "
        "raw JSON object. No markdown fences, no commentary, no preamble.\n"
        "7. Use double quotes for all keys/strings and escape any inner double quotes. End with a closing "
        "curly brace — do not truncate."
    )


def _text_task_block(profile: Dict[str, Any]) -> str:
    return (
        "TASK:\n"
        f"Rewrite and expand the user concept into a rich, descriptive image diffusion prompt "
        f"({profile['length']}).\n"
        "Do NOT just repeat or capitalize the user concept as-is. Elaborate on the lighting, environment, "
        "textures, and composition.\n\n"
        "STRONG OUTPUT RULES (NEVER violate — your response goes straight into the image engine):\n"
        "1. Your ENTIRE response is the enhanced prompt and NOTHING else. One continuous block. No preamble "
        "('Here is...', 'Enhanced prompt:'), no explanation, no commentary, no apology, no JSON, no bullets, "
        "no quotes or markdown around the whole prompt, and no closing remark.\n"
        f"2. HARD LENGTH CAP: at most {profile['max_words']} words. Stop the instant the prompt is complete — "
        "never pad, never continue, never summarize. A focused prompt beats a padded one.\n"
        "3. Preserve the core subject, action, and intent of the original user prompt without replacing it "
        "or inventing new subjects/props/characters unless clearly implied.\n"
        "4. If the user already wrote a detailed prompt, polish and finalize lightly instead of heavily "
        "expanding — respect their phrasing and direction.\n"
        "5. Keep the output cohesive, vivid, and free of tag repetition."
    )


def default_json_instructions(engine_key: str) -> str:
    """Built-in JSON-mode instruction body: engine guidance + schema structure +
    the fill-in guidelines. This is what the Settings editor shows by default for
    JSON mode so the user can see and edit the exact JSON contract."""
    if engine_key == "ming":
        return MING_JSON_INSTRUCTIONS
    profile = ENGINE_PROFILES.get(engine_key, ENGINE_PROFILES["flux2"])
    return profile["instructions"] + "\n\n" + _json_task_block()


def _build_system_prompt(
    engine_key: str,
    output_format: str,
    mandatory_triggers: List[str],
    lora_names: List[str],
    custom_instructions: Optional[str] = None,
    model_id: str = "",
) -> str:
    profile = ENGINE_PROFILES.get(engine_key, ENGINE_PROFILES["flux2"])
    header = "You are an expert AI prompt engineer specializing in state-of-the-art image diffusion models."
    target_line = _engine_target_line(engine_key, model_id or "")
    trigger = _trigger_block(mandatory_triggers, lora_names) if mandatory_triggers else ""

    if output_format == "json":
        # The JSON override is the complete JSON instruction body (guidance +
        # structure + fill rules), so it is used verbatim instead of appending a
        # fixed task block.
        if custom_instructions is None:
            custom_instructions = app_settings.prompt_enhancer_json_override(engine_key)
        body_source = (custom_instructions or "").strip() or default_json_instructions(engine_key)
        body = body_source.replace("{target_line}", target_line)
        parts = [header, body]
        if trigger:
            parts.append(trigger)
        return "\n\n".join(parts)

    # The profile is stored with a {target_line} placeholder rather than a fixed size,
    # so a 9B request is not described to the model as a 4B one.
    if custom_instructions is None:
        custom_instructions = app_settings.prompt_enhancer_override(engine_key)
    engine_guidance_source = (custom_instructions or "").strip() or profile["instructions"]
    engine_guidance = engine_guidance_source.replace("{target_line}", target_line)

    parts = [header, engine_guidance]
    if trigger:
        parts.append(trigger)
    parts.append(_text_task_block(profile))
    return "\n\n".join(parts)


def list_engine_system_prompts() -> List[Dict[str, Any]]:
    """Per-engine prompt-enhancer metadata for the Settings > Enhancement UI.

    Each entry exposes the built-in engine guidance, the user's custom override
    (if any) and the fully-assembled system prompt for both text and JSON output
    so the UI can preview exactly what the local LLM receives."""
    out: List[Dict[str, Any]] = []
    for key, profile in ENGINE_PROFILES.items():
        custom = app_settings.prompt_enhancer_override(key)
        custom_json = app_settings.prompt_enhancer_json_override(key)
        out.append(
            {
                "key": key,
                "label": profile["label"],
                "length": profile["length"],
                "default_instructions": profile["instructions"],
                "custom_instructions": custom,
                "is_custom": bool(custom.strip()),
                "default_json_instructions": default_json_instructions(key),
                "custom_json_instructions": custom_json,
                "is_json_custom": bool(custom_json.strip()),
                "system_prompt": _build_system_prompt(key, "text", [], [], custom_instructions=custom),
                "default_system_prompt": _build_system_prompt(key, "text", [], [], custom_instructions=""),
                "json_system_prompt": _build_system_prompt(key, "json", [], [], custom_instructions=custom_json),
                "default_json_system_prompt": _build_system_prompt(key, "json", [], [], custom_instructions=""),
            }
        )
    return out


def _chat_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return (
        f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
        f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _clean_output(raw: str) -> str:
    out = re.sub(r"^(?:Enhanced image prompt|Enhanced prompt|Prompt):\s*", "", raw, flags=re.IGNORECASE)
    return out.strip().strip('"').strip("'").strip("`").strip()


# Phrases that appear in the instructions we send and must therefore never come back.
#
# Sourced from the live profiles below, not transcribed: a list written by copying the
# prompt drifts from it silently, and a leak detector that only knows last month's
# wording catches nothing. `_leak_markers()` is built from ENGINE_PROFILES at import so
# the two cannot disagree.



def _looks_like_prompt_leak(text: str) -> bool:
    """Is `text` the system prompt coming back instead of an answer?

    True when the output quotes our own instructions. On a 0.5B instruct model this
    happens routinely under a long instruction block, and it is indistinguishable from a
    good answer to the user -- so it is detected here rather than in the textarea.

    Two detection styles, because the observed leak was not a verbatim quote. The real
    output opened

        ## FLUX.2, Klein 4B (mflux/MLX; guidance-distilled flow-matching DiT ...

    which contains none of LEAK_MARKERS: the model had dropped the "Target Engine:"
    prefix and promoted the rest to a markdown heading. A literal-only detector reports
    that as a clean prompt, which is precisely the case this exists to catch. So beyond
    the literal phrases, a line that names the runtime AND the architecture
    ("flux.2"/"mflux" plus "flow-matching DiT"/"T5 text encoder") is treated as a leak
    too -- that pairing is what our own header says and no image prompt has any reason
    to contain it.
    """
    if not text:
        return False
    if any(marker in text for marker in LEAK_MARKERS):
        return True
    lowered = text.lower()
    names_engine = any(t in lowered for t in ("flux.2", "flux2", "klein", "mflux"))
    names_architecture = (
        any(t in lowered for t in ("flow-matching", "flow matching", "t5 text encoder"))
        or re.search(r"\bdit\b", lowered) is not None
    )
    return names_engine and names_architecture


def _ends_mid_word(text: str) -> bool:
    """Did generation stop mid-token?

    The budget is a token count, not a sentence count, so it lands wherever it lands.
    The tell is a trailing fragment with no terminal punctuation, and -- more reliably --
    a last 'word' glued to the previous one, which is what a cut mid-token looks like
    ("...to the pendant. TheLOG").
    """
    stripped = text.rstrip()
    if not stripped:
        return False
    if stripped[-1] in ".!?\"')]}:;,—-":
        return False
    # In-image text commonly ENDS the prompt on a quote: 'a sign reading "OPEN"'. That
    # is a finished prompt, not a cut, even though there is no full stop.
    if stripped[-1] == '"' and stripped.count('"') % 2 == 0:
        return False
    # A space-separated final fragment is still a word; a run-on glued to a previous
    # word is a cut. Uppercase mid-run is the usual giveaway but not the only one, so
    # the conservative rule is: unpunctuated tail that does not end a clean word.
    return True


def _trim_to_complete_sentence(text: str, mandatory_triggers: Optional[List[str]] = None) -> str:
    """Cut a truncated prompt back to its last finished sentence.

    Only does anything when the text is genuinely cut. An empty result is worse than a
    rough one -- the user would see an empty box and assume the app broke -- so text
    with no sentence boundary is returned unchanged.
    """
    stripped = text.rstrip()
    if not _ends_mid_word(stripped):
        return stripped
    matches = list(re.finditer(r"[.!?][\"')]?(?=\s|$)", stripped))
    if not matches:
        return stripped
    trimmed = stripped[: matches[-1].end()].strip()
    # A trim that removes a trigger changes what the LoRA does; put it back rather
    # than quietly handing the model an adapter with no activation word.
    if mandatory_triggers:
        for trigger in mandatory_triggers:
            if trigger and trigger in text and trigger not in trimmed:
                trimmed = f"{trigger}, {trimmed}"
                break
    return trimmed


def _engine_target_line(engine_key: str, model_id: str = "") -> str:
    """The 'Target Engine' line, naming the size the user actually selected.

    It used to say "FLUX.2 Klein 4B" unconditionally, so a 9B request was handed to the
    model as a 4B one -- and that line then surfaced verbatim in leaked output.
    """
    label = ENGINE_PROFILES.get(engine_key, ENGINE_PROFILES["flux2"]).get("label", "FLUX.2")
    size = ""
    if "9b" in (model_id or "").lower():
        size = " 9B"
    elif "4b" in (model_id or "").lower():
        size = " 4B"
    if engine_key == "moss":
        return (
            f"Target Engine: {label}{size} (mflux/MLX; 5B single-stream DiT with a "
            "Qwen3.5 text encoder)."
        )
    if engine_key == "flux3":
        return (
            f"Target Engine: {label}{size} (mflux/MLX; distilled flow-matching DiT)."
        )
    if engine_key == "sdxl":
        return (
            f"Target Engine: {label}{size} (mflux/MLX; dual CLIP text encoders; "
            "SDXL UNet)."
        )
    if engine_key == "dreamlite":
        return (
            f"Target Engine: {label}{size} (mflux/MLX; compact 0.39B 4-step "
            "guidance-free UNet with a Qwen3-VL text encoder)."
        )
    if engine_key == "anima":
        return (
            f"Target Engine: {label}{size} (mflux/MLX; 2B anime DiT with "
            "Qwen3 + T5-XXL text encoders)."
        )
    if engine_key == "ming":
        return (
            f"Target Engine: {label}{size} (Swift/MLX; 6B S3-DiT with a "
            "BailingMM2/Ling-mini MoE multimodal text conditioner)."
        )
    return f"Target Engine: {label}{size} (mflux/MLX; guidance-distilled flow-matching DiT with a T5 text encoder)."


def _leak_markers() -> Tuple[str, ...]:
    """Instruction phrases that must never appear in output.

    Derived from the live profiles rather than transcribed, so the detector cannot drift
    from the prompt it guards. `{target_line}` is expanded first: it is a placeholder
    now, and a search for a phrase the profile no longer literally contains would return
    an EMPTY marker set -- which makes every leak look clean. That failure is silent and
    was the first version of this.
    """
    candidates = (
        "STRICTLY FORBIDDEN",
        "attends most to the START",
        "Order elements Main subject",
        "Target Engine:",
    )
    found = {
        phrase
        for profile in ENGINE_PROFILES.values()
        for phrase in candidates
        if phrase in profile["instructions"] or phrase in _engine_target_line("flux2")
    }
    return tuple(sorted(found))


LEAK_MARKERS: Tuple[str, ...] = _leak_markers()

def _json_fallback_prose(raw: str) -> Optional[str]:
    """The text to fall back on when JSON parsing failed, or None to refuse.

    "Fall back to prose" only makes sense if the output really is prose. Output that is
    still recognisably JSON -- an unclosed object, a stray brace, quoted keys -- is a
    broken document, not a prompt, and was being handed over as one. A diffusion model
    has no way to render "{ "subject":" faithfully; it draws braces.
    """
    text = (raw or "").strip()
    if not text:
        return None
    json_shaped = (
        text.count("{") != text.count("}")
        or text.count('"') >= 2
        or re.search(r'"?\w+"?\s*:\s*"?', text) is not None
    )
    return None if json_shaped else text


def _ensure_triggers_present(enhanced: str, mandatory_triggers: List[str]) -> List[str]:
    """Guarantee every mandatory trigger appears; prepend any missing one."""
    preserved: List[str] = []
    for trigger in mandatory_triggers:
        if re.search(re.escape(trigger), enhanced, re.IGNORECASE):
            preserved.append(trigger)
        else:
            enhanced = f"{trigger}, {enhanced}"
            preserved.append(trigger)
    return enhanced, preserved


def _sanitize_json_text(s: str) -> str:
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    return re.sub(r',\s*}', "}", s)


def _normalize_ming_structured_prompt(parts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate and pin inclusionAI's Figma-style Ming prompt schema.

    Ming consumes the JSON as a layout contract, not as generic prompt metadata. A
    partial object with missing coordinate or ownership fields is worse than no
    enhancement because the DiT conditioner still treats it as meaningful layout.
    """
    if set(parts.keys()) - {"canvas_settings", "layers"}:
        parts = {k: parts[k] for k in ("canvas_settings", "layers") if k in parts}
    canvas = parts.get("canvas_settings")
    layers = parts.get("layers")
    if not isinstance(canvas, dict) or not isinstance(layers, list) or not layers:
        return None
    if any(not isinstance(canvas.get(k), str) or not canvas.get(k, "").strip() for k in ("aspect_ratio", "ambient_lighting", "image_style")):
        return None
    normalized_layers = []
    coord_re = re.compile(
        r"^cx:\s*[01](?:\.\d+)?,\s*cy:\s*[01](?:\.\d+)?,\s*w:\s*(?:0?\.\d+|1(?:\.0+)?),\s*h:\s*(?:0?\.\d+|1(?:\.0+)?)$"
    )
    for layer in layers:
        if not isinstance(layer, dict):
            return None
        if any(not isinstance(layer.get(k), str) or not layer.get(k, "").strip() for k in ("description", "coordinates", "hierarchy_and_relation")):
            return None
        colors = layer.get("color_specs")
        if not isinstance(colors, list) or not all(isinstance(c, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", c) for c in colors):
            return None
        coordinates = str(layer["coordinates"]).strip()
        if not coord_re.match(coordinates):
            return None
        values = re.findall(r"(?:cx|cy|w|h):\s*([0-9]*\.?[0-9]+)", coordinates)
        if len(values) != 4:
            return None
        cx, cy, width, height = (float(v) for v in values)
        if (
            cx - width / 2 < -1e-6
            or cx + width / 2 > 1 + 1e-6
            or cy - height / 2 < -1e-6
            or cy + height / 2 > 1 + 1e-6
        ):
            return None
        normalized_layers.append({
            "description": str(layer["description"]).strip(),
            "coordinates": coordinates,
            "hierarchy_and_relation": str(layer["hierarchy_and_relation"]).strip(),
            "color_specs": colors,
        })
    return {
        "canvas_settings": {
            "aspect_ratio": str(canvas["aspect_ratio"]).strip(),
            "ambient_lighting": str(canvas["ambient_lighting"]).strip(),
            "image_style": str(canvas["image_style"]).strip(),
        },
        "layers": normalized_layers,
    }


def _parse_json_output(raw: str) -> Optional[Dict[str, Any]]:
    # Strip code fences the small model sometimes adds. Try the untouched text first:
    # Ming exact copy may legitimately contain curly apostrophes/quotes inside JSON
    # strings, and the compatibility sanitizer must not rewrite those characters.
    unfenced = re.sub(r"^\s*```(?:json)?\s*", "", raw)
    unfenced = re.sub(r"\s*```\s*$", "", unfenced)
    try:
        direct = json.loads(unfenced)
        if isinstance(direct, dict):
            return direct
    except Exception:
        pass

    sani = _sanitize_json_text(unfenced)
    try:
        return json.loads(sani)
    except Exception:
        pass

    # Repair truncation: if braces are unbalanced (opens > closes), pad with closing braces
    # and retry — covers outputs cut off at the token budget.
    if sani.count("{") > sani.count("}"):
        repaired = sani + "}" * (sani.count("{") - sani.count("}"))
        try:
            return json.loads(repaired)
        except Exception:
            pass

    # Scan from each "{" for the first complete, valid JSON object
    # (tolerates prose/junk prefixes and braces inside string values).
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", sani):
        try:
            obj, _ = decoder.raw_decode(sani, m.start())
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _flatten_json(parts: Dict[str, Any]) -> str:
    """Natural-language flattening of the structured schema (Z-Image/FLUX prefer prose)."""
    order = [
        "subject", "appearance", "action", "setting", "lighting",
        "atmosphere", "composition", "details", "text_elements", "technical",
    ]
    sentences = [str(parts[k]).strip() for k in order if str(parts.get(k, "")).strip()]
    tw = str(parts.get("trigger_word", "")).strip()
    if tw and tw not in " ".join(sentences):
        sentences.insert(0, tw)
    return " ".join(sentences).strip() if sentences else ""


def _generate_raw(
    messages: List[Dict[str, str]],
    max_tokens: int,
    output_format: str,
    cancel_event: Optional[threading.Event] = None,
    allow_downloads: bool = False,
) -> str:
    """Run the model and return its raw text.

    A seam, not a refactor for its own sake: the output-quality guards downstream are
    the entire point of this module, and they cannot be tested against a 0.5B model
    whose output varies run to run. Tests need to hand in a known-bad string.
    """
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler, make_logits_processors

    model, tokenizer = _get_model(cancel_event, allow_downloads=allow_downloads)
    prompt_str = _chat_prompt(tokenizer, messages)

    # JSON output needs a more deterministic sampler to stay well-formed.
    if output_format == "json":
        sampler = make_sampler(temp=0.4, top_p=0.85)
    else:
        sampler = make_sampler(temp=0.7, top_p=0.9)
    lproc = make_logits_processors(repetition_penalty=1.2, repetition_context_size=50)
    if cancel_event is not None:
        lproc = list(lproc) + [_make_cancel_processor(cancel_event)]

    return generate(
        model,
        tokenizer,
        prompt=prompt_str,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=lproc,
        verbose=False,
    )


def enhance_prompt(
    prompt: str,
    engine: str = "flux2",
    style: Optional[str] = None,
    loras: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 140,
    output_format: str = "text",
    cancel_event: Optional[threading.Event] = None,
    allow_downloads: bool = False,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> dict:
    """Enhance a user prompt for the given diffusion engine with LoRA trigger preservation.

    output_format: "text" (engine-shaped prose/tags) or "json" (structured schema).
    Returns:
        dict: {
            "original", "enhanced", "engine",
            "preserved_triggers", "active_loras", "elapsed",
            # json mode extras:
            "format": "json", "enhanced_parts": {...}, "enhanced_prose": str,
            "parse_error": bool  (True if the model's JSON could not be parsed and prose fallback was used)
        }
    """
    cleaned_prompt = (prompt or "").strip()
    if not cleaned_prompt:
        return {
            "original": prompt,
            "enhanced": prompt,
            "engine": _normalize_engine(engine),
            "preserved_triggers": [],
            "active_loras": [],
            "elapsed": 0.0,
        }

    engine_key = _normalize_engine(engine)
    max_tokens = max(
        max_tokens,
        int(ENGINE_PROFILES.get(engine_key, {}).get("max_tokens", 0)),
    )
    if output_format == "json":
        # JSON objects are verbose; give the small model enough budget.
        max_tokens = max(max_tokens, 640)

    # Extract active LoRA triggers and identifiers
    mandatory_triggers, lora_names = extract_active_lora_info(loras)

    if style:
        style = str(style).strip()
    else:
        style = None

    system_prompt = _build_system_prompt(engine_key, output_format, mandatory_triggers, lora_names, model_id=engine or "")
    if engine_key == "ming":
        if width and height:
            import math
            divisor = math.gcd(int(width), int(height))
            aspect = f"{int(width)//divisor}:{int(height)//divisor}, {int(width)} × {int(height)} px"
        else:
            aspect = "the aspect ratio requested by the user; use 1:1 when unspecified"
        system_prompt = system_prompt.replace("{aspect_ratio}", aspect)
    style_note = f"\nCreative direction hint to honor if compatible: {style}" if style else ""
    user_msg = f"Concept to expand: \"{cleaned_prompt}\"{style_note}\n\nEnhanced prompt:"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    t0 = time.time()
    try:
        raw_output = _generate_raw(
            messages, max_tokens, output_format, cancel_event, allow_downloads
        )

        enhanced = _clean_output(raw_output)

        # A leak is not a parse error and not a truncation: it parses fine and ends in
        # punctuation, so nothing downstream would object to it. It is the instructions
        # coming back as the answer, and the user cannot tell it from a real prompt.
        if _looks_like_prompt_leak(enhanced):
            return {
                "original": cleaned_prompt,
                "enhanced": "",
                "engine": engine_key,
                "active_loras": lora_names,
                "elapsed": round(time.time() - t0, 3),
                "rejected": True,
                "reason": "prompt_leak",
                "message": "The enhancer returned its own instructions instead of a "
                           "prompt. The prompt was left unchanged.",
            }

        # Cut back to the last finished sentence before anything stores it.
        enhanced = _trim_to_complete_sentence(enhanced, mandatory_triggers)

        result: Dict[str, Any] = {
            "original": cleaned_prompt,
            "engine": engine_key,
            "active_loras": lora_names,
            "elapsed": round(time.time() - t0, 3),
        }

        if output_format == "json":
            parts = _parse_json_output(enhanced)
            if engine_key == "ming" and parts is not None:
                parts = _normalize_ming_structured_prompt(parts)
            result["format"] = "json"
            result["parse_error"] = parts is None
            if parts is not None:
                # Pin the generic schema for normal engines. Ming intentionally keeps
                # inclusionAI's own canvas_settings/layers schema instead.
                if engine_key == "ming":
                    cleaned_parts = _normalize_ming_structured_prompt(parts)
                else:
                    cleaned_parts: Dict[str, Any] = {k: "" for k in JSON_PROMPT_SCHEMA}
                    for k in JSON_PROMPT_SCHEMA:
                        v = parts.get(k)
                        if v is not None:
                            cleaned_parts[k] = str(v).strip() if not isinstance(v, list) else ", ".join(str(x) for x in v)
                if mandatory_triggers:
                    held = cleaned_parts.get("trigger_word", "")
                    cleaned_parts["trigger_word"] = ", ".join(
                        dict.fromkeys([*mandatory_triggers, *(t.strip() for t in held.split(",") if t.strip())])
                    )
                # Keep JSON string compact but escaped for a single-line prompt field.
                result["enhanced"] = json.dumps(cleaned_parts, ensure_ascii=False)
                result["enhanced_parts"] = cleaned_parts
                result["enhanced_prose"] = _flatten_json(cleaned_parts)
                # Triggers are guaranteed inside "trigger_word"; treat as present.
                result["preserved_triggers"] = list(mandatory_triggers)
            else:
                # JSON parse failed. The old fallback shipped the raw output as a text
                # prompt, so unparseable JSON reached the user verbatim and the
                # diffusion model rendered its punctuation and braces as literal
                # characters. Only genuine prose is a valid fallback now; anything that
                # still looks like JSON is refused, because a partial object is worse
                # than no enhancement.
                prose = _json_fallback_prose(enhanced)
                if prose is None:
                    result.update({
                        "enhanced": "",
                        "enhanced_prose": "",
                        "preserved_triggers": [],
                        "rejected": True,
                        "reason": "json_unparseable",
                        "message": "The enhancer returned JSON that could not be "
                                   "parsed. The prompt was left unchanged.",
                    })
                    return result
                enhanced, preserved = _ensure_triggers_present(prose, mandatory_triggers)
                result["enhanced"] = enhanced
                result["enhanced_prose"] = enhanced
                result["preserved_triggers"] = preserved
            return result

        enhanced, preserved = _ensure_triggers_present(enhanced, mandatory_triggers)
        result["enhanced"] = enhanced.strip().strip('"').strip("'").strip("`").strip()
        result["preserved_triggers"] = preserved
        return result

    except EnhancementCancelled:
        raise
    except Exception as e:
        print(f"[prompt_enhancer] ERROR: {e}", file=sys.stderr)
        print("[prompt_enhancer] traceback:\n" + traceback.format_exc(), file=sys.stderr)
        return {
            "original": cleaned_prompt,
            "enhanced": cleaned_prompt,
            "engine": engine_key,
            "preserved_triggers": [],
            "active_loras": lora_names,
            "error": str(e),
            "elapsed": round(time.time() - t0, 3),
        }


def unload_model() -> None:
    """Unload prompt enhancer model to free unified memory for diffusion generation."""
    global _model, _tokenizer, _loaded_repo
    with _model_lock:
        if _model is not None:
            del _model
            del _tokenizer
            _model = None
            _tokenizer = None
            _loaded_repo = None
            import gc
            gc.collect()
            try:
                import mlx.core as mx
                clear = getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", None)
                if clear:
                    clear()
            except Exception:
                pass
            print("[prompt_enhancer] model unloaded to reclaim unified memory", file=sys.stderr)
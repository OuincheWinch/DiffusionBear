"""Engine-Adaptive Prompt Enhancer with Active LoRA Trigger Preservation.

Powered by a lightweight local LLM (mlx-community/Qwen2.5-0.5B-Instruct-4bit)
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
_model_lock = threading.Lock()
_lora_registry_cache: Optional[List[Dict[str, Any]]] = None

DATA_DIR = app_settings.DATA_DIR
ASSET_DIR = app_settings.ASSET_DIR
LOCAL_MODEL_DIR = ASSET_DIR / "models" / "qwen2.5-0.5b-instruct-4bit"

if LOCAL_MODEL_DIR.exists():
    DEFAULT_MODEL_REPO = str(LOCAL_MODEL_DIR)
else:
    DEFAULT_MODEL_REPO = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

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

ENGINE_PROFILES = {
    "flux2": {
        "label": "FLUX.2 Klein (T5 Narrative)",
        "length": "40 to 75 words",
        "max_words": 75,
        "instructions": (
            "Target Engine: FLUX.2 Klein 4B (mflux/MLX; guidance-distilled flow-matching DiT with a T5 text encoder).\n"
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
}


def _load_lora_registry() -> List[Dict[str, Any]]:
    global _lora_registry_cache
    if _lora_registry_cache is not None:
        return _lora_registry_cache
    registry_file = DATA_DIR / "loras.json"
    if registry_file.exists():
        try:
            data = json.loads(registry_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                _lora_registry_cache = data
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


def _get_model(cancel_event: Optional[threading.Event] = None):
    global _model, _tokenizer
    if cancel_event is not None and cancel_event.is_set():
        raise EnhancementCancelled()
    if _model is None:
        with _model_lock:
            if _model is None:
                from mlx_lm import load
                print(f"[prompt_enhancer] loading model {DEFAULT_MODEL_REPO}...", file=sys.stderr)
                _model, _tokenizer = load(DEFAULT_MODEL_REPO)
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
    if "sdxl" in engine_lower or "juggernaut" in engine_lower:
        engine_key = "sdxl"
    elif "krea" in engine_lower:
        engine_key = "krea2"
    elif "z-image" in engine_lower or "zit" in engine_lower:
        engine_key = "z-image-turbo"
    elif "qwen" in engine_lower:
        engine_key = "qwen"
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
    profile = ENGINE_PROFILES.get(engine_key, ENGINE_PROFILES["flux2"])
    return profile["instructions"] + "\n\n" + _json_task_block()


def _build_system_prompt(
    engine_key: str,
    output_format: str,
    mandatory_triggers: List[str],
    lora_names: List[str],
    custom_instructions: Optional[str] = None,
) -> str:
    profile = ENGINE_PROFILES.get(engine_key, ENGINE_PROFILES["flux2"])
    header = "You are an expert AI prompt engineer specializing in state-of-the-art image diffusion models."
    trigger = _trigger_block(mandatory_triggers, lora_names) if mandatory_triggers else ""

    if output_format == "json":
        # The JSON override is the complete JSON instruction body (guidance +
        # structure + fill rules), so it is used verbatim instead of appending a
        # fixed task block.
        if custom_instructions is None:
            custom_instructions = app_settings.prompt_enhancer_json_override(engine_key)
        body = (custom_instructions or "").strip() or default_json_instructions(engine_key)
        parts = [header, body]
        if trigger:
            parts.append(trigger)
        return "\n\n".join(parts)

    if custom_instructions is None:
        custom_instructions = app_settings.prompt_enhancer_override(engine_key)
    engine_guidance = (custom_instructions or "").strip() or profile["instructions"]
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


def _parse_json_output(raw: str) -> Optional[Dict[str, Any]]:
    sani = _sanitize_json_text(raw)
    # Strip code fences the small model sometimes adds.
    sani = re.sub(r"^\s*```(?:json)?\s*", "", sani)
    sani = re.sub(r"\s*```\s*$", "", sani)

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


def enhance_prompt(
    prompt: str,
    engine: str = "flux2",
    style: Optional[str] = None,
    loras: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 140,
    output_format: str = "text",
    cancel_event: Optional[threading.Event] = None,
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
    if output_format == "json":
        # JSON objects are verbose; give the small model enough budget.
        max_tokens = max(max_tokens, 640)

    # Extract active LoRA triggers and identifiers
    mandatory_triggers, lora_names = extract_active_lora_info(loras)

    if style:
        style = str(style).strip()
    else:
        style = None

    system_prompt = _build_system_prompt(engine_key, output_format, mandatory_triggers, lora_names)
    style_note = f"\nCreative direction hint to honor if compatible: {style}" if style else ""
    user_msg = f"Concept to expand: \"{cleaned_prompt}\"{style_note}\n\nEnhanced prompt:"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    t0 = time.time()
    try:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler, make_logits_processors

        model, tokenizer = _get_model(cancel_event)
        prompt_str = _chat_prompt(tokenizer, messages)

        # JSON output needs a more deterministic sampler to stay well-formed.
        if output_format == "json":
            sampler = make_sampler(temp=0.4, top_p=0.85)
        else:
            sampler = make_sampler(temp=0.7, top_p=0.9)
        lproc = make_logits_processors(repetition_penalty=1.2, repetition_context_size=50)
        if cancel_event is not None:
            lproc = list(lproc) + [_make_cancel_processor(cancel_event)]

        raw_output = generate(
            model,
            tokenizer,
            prompt=prompt_str,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=lproc,
            verbose=False,
        )

        enhanced = _clean_output(raw_output)
        result: Dict[str, Any] = {
            "original": cleaned_prompt,
            "engine": engine_key,
            "active_loras": lora_names,
            "elapsed": round(time.time() - t0, 3),
        }

        if output_format == "json":
            parts = _parse_json_output(enhanced)
            result["format"] = "json"
            result["parse_error"] = parts is None
            if parts is not None:
                # Pin the schema: keep only known keys, coerce to strings.
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
                # JSON parse failed: fall back to the raw prose (still valid text prompt).
                enhanced, preserved = _ensure_triggers_present(enhanced, mandatory_triggers)
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
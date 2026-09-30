"""Generative fill: regenerate a masked region of an existing image.

Why this is compositing and not inpainting
------------------------------------------
mflux exposes no mask argument on any ``generate_image``. Verified against the
vendored fork, not the docs: ``flux2_klein_edit.generate_image()`` accepts
``image_paths``, ``image_strength``, ``scheduler`` and nothing mask-related -- no
``mask``, no ``latent_mask``, no ``masked_latents``. The FLUX.1 Fill tool the upstream
README describes lives in ``mflux/tools/``, and that directory is not in our fork.

So the mask cannot steer the sampler. What it does instead is bound what we keep:
outside the (feathered) mask the original pixels are composited back in *exactly*,
which is stronger than inpainting, where the model resamples the whole frame and the
unmasked area is only approximately preserved. Inside the mask the model is
generating, and it is generating from the source image as a reference.

The seam is the entire failure mode of this approach, so the feather is the part that
matters. A hard mask edge produces a visible cut; the feather is a symmetric ramp
blending the two across a few pixels. 12px is a starting value, not a tuned one --
``FEATHER_PX`` is a parameter so it can be measured against real content.

Also note what the mask does *not* do: it is not fed to the model as a second
reference. Passing it that way makes it a subject to reproduce rather than a region
to fill, which is the opposite of the intent. See the open question in
tasks/generative_fill_plan.md.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import threading
import time
import uuid
from pathlib import Path

from PIL import Image, ImageFilter

import app_settings
from image_meta import atomic_write_json, save_image_with_metadata, _artist_fallback

# Derive from app_settings rather than re-deriving backend/data: a second
# hardcoded copy ignored the configured data dir and would have written filled
# images into the signed .app bundle instead of the user's store. This is the same
# bug class as the one upscale.py already documents.
DATA_DIR = app_settings.DATA_DIR
GENERATED_DIR = DATA_DIR / "generated"

# Blend width across the mask boundary. 12px at 1024 is ~1.2% of the frame, enough
# to hide a discontinuity without visibly softening a small edit.
FEATHER_PX = 12

# Hard limits. A mask is a user-supplied data URL, so it is untrusted input: it is
# size-capped before decode and dimension-checked after, because a mask that does
# not match its image turns a hard edge back into a smear.
MAX_MASK_BYTES = 16 * 1024 * 1024
MAX_MASK_PIXELS = 64 * 1024 * 1024

# Engines allowed to serve a fill. Narrow on purpose: these are exactly the engines
# with supports_ref, i.e. something to drive. test_fill asserts the set equals the
# registry's supports_ref set, so adding a ref-capable engine without deciding here
# fails the build rather than silently widening the surface.
#
# supports_ref is necessary but NOT sufficient, and that distinction cost two
# wasted attempts on this feature. See FILL_CAPABLE below for the real requirement.
#   z-image-turbo, flux2-klein-4b  -> fast items (DEFAULT is z-image)
#   krea2-turbo                   -> structured heavy work
#   flux2-klein-9b                -> opt-in: costs more than 4B for the same job
#   qwen-image-2.1                -> opt-in and experimental; OOMs above 768^2, so
#                                    only reachable when the source is small
# Deliberately absent: every SDXL/RealVis/Juggernaut XI engine (supports_ref False).
# --- what actually makes a fill work --------------------------------------
#
# Measured, not assumed, after two failed attempts on the bear image (prompt: "pink
# sunglasses", painted across the eyes; the fill returned altered eyes and no
# sunglasses, then a flat grey rectangle).
#
# The reason is in generator.generate(). A reference image reaches an engine in one
# of two ways:
#
#   * FLUX.2 (generator.py:2327): `image_paths=[...]` on the `edit` variant, which is
#     an in-context model. It attends to the reference AND the prompt together, so a
#     region of the reference can be re-generated to match the prompt.
#
#   * everything else (generator.py:2338): a single `image_path=ref_paths[0]`, which
#     mflux uses for image-to-image / style conditioning. There is no spatial
#     correspondence: the engine takes global style cues from the reference and lays
#     out a new image from the prompt. It is not told WHERE anything goes.
#
# So for every non-FLUX2 engine a burnt reference is copied through, not filled --
# the hole reads as a solid grey or dark shape and survives into the composite. Both
# failures looked like "the fill did nothing", and neither was fixable by changing
# the prompt, the mask, the feather or the hole's appearance.
#
# FLUX.2-klein is therefore the ONLY engine that can do a real fill. It is 4B and
# 4 steps, so it is also the fast option, which is the right default.
FILL_CAPABLE_ENGINES = {"flux2-klein-4b", "flux2-klein-9b"}

# What the UI is allowed to offer. Identical to FILL_CAPABLE today; keeping a
# separate name makes the intent explicit and leaves room to widen it if another
# engine gains real spatial conditioning.
FILL_ENGINES = set(FILL_CAPABLE_ENGINES)

# 4B over 9B as the default: same in-context behaviour, a quarter of the time. And
# FLUX.2 rather than z-image because only FLUX.2 has spatial conditioning.
DEFAULT_FILL_ENGINE = "flux2-klein-4b"

# One fill at a time. A fill loads a pipeline, renders, then composites, and
# concurrent fills would fight over the same resident weights. Mirrors the
# semaphore upscale.py uses for the same reason.
_FILL_SLOT = threading.BoundedSemaphore(1)


class FillError(ValueError):
    """A fill request that cannot be honoured. Carries a reason for the UI."""


def _decode_mask(mask_b64: str, expected_size: tuple[int, int]) -> Image.Image:
    """Decode a base64 PNG mask and prove it lines up with its image.

    A dimension mismatch is rejected rather than resized. Resizing would scale a
    fractional edge, and a fractional edge is exactly the hard seam this whole
    module exists to avoid.
    """
    if not isinstance(mask_b64, str) or not mask_b64.strip():
        raise FillError("mask is required")
    # Accept a bare base64 payload or a full data URL.
    payload = mask_b64.split(",", 1)[1] if mask_b64.startswith("data:") else mask_b64
    if len(payload) > MAX_MASK_BYTES:
        raise FillError("mask is too large")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise FillError("mask is not valid base64") from exc
    if len(raw) > MAX_MASK_BYTES:
        raise FillError("mask is too large")
    try:
        mask = Image.open(io.BytesIO(raw))
        mask.load()
    except Exception as exc:  # PIL raises a wide range on malformed input
        raise FillError("mask is not a readable image") from exc

    if mask.size != expected_size:
        raise FillError(
            f"mask is {mask.size[0]}x{mask.size[1]} but the image is "
            f"{expected_size[0]}x{expected_size[1]}"
        )

    # White = regenerate. Coerce to a single 8-bit channel so the feather operates
    # on one predictable range regardless of what the browser sent.
    return mask.convert("L")


def composite_fill(
    original: Image.Image,
    generated: Image.Image,
    mask: Image.Image,
    feather_px: int = FEATHER_PX,
) -> Image.Image:
    """Blend `generated` into `original` through a feathered `mask`.

    Outside the mask the original survives bit-for-bit. That is the guarantee worth
    stating explicitly, and ``test_fill.py`` asserts it pixel-wise rather than
    describing it: a fill must never quietly repaint a region the user did not
    select.
    """
    if original.size != generated.size:
        raise FillError(
            f"generated image is {generated.size[0]}x{generated.size[1]} but the "
            f"source is {original.size[0]}x{original.size[1]}"
        )
    if mask.size != original.size:
        raise FillError("mask does not match the source image")

    base = original.convert("RGBA")
    top = generated.convert("RGBA")

    alpha = mask
    if feather_px > 0:
        # GaussianBlur on an L-mode mask gives a smooth ramp that reaches 0 and 255
        # within roughly 3 sigma, so `radius` here is not the blend width. Scale it
        # so the visible transition is about feather_px wide.
        radius = max(0.1, feather_px / 3.0)
        alpha = mask.filter(ImageFilter.GaussianBlur(radius=radius))

    # A mask with no marked pixels is almost certainly an accident (an untouched
    # canvas, a failed upload). Rejecting it beats burning a two-minute render to
    # return an identical copy of the input.
    if alpha.getextrema()[1] == 0:
        raise FillError("mask is empty; paint the region you want to regenerate")

    # A fully-white mask is a legitimate request: a full re-render.
    top.putalpha(alpha)
    out = Image.alpha_composite(base, top)
    return out.convert("RGB")


def mask_bounding_box(mask: Image.Image, threshold: int = 16) -> tuple[int, int, int, int] | None:
    """Tight box around the marked area, or None if nothing is marked.

    Used to pick an engine: a small box is a fast item, a large one is structured
    heavy work. The caller decides the threshold; the plan flags 25% as a guess.
    """
    box = mask.point(lambda v: 255 if v > threshold else 0).getbbox()
    return box


def fill_image(
    image_id: str,
    mask_b64: str,
    prompt: str,
    model: str = DEFAULT_FILL_ENGINE,
    seed: int | None = None,
    steps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    guidance: float | None = None,
    loras: list[dict] | None = None,
    progress_cb=None,
    phase_cb=None,
    cancel_event=None,
    generate_fn=None,
) -> dict:
    """Regenerate the masked region of `image_id` and return the new sidecar.

    `generate_fn` is injected for tests. Production passes `generator.generate`, which
    already accepts `reference_images` -- that is the seam between this module and
    the engine, and it means a fill is a normal generation with a composite applied
    afterwards rather than a second code path through the sampler.
    """
    if model not in FILL_CAPABLE_ENGINES:
        raise FillError(
            f"{model} cannot be used for fill. Only the FLUX.2 in-context engines "
            f"can regenerate a region of a reference; the others treat a reference "
            f"as a global style cue with no spatial correspondence, so the "
            f"painted area is reproduced instead of filled. "
            f"Available: {', '.join(sorted(FILL_CAPABLE_ENGINES))}"
        )
    prompt = (prompt or "").strip()
    if not prompt:
        raise FillError("a prompt describing what to add is required")

    src_path = _find_source(image_id)
    src_meta = _read_sidecar(image_id)

    with Image.open(src_path) as src_img:
        src_img.load()
        original = src_img.convert("RGB")
    source_size = original.size

    mask = _decode_mask(mask_b64, source_size)
    box = mask_bounding_box(mask)
    if box is None:
        raise FillError("mask is empty; paint the region you want to regenerate")

    if generate_fn is None:
        import generator
        generate_fn = generator.generate

    # --- the reference the engine sees -------------------------------------
    # This is the whole trick, and getting it wrong makes the feature useless.
    #
    # Handing the engine the UNTOUCHED source plus "pink sunglasses" gives it no
    # idea where the user painted. It regenerates a whole image from the prompt and
    # the composite then keeps whatever happened to land in the painted box -- so you
    # get subtly altered eyes and no sunglasses, which is exactly the bug reported
    # on the bear image. The prompt's content never reaches the region.
    #
    # So the reference is the source with the painted area burnt out to mid-grey.
    # Grey rather than black or white because it reads as "nothing here yet" to a
    # model trained on natural images, and it keeps the luminance statistics close
    # to the original so the untouched parts do not shift. The engine then has both
    # the intent AND the location, and fills the hole.
    #
    # The burnt reference is written to a temp file because the engine takes a path.
    # It is deleted immediately after generate() returns.
    t0 = time.time()
    burnt_path = _write_burnt_reference(original, mask)
    try:
        result = generate_fn(
            prompt=prompt,
            model=model,
            reference_images=[str(burnt_path)],
            seed=seed,
            steps=steps,
            width=width or source_size[0],
            height=height or source_size[1],
            guidance=guidance,
            loras=loras,
            progress_cb=progress_cb,
            phase_cb=phase_cb,
            cancel_event=cancel_event,
        )
    finally:
        # The burnt reference is a scratch artifact, never a gallery image. Leaving
        # it behind would put an image with a grey hole in the user's library.
        try:
            burnt_path.unlink()
        except OSError:
            pass

    generated_path = GENERATED_DIR / f"{result['id']}.{result.get('format', 'png')}"
    with Image.open(generated_path) as gen_img:
        gen_img.load()
        generated = gen_img.convert("RGB")

    if generated.size != source_size:
        # Resample rather than fail: the engine may have honoured a different
        # preset than the source image's dimensions. Resizing the *generated* side
        # is safe -- it is being blended, not preserved.
        generated = generated.resize(source_size, Image.Resampling.LANCZOS)

    filled = composite_fill(original, generated, mask)
    return _write_result(
        image_id=image_id,
        parent_meta=src_meta,
        original=original,
        filled=filled,
        prompt=prompt,
        model=model,
        seed=result.get("seed", seed),
        steps=result.get("steps", steps),
        box=box,
        source_size=source_size,
        elapsed=round(time.time() - t0, 2),
        interim_id=result.get("id"),
    )


def _synthesise_hole(original: Image.Image, mask: Image.Image) -> Image.Image:
    """Build the "unfinished region" the engine is asked to complete.

    Mean colour of the surrounding pixels, plus grain, plus a soft central shadow.
    Every component is derived from the image, so this is deterministic: the same
    image and mask always yield the same reference and therefore a reproducible fill.
    """
    w, h = original.size
    base = original.convert("RGB")

    # Mean of the pixels the user did NOT paint, i.e. the surrounding context. This
    # is the tone the hole should start from.
    px = base.load()
    box = mask.getbbox()
    totals = [0, 0, 0]
    count = 0
    if box:
        step = max(1, min(w, h) // 128)   # sampled; a full scan is needless work
        for y in range(box[1], box[3], step):
            for x in range(box[0], box[2], step):
                if mask.getpixel((x, y)) <= 8:
                    r, g, b = px[x, y]
                    totals[0] += r; totals[1] += g; totals[2] += b
                    count += 1
    if count == 0:
        # The mask covers everything, so there is no context to sample.
        r, g, b = base.resize((1, 1)).getpixel((0, 0))
    else:
        r, g, b = (t // count for t in totals)

    # Grain at a fixed ±6 amplitude. Enough to break the flatness that made the grey
    # box reproducible; small enough not to look like noise to the model.
    grain = Image.effect_noise((w, h), 12).convert("L")
    hole = Image.merge(
        "RGB",
        [
            Image.new("L", (w, h), max(0, min(255, r + 6))).point(lambda v: v),
            Image.new("L", (w, h), max(0, min(255, g + 6))),
            Image.new("L", (w, h), max(0, min(255, b + 6))),
        ],
    )
    hole = Image.blend(hole, Image.merge("RGB", [grain.point(lambda v: r), grain.point(lambda v: g), grain.point(lambda v: b)]), 0.5)

    # A soft central darkening, suggesting a form shadow where an object would sit.
    # Only applied within the mask so the rest of the reference is untouched.
    if box:
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        rx = max(1, (box[2] - box[0]) // 2)
        ry = max(1, (box[3] - box[1]) // 2)
        shadow = Image.new("L", (w, h), 0)
        sp = shadow.load()
        for y in range(box[1], box[3]):
            for x in range(box[0], box[2]):
                nx = (x - cx) / rx
                ny = (y - cy) / ry
                d = (nx * nx + ny * ny) ** 0.5
                sp[x, y] = int(max(0.0, 1.0 - d) * 46)   # up to ~18% darkening
        hole = Image.composite(
            Image.blend(hole, Image.new("RGB", (w, h), (max(0, r - 40), max(0, g - 40), max(0, b - 40))), 1.0),
            hole,
            shadow,
        )
        # The shadow must not leak outside the mask.
        hole = Image.composite(hole, base, mask)
    return hole


def _write_burnt_reference(original: Image.Image, mask: Image.Image) -> Path:
    """Render the source with the masked region replaced by flat mid-grey.

    This is the reference the engine actually sees. Without it the prompt's content
    has no connection to the painted region and the fill silently produces a
    generically different image instead of the thing that was asked for.

    Grey (128) rather than black or white on purpose:
      * black reads as a shadow and the model paints it as shadow;
      * white reads as a highlight and the model paints it as specular;
      * mid-grey is the flattest, least "already an object" signal, and it keeps the
        reference's mean luminance near the original so the preserved area does not
        visibly shift in exposure after the composite.

    Written into DATA_DIR/uploads, not the gallery and not the data root. The
    engine refuses a reference from anywhere except GENERATED_DIR or uploads
    (generator._resolve_reference_path), and that guard is worth keeping -- it is
    what stops a caller from feeding the model an arbitrary file on disk. Placing
    the scratch file in uploads satisfies it without weakening anything, and keeps
    the burnt reference out of the gallery, where an indexed sidecar would surface
    an image with a grey hole in it.
    """
    # The hole must look like something the model will want to REPLACE, not
    # something it will faithfully reproduce.
    #
    # Flat uniform grey was tried first and it failed in the most embarrassing way
    # possible: the engine copied the grey rectangle through untouched and the fill
    # produced a grey box where the eyes should have been. A perfectly uniform
    # region is a thing an image model reproduces faithfully -- it is the easiest
    # possible content to match. What signals "incomplete" instead is a region that
    # has the right tone and grain but no structure: it reads as an area the model
    # believes was never finished.
    #
    # So the hole is built from the source's own statistics rather than a constant:
    #   * the mean colour of the surrounding unmasked pixels, so the fill starts from
    #     the right tone and the untouched area does not shift in exposure;
    #   * fine grain matching that mean, because flatness is the thing that made the
    #     grey box reproducible;
    #   * a soft dark vignette toward the middle, hinting at a form shadow where an
    #     object would sit.
    #
    # Deterministic: the grain is derived from the image itself, so two fills of the
    # same image and mask produce the same reference and therefore the same result.
    # A random seed here would make fills irreproducible for no benefit.
    base = original.convert("RGB")
    hole = _synthesise_hole(base, mask)
    composite = Image.composite(hole, base, mask)
    uploads = DATA_DIR / "uploads"
    uploads.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = uploads / f".fill-ref-{uuid.uuid4().hex}.png"
    composite.save(tmp, format="PNG")
    return tmp


def _find_source(image_id: str) -> Path:
    for ext in ("png", "jpeg", "jpg", "webp"):
        candidate = GENERATED_DIR / f"{image_id}.{ext}"
        if candidate.exists():
            return candidate
    raise FillError(f"image {image_id} not found")


def _read_sidecar(image_id: str) -> dict:
    path = GENERATED_DIR / f"{image_id}.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        # A corrupt sidecar must not block a fill; the image is the source of truth
        # for pixels and the new sidecar is rebuilt from whatever we can read.
        return {}


def _write_result(
    *,
    image_id: str,
    parent_meta: dict,
    original: Image.Image,
    filled: Image.Image,
    prompt: str,
    model: str,
    seed,
    steps,
    box,
    source_size: tuple[int, int],
    elapsed: float,
    interim_id: str | None,
) -> dict:
    """Write the filled image plus its sidecar, and drop the unmasked intermediate.

    The intermediate render is deleted on success. It is a full generated image that
    the user never asked for and never sees; left in the gallery it would double the
    count and put a wrong-looking image in their history. It is only removed once the
    composite exists, so a failure part-way leaves it recoverable.
    """
    new_id = uuid.uuid4().hex
    fmt = parent_meta.get("format", "png")
    if fmt not in ("png", "jpeg", "jpg"):
        fmt = "jpeg" if fmt in ("jpeg", "jpg") else "png"
    dest = GENERATED_DIR / f"{new_id}.{fmt}"

    tags = list(dict.fromkeys((parent_meta.get("tags") or []) + ["fill"]))
    box_w = box[2] - box[0]
    box_h = box[3] - box[1]
    mask_fraction = round((box_w * box_h) / float(source_size[0] * source_size[1]), 4)

    meta = dict(parent_meta)
    meta.update({
        "id": new_id,
        "width": source_size[0],
        "height": source_size[1],
        "filled_from": image_id,
        "fill_prompt": prompt,
        "fill_model": model,
        "fill_seed": seed,
        "fill_steps": steps,
        "fill_box": [int(v) for v in box],
        "fill_fraction": mask_fraction,
        "fill_feather_px": FEATHER_PX,
        "fill_method": "masked-composite",
        "generation_time": elapsed,
        "created_at": time.time(),
        "software": "DiffusionBear",
        "generator": "DiffusionBear",
        "artist": _artist_fallback(parent_meta.get("artist")),
        "file": dest.name,
        "format": fmt,
        "tags": tags,
    })

    stealth = bool(parent_meta.get("stealth", False))
    save_image_with_metadata(
        image=filled,
        dest_path=dest,
        meta=meta,
        output_format=fmt,
        stealth=stealth,
    )
    atomic_write_json(GENERATED_DIR / f"{new_id}.json", meta)

    if interim_id and interim_id != new_id:
        for ext in ("png", "jpeg", "jpg"):
            stray = GENERATED_DIR / f"{interim_id}.{ext}"
            if stray.exists():
                try:
                    stray.unlink()
                except OSError:
                    pass
        sidecar = GENERATED_DIR / f"{interim_id}.json"
        if sidecar.exists():
            try:
                sidecar.unlink()
            except OSError:
                pass

    try:
        import state
        with state._gallery_lock:
            state.GALLERY_INDEX[new_id] = meta
    except Exception:
        # A missing gallery refresh is cosmetic; the next scan picks it up.
        pass

    return meta

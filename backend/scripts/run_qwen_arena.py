"""Batch-run the 10 grand-arena scenes with Qwen-Image 2.1 via the live API
(POST /api/generate — exactly the endpoint the frontend uses, so jobs go
through the same worker thread, lock, and qwen_engine subprocess path).

Runs Qwen at 512x512 / 20 steps / guidance 1.0 / sampler linear / seeds
1001-1010. Copies each finished PNG into
comparison_12_vs_4_steps/images/qwen-image-2.1_{scene}.png and writes a JSON
report (backend/data/generated/qwen_arena_results.json) with per-scene
generation/load times for the article.

Usage:
    ../venv/bin/python scripts/run_qwen_arena.py [--only scene_id ...]
"""
import argparse
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

BASE = "http://localhost:8001"
REPO = Path(__file__).resolve().parents[2]
IMAGES_DIR = REPO / "comparison_12_vs_4_steps" / "images"
REPORT = REPO / "backend" / "data" / "generated" / "qwen_arena_results.json"
POLL_INTERVAL = 5
sys.path.insert(0, str(REPO / "backend"))

from image_meta import atomic_write_json

SCENES = [
    {
        "id": "misty_fjord",
        "title": "1. Scandinavian Fjord",
        "seed": 1001,
        "prompt": "A serene and atmospheric photograph of a Scandinavian fjord at the break of dawn. In the foreground, a weathered wooden dock with visible wood grain and damp planks extends into still, emerald-tinted water that perfectly mirrors the sky. Dense evergreen pine forests climb the steep granite cliffs on either side. Thick morning mist drifts low across the surface, catching the gentle golden rays of the rising sun. The lighting is soft and diffuse, casting subtle specular glints on the undisturbed water.",
    },
    {
        "id": "elderly_watchmaker",
        "title": "2. Master Watchmaker",
        "seed": 1002,
        "prompt": "A detailed environmental portrait of an elderly master watchmaker in his antique workshop. The man wears small wire-rimmed spectacles and focuses intently on the exposed movement of a pocket watch resting on his worn wooden workbench. His hands show delicate age and precision as he holds fine watchmaking tweezers. Surrounding him are miniature brass gears, coiled springs, magnifying loupes, and glass jars. Warm amber light from an adjustable desk lamp creates deep shadows and highlights the fine metallic teeth of the clockwork, while soft ambient light filters in from an unseen window.",
    },
    {
        "id": "modern_glass_villa",
        "title": "3. Modern Glass Villa",
        "seed": 1003,
        "prompt": "A striking architectural photograph taken at twilight of a sleek minimalist villa cantilevered daringly over a rocky ocean cliff. Floor-to-ceiling glass facades and smooth board-formed concrete slabs define the geometric structure. Inside, warm recessed lighting casts a cozy amber illumination that contrasts with the deep navy and slate-gray storm clouds in the evening sky. Below the cliff, turbulent dark blue waves crash into white seafoam against dark basalt rocks. A serene infinity pool on the terrace reflects both the moody twilight sky and the villa's warm interior.",
    },
    {
        "id": "snow_leopard",
        "title": "4. Snow Leopard",
        "seed": 1004,
        "prompt": "A high-fidelity wildlife photograph of an adult snow leopard resting on a rugged, windswept ridge high in the Himalayas. The animal's thick, pale-gray coat is patterned with dark rosettes, with individual hairs dusted in fine, powdery snow. Its intense, pale eyes look alertly forward, and its long, heavy tail rests across the cold stone. In the background, dramatic razor-sharp alpine peaks rise beneath a crisp, pale-blue winter sky, illuminated by the harsh, direct sidelight of early morning mountain sun.",
    },
    {
        "id": "vintage_cafe_racer",
        "title": "5. Vintage Cafe Racer",
        "seed": 1005,
        "prompt": "A moody nighttime urban photograph featuring a vintage 1960s cafe racer motorcycle parked along a narrow, damp cobblestone street in London. The motorcycle features a sculpted aluminum gas tank, polished chrome exhaust headers, and a stitched dark leather seat. Wet cobblestones glisten with vivid pink and amber reflections from a nearby vintage neon shop sign. Shallow depth of field blurs brick row houses and Victorian street lamps in the misty night air, emphasizing the crisp metallic details and condensation droplets on the bike.",
    },
    {
        "id": "honeybee_macro",
        "title": "6. Honeybee Macro",
        "seed": 1006,
        "prompt": "An extreme close-up macro photograph capturing a honeybee actively collecting nectar on a fresh purple lavender blossom. The bee's compound eyes, delicate antennae, and fine golden hairs covering its thorax are in sharp microscopic focus, coated with tiny yellow specks of pollen. Its delicate, iridescent wings shimmer with translucent veins. Warm natural sunlight backlights the flower petals and catches floating pollen particles suspended in the air. The background dissolves into a smooth, creamy green and violet bokeh.",
    },
    {
        "id": "lunar_biodome",
        "title": "7. Lunar Biodome",
        "seed": 1007,
        "prompt": "A sweeping interior photograph of a massive geodesic greenhouse dome established on the surface of the Moon. Through the transparent triangular glass panels of the dome, the stark gray lunar regolith and the vibrant blue-and-white sphere of Earth hang in the pitch-black starry sky. Inside the climate-controlled facility, terraced garden beds flourish with dense exotic plants emitting gentle teal and magenta bioluminescence. Fine humidity mist drifts between elevated catwalks made of brushed composite metal, catching soft ambient glows.",
    },
    {
        "id": "moss_stone_golem",
        "title": "8. Moss Stone Golem",
        "seed": 1008,
        "prompt": "A serene and magical realistic photograph of a monumental stone golem resting seated in a verdant forest clearing. The golem is sculpted from weathered, age-old boulders, with emerald moss, creeping ivy, and delicate woodland flowers growing within the cracks of its granite body. Gentle European robins and blue tits perch calmly along its broad stone shoulders. Beams of morning sunlight cut through the canopy of towering ancient oak trees, illuminating floating dust and dew on the damp forest floor.",
    },
    {
        "id": "rustic_sourdough",
        "title": "9. Rustic Sourdough",
        "seed": 1009,
        "prompt": "A mouthwatering culinary photograph of a rustic, freshly baked sourdough loaf resting on a weathered oak cutting board lightly dusted with flour. One thick slice has been cut, revealing an airy, glossy crumb with natural irregular pockets and a deep golden, blistered crust with intricate decorative scoring. A wisp of steam rises gently from the warm interior. Next to the loaf lies a forged steel bread knife with a wooden handle, accompanied by a few dried wheat stalks. Soft, warm daylight streams across the table from a nearby window, creating gentle shadows and highlighting the tactile crust texture.",
    },
    {
        "id": "saturn_from_enceladus",
        "title": "10. Saturn from Enceladus",
        "seed": 1010,
        "prompt": "An awe-inspiring astronomical photograph captured from the jagged, ice-chasm surface of Saturn's moon Enceladus. In the foreground, ridges of blue-tinted crystalline ice and deep fissures catch faint, distant sunlight, with subtle cryovolcanic plumes rising into the vacuum. Dominating the upper sky is the immense, magnificent sphere of Saturn, showing subtle golden-yellow and pale butterscotch atmospheric storm bands. Its razor-thin, brilliantly illuminated ring system cuts diagonally across the dark cosmic void, with distant pinpoint stars gleaming sharply in deep space.",
    },
]


def _post(path, payload):
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def _get(path):
    with urllib.request.urlopen(f"{BASE}{path}") as r:
        return json.loads(r.read())


def run_scene(scene, report, only=None, width=512, height=512, force=False, report_path=REPORT):
    sid = scene["id"]
    if only and sid not in only:
        return
    if sid in report and not force:
        print(f"[skip] {sid} already done", flush=True)
        return
    payload = {
        "prompt": scene["prompt"],
        "model": "qwen-image-2.1",
        "width": width,
        "height": height,
        "steps": 20,
        "guidance": 1.0,
        "seed": scene["seed"],
        "quantization": 4,
        "sampler": "linear",
        "output_format": "png",
        "fast_vae": True,
    }
    print(f"[start] {sid} (seed {scene['seed']}, {width}x{height}, 20 steps)", flush=True)
    job = _post("/api/generate", payload)
    jid = job["job_id"]
    t0 = time.time()
    while True:
        time.sleep(POLL_INTERVAL)
        st = _get(f"/api/jobs/{jid}")
        status = st.get("status")
        if status in ("done", "error") or (status == "cancelled"):
            break
    wall = round(time.time() - t0, 1)
    if status != "done":
        print(f"[FAIL] {sid} status={status} err={st.get('error', st.get('phase_detail', '?'))}", flush=True)
        report[sid] = {"status": status, "wall": wall, "error": st.get("error")}
        _save(report, report_path)
        return
    results = st.get("results") or []
    gen_time = results[0].get("generation_time", 0.0) if results else 0.0
    load_time = results[0].get("load_time", 0.0) if results else 0.0
    saved = (st.get("last_saved") or {}).get("id")
    if not saved:
        print(f"[FAIL] {sid} no saved id", flush=True)
        report[sid] = {"status": "error", "wall": wall, "error": "no saved id"}
        _save(report, report_path)
        return
    src = REPO / "backend" / "data" / "generated" / f"{saved}.png"
    dst = IMAGES_DIR / f"qwen-image-2.1_{sid}.png"
    if not src.exists():
        print(f"[FAIL] {sid} missing artifact {src}", flush=True)
        report[sid] = {"status": "error", "wall": wall}
        _save(report, report_path)
        return
    shutil.copy2(src, dst)
    report[sid] = {
        "status": "done",
        "seed": scene["seed"],
        "generation_time": gen_time,
        "load_time": load_time,
        "wall": wall,
        "image": str(dst.relative_to(REPO)),
        "job_id": jid,
        "cache_state": "cold-per-job-subprocess",
        "timing_scope": "load_time and generation_time come from the engine result; wall includes API polling",
    }
    report["_meta"] = {"width": width, "height": height, "steps": 20, "guidance": 1.0, "sampler": "linear", "model": "qwen-image-2.1", "cache_state": "cold-per-job-subprocess"}
    print(f"[done] {sid} gen={gen_time}s load={load_time}s wall={wall}s -> {dst.name}", flush=True)
    _save(report, report_path)


def _save(report, path=REPORT):
    atomic_write_json(path, report, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="scene ids to run (default: all undone)")
    ap.add_argument("--width", type=int, default=512, help="output width")
    ap.add_argument("--height", type=int, default=512, help="output height")
    ap.add_argument("--force", action="store_true", help="re-run scenes already present in the report")
    ap.add_argument("--report", type=Path, default=None, help="override report path")
    args = ap.parse_args()
    only = set(args.only) if args.only else None
    report = {}
    _rp = args.report or REPORT
    if _rp.exists():
        report = json.loads(_rp.read_text())
        report = {k: v for k, v in report.items() if k != "_meta"}
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for scene in SCENES:
        sid = scene["id"]
        if only and sid not in only:
            continue
        if sid in report and not args.force:
            print(f"[skip] {sid} already done", flush=True)
            continue
        try:
            run_scene(scene, report, only, args.width, args.height, force=args.force, report_path=_rp)
        except Exception as e:
            print(f"[ERR] {scene['id']}: {type(e).__name__}: {e}", flush=True)
            report[scene["id"]] = {"status": "error", "error": str(e)}
            _save(report, _rp)
    done = sum(1 for v in report.values() if v.get("status") == "done")
    print(f"== {done}/{len(SCENES)} scenes done. report: {_rp}", flush=True)


if __name__ == "__main__":
    main()
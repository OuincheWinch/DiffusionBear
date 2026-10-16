import sys
import time

import generator

PROMPT = "a red fox sitting in a snowy forest, golden hour light, photorealistic"
SEED = 42


def run(label: str, width: int, height: int):
    times = []
    for i in range(2):  # first run includes any lazy init, second is warm
        t0 = time.time()
        meta = generator.generate(
            prompt=PROMPT,
            width=width,
            height=height,
            steps=4,
            guidance=1.0,
            seed=SEED,
            quantization=4,
        )
        dt = time.time() - t0
        times.append(dt)
        print(f"[{label}] run{i + 1}: {dt:.1f}s", flush=True)
    print(f"RESULT {label} warm={times[1]:.1f}s cold={times[0]:.1f}s", flush=True)


if __name__ == "__main__":
    size = sys.argv[1] if len(sys.argv) > 1 else "1024"
    n = int(size)
    run(f"{n}x{n}", n, n)

"""Record a REAL `culprit demo` run and render its terminal output as results/demo.gif."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
COLS, ROWS = 100, 34
FONT = "/System/Library/Fonts/Menlo.ttc"


def record(model: str) -> list[tuple[float, str]]:
    cmd = [str(ROOT / ".venv" / "bin" / "culprit"), "demo", "--model", model]
    env = {**os.environ, "COLUMNS": str(COLS), "TERM": "dumb", "NO_COLOR": "1"}
    t0 = time.time()
    lines = [(0.0, f"$ culprit demo --model {model}")]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=ROOT)
    for line in p.stdout:
        lines.append((round(time.time() - t0, 2), line.rstrip("\n")))
    p.wait()
    return lines


def render(lines: list[tuple[float, str]], out: Path) -> None:
    try:
        font = ImageFont.truetype(FONT, 15)
    except OSError:
        font = ImageFont.load_default()
    cw, ch = font.getbbox("M")[2], 20
    W, H = COLS * cw + 24, ROWS * ch + 40
    frames, durations, shown, prev_t = [], [], [], 0.0

    def frame(note=""):
        img = Image.new("RGB", (W, H), (22, 24, 29))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W, 26], fill=(40, 43, 50))
        d.text((12, 5), "culprit demo  (real run, waits shortened)" + note, font=font, fill=(160, 165, 175))
        for i, l in enumerate(shown[-ROWS:]):
            color = (120, 200, 255) if l.startswith("$") else (230, 230, 230)
            if "VERIFIED" in l or "HANDED OFF" in l or "NO REGRESSION" in l:
                color = (255, 210, 90)
            d.text((12, 32 + i * ch), l[:COLS], font=font, fill=color)
        return img

    for t, l in lines:
        gap = t - prev_t
        if gap > 1.5 and frames:
            frames.append(frame(f"   ... {gap:.0f}s ..."))
            durations.append(700)
        shown.append(l)
        frames.append(frame())
        durations.append(110 if l.strip() else 40)
        prev_t = t
    durations[-1] = 6000
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5-coder:3b")
    args = p.parse_args()
    lines = record(args.model)
    (RESULTS / "demo_cast.json").write_text(json.dumps(lines, indent=0))
    (RESULTS / "demo_output.txt").write_text("\n".join(l for _, l in lines) + "\n")
    render(lines, RESULTS / "demo.gif")
    print(f"recorded {len(lines)} lines over {lines[-1][0]:.0f}s -> {RESULTS / 'demo.gif'}")


if __name__ == "__main__":
    sys.exit(main())

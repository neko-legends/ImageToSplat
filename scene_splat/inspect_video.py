"""Check an AI-made walkthrough clip before reconstruction: hard cuts, cross-fades (dissolves), too little motion.

AI video does not always follow the prompt. Cuts and dissolves split structure-from-motion into pieces, and a clip
that barely moves adds nothing. Also writes a contact sheet so a person can look at the clip.
"""
import subprocess
from pathlib import Path

import numpy as np

from .frames import ffmpeg_exe

SAMPLE_FPS = 8
W, H = 192, 108
MATCH_FPS = 4


def _match_inliers(video: Path) -> np.ndarray:
    """Geometrically consistent SIFT matches between neighbouring frames (MATCH_FPS, 640x360)."""
    import warnings

    import pycolmap

    w, h = 640, 360
    raw = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-i", str(video), "-vf", f"fps={MATCH_FPS},scale={w}:{h}", "-pix_fmt", "gray",
         "-f", "rawvideo", "-"], check=True, capture_output=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32) / 255.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        sift = pycolmap.Sift()
    feats = [sift.extract(f) for f in frames]
    counts = []
    for (k1, d1), (k2, d2) in zip(feats, feats[1:]):
        sim = d1.astype(np.float32) @ d2.astype(np.float32).T
        a, b = sim.argmax(1), sim.argmax(0)
        mutual = np.where(b[a] == np.arange(len(a)))[0]
        if len(mutual) < 8:
            counts.append(0)
            continue
        fit = pycolmap.estimate_fundamental_matrix(k1[mutual, :2].astype(np.float64), k2[a[mutual], :2].astype(np.float64))
        counts.append(int(np.sum(fit["inlier_mask"])) if fit else 0)
    return np.array(counts, dtype=np.float64)


def _frames(video: Path) -> np.ndarray:
    raw = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-i", str(video), "-vf", f"fps={SAMPLE_FPS},scale={W}:{H}", "-pix_fmt", "gray",
         "-f", "rawvideo", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32) / 255.0


def _contact_sheet(video: Path, out: Path, cols: int = 8, rows: int = 3) -> Path:
    from PIL import Image

    raw = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-i", str(video), "-vf", "scale=320:180", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        check=True, capture_output=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 180, 320, 3)
    picks = np.linspace(0, len(frames) - 1, cols * rows).round().astype(int)
    sheet = Image.new("RGB", (320 * cols, 180 * rows))
    for k, i in enumerate(picks):
        sheet.paste(Image.fromarray(frames[i]), ((k % cols) * 320, (k // cols) * 180))
    sheet.save(out, quality=85)
    return out


def inspect_clip(video: Path, sheet: Path | None = None) -> dict:
    f = _frames(Path(video))
    if len(f) < 8:
        return {"ok": False, "problems": ["clip too short"], "frames": len(f)}
    diff = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))  # motion between neighbouring samples
    med = float(np.median(diff)) + 1e-6
    problems, events = [], []
    for i in np.where(diff > max(6 * med, 0.08))[0]:
        events.append({"type": "cut", "time": round((i + 1) / SAMPLE_FPS, 2)})
    # dissolves and glitches: consistent SIFT matches between neighbouring frames drop suddenly
    # (a smooth dip only means the camera turns faster)
    inliers = _match_inliers(Path(video))
    for i in range(len(inliers)):
        local = np.median(inliers[max(0, i - 4):i + 5])
        if inliers[i] < 0.55 * local or inliers[i] < 150:
            events.append({"type": "dissolve", "time": round((i + 1) / MATCH_FPS, 2),
                           "matches": int(inliers[i]), "localMatches": int(local)})
    merged = []
    for e in sorted(events, key=lambda e: e["time"]):
        if merged and e["type"] == merged[-1]["type"] and e["time"] - merged[-1]["time"] < 1.0:
            continue
        merged.append(e)
    if any(e["type"] == "cut" for e in merged):
        problems.append("hard cut")
    if any(e["type"] == "dissolve" for e in merged):
        problems.append("cross-fade / dissolve")
    total_motion = float(diff.sum())
    if total_motion < 0.6:
        problems.append("camera barely moves")
    report = {"ok": not problems, "problems": problems, "events": merged, "motion": round(total_motion, 3),
              "seconds": round(len(f) / SAMPLE_FPS, 2)}
    if sheet:
        report["contactSheet"] = str(_contact_sheet(Path(video), Path(sheet)))
    return report


if __name__ == "__main__":
    import json
    import sys

    for v in sys.argv[1:]:
        print(v, json.dumps(inspect_clip(Path(v))))

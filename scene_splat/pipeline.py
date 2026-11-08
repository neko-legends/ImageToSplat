"""Videos -> splat scene, end to end."""
import json
import shutil
import time
from pathlib import Path

from .frames import extract_frames


def run_scene(videos, out_dir, name="scene", fps=6.0, width=1280, steps=20000, max_splats=600_000, sh_degree=1,
              keep_work=False, device="cuda", sfm="colmap", progress=lambda msg, frac=None: None) -> dict:
    """Build a .spz splat scene from one or more videos of the same static scene.

    Returns paths to <out_dir>/<name>.spz and <name>.cameras.json plus quality stats.
    Progress fractions: frames 0-5 %, camera solve 5-35 %, training 35-100 %.
    """
    from .cameras import export_cameras
    from .train import train_splats

    if sfm == "mapanything":
        from .sfm_mapanything import reconstruct
    elif sfm == "colmap":
        from .sfm import reconstruct
    else:
        raise ValueError(f"Unknown sfm method: {sfm}")

    t0 = time.time()
    videos = [Path(v).expanduser().resolve() for v in videos]
    for v in videos:
        if not v.is_file():
            raise FileNotFoundError(f"Video not found: {v}")
    out_dir = Path(out_dir).expanduser().resolve()
    work = out_dir / f"{name}.work"
    if work.exists():
        shutil.rmtree(work)
    images = work / "images"

    def sub(lo, hi):
        return lambda msg, frac=None: progress(msg, None if frac is None else lo + (hi - lo) * frac)

    from .inspect_video import inspect_clip

    sheets = out_dir / f"{name}.clips"
    sheets.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, v in enumerate(videos):
        progress(f"Inspecting {v.name}", 0.05 * i / len(videos))
        report = inspect_clip(v, sheets / f"{i + 1:02d}_{v.stem}.jpg")
        clips.append({"video": str(v), **report})
        if not report["ok"]:
            progress(f"Warning: {v.name}: {', '.join(report['problems'])} (parts may not join the scene)", None)
        extract_frames(v, images, f"v{i:02d}", fps, width)
    progress("Frames ready", 0.05)

    t_sfm = time.time()
    undistorted, sfm_stats = reconstruct(work, sub(0.05, 0.35))
    sfm_stats["seconds"] = round(time.time() - t_sfm, 1)
    sfm_stats.setdefault("method", sfm)
    progress("Camera positions solved", 0.35)
    trained = train_splats(undistorted, work / "train", steps=steps, max_splats=max_splats, sh_degree=sh_degree,
                           device=device, pose_opt=(sfm == "mapanything"), progress=sub(0.35, 0.98))

    out_dir.mkdir(parents=True, exist_ok=True)
    spz = out_dir / f"{name}.spz"
    shutil.move(trained["spz"], spz)
    cams = out_dir / f"{name}.cameras.json"
    export_cameras(undistorted / "sparse", cams)
    result = {
        "spz": str(spz),
        "cameras": str(cams),
        "videos": [str(v) for v in videos],
        "clips": clips,
        "sfm": sfm_stats,
        "splats": trained["splats"],
        "sizeBytes": spz.stat().st_size,
        "heldOutPsnr": trained["heldOutPsnr"],
        "seconds": round(time.time() - t0, 1),
        "settings": {"sfm": sfm, "fps": fps, "width": width, "steps": steps, "maxSplats": max_splats, "shDegree": sh_degree},
    }
    (out_dir / f"{name}.scene.json").write_text(json.dumps(result, indent=1))
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    progress("Scene ready", 1.0)
    return result

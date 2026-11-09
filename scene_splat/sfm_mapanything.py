"""Camera poses with MapAnything (Meta, Apache 2.0 weights) instead of COLMAP: one GPU pass over all frames.

Writes the same layout as sfm.reconstruct (<work>/undistorted/images + sparse), so training is unchanged.
MapAnything predicts pinhole cameras, so no undistortion is needed. Frames are resized (not cropped) to the model
input size, and the predicted intrinsics are scaled back to the full frames.
"""
import shutil
from pathlib import Path

import numpy as np

MODEL_ID = "facebook/map-anything-apache"
MAX_POINTS = 400_000
DINOV2_HUB_DIR_ENV = "IMAGETOSPLAT_DINOV2_DIR"


def _offline_dinov2_hub():
    """MapAnything builds its DINOv2 encoder with torch.hub.load("facebookresearch/dinov2", ...), which contacts GitHub
    and runs the latest code from there. Load a local, reviewed copy instead, without its separate pretrained weights
    (the MapAnything checkpoint already contains the encoder weights).

    Local copy: IMAGETOSPLAT_DINOV2_DIR, else <torch hub dir>/facebookresearch_dinov2_main.
    """
    import os

    import torch

    local = Path(os.environ.get(DINOV2_HUB_DIR_ENV) or Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main")
    if not (local / "hubconf.py").is_file():
        raise RuntimeError(f"DINOv2 code not found at {local}. Put a reviewed copy of github.com/facebookresearch/dinov2 "
                           f"there or set {DINOV2_HUB_DIR_ENV}.")
    original = torch.hub.load

    def load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir == "facebookresearch/dinov2":
            kwargs.pop("force_reload", None)
            kwargs["pretrained"] = False
            return original(str(local), model, *args, source="local", **kwargs)
        return original(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = load
    return original


def reconstruct(work: Path, progress=lambda msg, frac=None: None, device="cuda", model_id=MODEL_ID):
    import os

    import pycolmap
    import torch
    from PIL import Image

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    frames = sorted((work / "images").glob("*.jpg"))
    full_w, full_h = Image.open(frames[0]).size
    in_w = 518
    in_h = max(14, round(in_w * full_h / full_w / 14) * 14)
    small = work / "ma_input"
    small.mkdir(exist_ok=True)
    for f in frames:
        Image.open(f).convert("RGB").resize((in_w, in_h), Image.LANCZOS).save(small / f.name, quality=95)

    progress(f"MapAnything: placing {len(frames)} frames on the GPU", None)
    original_hub_load = _offline_dinov2_hub()
    try:
        model = MapAnything.from_pretrained(model_id).to(device).eval()
    finally:
        torch.hub.load = original_hub_load
    views = load_images([str(small / f.name) for f in frames], resize_mode="fixed_size", size=(in_w, in_h))
    with torch.no_grad():
        preds = model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16",
                            apply_mask=True, mask_edges=True)
    del model
    torch.cuda.empty_cache()

    sx, sy = full_w / in_w, full_h / in_h
    Ks = np.stack([p["intrinsics"][0].float().cpu().numpy() for p in preds])
    fx, fy = float(np.median(Ks[:, 0, 0])) * sx, float(np.median(Ks[:, 1, 1])) * sy
    cx, cy = full_w / 2, full_h / 2

    rec = pycolmap.Reconstruction()
    rec.add_camera_with_trivial_rig(pycolmap.Camera(model="PINHOLE", width=full_w, height=full_h,
                                                    params=[fx, fy, cx, cy], camera_id=1))
    pts, cols = [], []
    for i, (f, p) in enumerate(zip(frames, preds), start=1):
        c2w = p["camera_poses"][0].float().cpu().numpy().astype(np.float64)
        w2c = np.linalg.inv(c2w)
        rec.add_image_with_trivial_frame(pycolmap.Image(name=f.name, camera_id=1, image_id=i),
                                         pycolmap.Rigid3d(w2c[:3, :4]))
        xyz = p["pts3d"][0].float().cpu().numpy().reshape(-1, 3)
        rgb = (p["img_no_norm"][0].float().cpu().numpy().reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)
        keep = p["mask"][0].cpu().numpy().reshape(-1).astype(bool)
        pts.append(xyz[keep])
        cols.append(rgb[keep])
    pts, cols = np.concatenate(pts), np.concatenate(cols)
    if len(pts) > MAX_POINTS:
        pick = np.random.default_rng(0).choice(len(pts), MAX_POINTS, replace=False)
        pts, cols = pts[pick], cols[pick]
    for xyz, rgb in zip(pts.astype(np.float64), cols):
        rec.add_point3D(xyz, pycolmap.Track(), rgb)

    from .level import level

    level(rec)
    out = work / "undistorted"
    (out / "sparse").mkdir(parents=True, exist_ok=True)
    rec.write(out / "sparse")
    if (out / "images").exists():
        shutil.rmtree(out / "images")
    shutil.copytree(work / "images", out / "images")
    shutil.rmtree(small, ignore_errors=True)
    stats = {"frames": len(frames), "registered": len(frames), "points": int(len(pts)), "reprojectionError": None,
             "models": 1, "method": "mapanything", "focal": [round(fx, 1), round(fy, 1)]}
    progress(f"Placed {len(frames)}/{len(frames)} frames", None)
    return out, stats

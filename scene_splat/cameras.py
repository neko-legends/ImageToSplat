"""Export COLMAP camera poses in three.js conventions (y up, camera looks down -z)."""
import json
from pathlib import Path

import numpy as np

FLIP = np.array([1.0, -1.0, -1.0])  # COLMAP (x right, y down, z forward) -> three.js


def export_cameras(sparse_dir: Path, out_json: Path) -> int:
    import pycolmap

    rec = pycolmap.Reconstruction(sparse_dir)
    cams = []
    for im in sorted(rec.images.values(), key=lambda i: i.name):
        pose = im.cam_from_world()
        R, t = pose.rotation.matrix(), pose.translation
        center = -R.T @ t
        forward = R.T @ np.array([0.0, 0.0, 1.0])
        up = -(R.T @ np.array([0.0, 1.0, 0.0]))
        cam = rec.cameras[im.camera_id]
        fy = cam.params[1] if len(cam.params) >= 4 else cam.params[0]
        cams.append({
            "name": im.name,
            "position": (center * FLIP).tolist(),
            "target": ((center + forward) * FLIP).tolist(),
            "up": (up * FLIP).tolist(),
            "fovY": float(np.degrees(2 * np.arctan(cam.height / 2 / fy))),
            "aspect": cam.width / cam.height,
        })
    positions = np.array([c["position"] for c in cams])
    doc = {
        "convention": "three.js (y up); the exported splat is already in this frame",
        "center": positions.mean(0).tolist(),
        "radius": float(np.linalg.norm(positions - positions.mean(0), axis=1).max()),
        "cameras": cams,
    }
    out_json.write_text(json.dumps(doc))
    return len(cams)

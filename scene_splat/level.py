"""Level a reconstruction: gravity down = +y (COLMAP convention), cameras centred on the origin.

Walkthrough cameras are held upright, so the average camera "up" is the room's up. Solvers leave the world tilted
(it follows the first frame), which tilts every view in the viewer and makes walking go up or down a slope.
"""
import numpy as np


def level(rec) -> np.ndarray:
    """Transform `rec` (pycolmap.Reconstruction) in place; returns the 3x4 world transform that was applied."""
    import pycolmap

    ups, centers = [], []
    for im in rec.images.values():
        pose = im.cam_from_world()
        R = pose.rotation.matrix()
        ups.append(R.T @ np.array([0.0, -1.0, 0.0]))  # camera up in world (image y points down)
        centers.append(-R.T @ pose.translation)
    up = np.mean(ups, 0)
    up /= np.linalg.norm(up)
    target = np.array([0.0, -1.0, 0.0])  # COLMAP: up is -y
    v = np.cross(up, target)
    s, c = np.linalg.norm(v), float(up @ target)
    if s < 1e-9:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
        ang = np.arctan2(s, c)
        R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    t = -R @ np.mean(centers, 0)
    M = np.hstack([R, t[:, None]])
    rec.transform(pycolmap.Sim3d(M))
    return M

"""Scene mode: one or more videos of a static scene -> camera poses -> Gaussian splat (.spz) for three.js.

Pipeline: ffmpeg frames -> COLMAP (pycolmap) structure-from-motion across all videos -> undistort ->
gsplat training (MCMC, capped splat count) -> compressed .spz + camera path JSON.

Heavy dependencies (pycolmap, torch with CUDA, gsplat) are optional; `python -m scene_splat --check`
reports what is missing. The app runs this package in a worker process so the main app does not need them.
"""

__all__ = ["run_scene"]


def run_scene(*args, **kwargs):
    from .pipeline import run_scene as _run

    return _run(*args, **kwargs)

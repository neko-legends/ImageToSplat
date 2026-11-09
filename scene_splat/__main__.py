"""CLI worker: python -m scene_splat --video a.mp4 [--video b.mp4 ...] --out DIR [--name NAME]

Prints one JSON object per line: {"event": "progress", ...}, then {"event": "result", ...} or {"event": "error", ...}.
`--check` reports which dependencies are available.
"""
import os

# OpenBLAS builds for a fixed number of threads (often 24 or 64); on CPUs with more cores, COLMAP's matcher threads
# corrupt its heap ("corrupted size vs. prev_size"). Cap it before numpy or pycolmap load it.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, str(min(16, os.cpu_count() or 1)))

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402


def emit(**payload):
    print(json.dumps(payload), flush=True)


def _prepare_cuda_build_env():
    """gsplat compiles its CUDA kernels on first use; point it at a CUDA toolkit if the user has not."""
    if not os.environ.get("CUDA_HOME"):
        nvcc = shutil.which("nvcc")
        candidates = [Path(nvcc).resolve().parent.parent] if nvcc else []
        candidates += [Path(p) for p in ("/opt/cuda", "/usr/local/cuda")]
        for c in candidates:
            if (c / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")).exists():
                os.environ["CUDA_HOME"] = str(c)
                break
    if os.environ.get("CUDA_HOME"):
        os.environ["PATH"] = os.pathsep.join([str(Path(os.environ["CUDA_HOME"]) / "bin"), os.environ.get("PATH", "")])
    # ninja ships inside the Python environment; make sure the build finds it.
    os.environ["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")])
    if os.name != "nt":
        os.environ.setdefault("NVCC_APPEND_FLAGS", "-allow-unsupported-compiler")


def check() -> dict:
    report = {"python": sys.executable}
    try:
        from .frames import ffmpeg_exe

        report["ffmpeg"] = ffmpeg_exe()
    except Exception as exc:
        report["ffmpeg"] = None
        report["ffmpegError"] = str(exc)
    for mod in ("pycolmap", "torch", "gsplat", "mapanything", "skimage", "trimesh", "xatlas", "fast_simplification", "scipy"):
        try:
            m = __import__(mod)
            report[mod] = getattr(m, "__version__", "installed")
        except Exception as exc:
            report[mod] = None
            report[f"{mod}Error"] = str(exc)
    if report.get("torch"):
        import torch

        report["cuda"] = torch.cuda.is_available()
        report["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    report["cudaHome"] = os.environ.get("CUDA_HOME")
    report["ready"] = bool(report.get("ffmpeg") and report.get("pycolmap") and report.get("gsplat") and report.get("cuda"))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m scene_splat", description="Videos of a static scene -> .spz splat scene")
    ap.add_argument("--video", action="append", default=[], help="Input video (repeat for several clips of one scene)")
    ap.add_argument("--out", help="Output directory")
    ap.add_argument("--name", default="scene")
    ap.add_argument("--fps", type=float, default=10.0, help="Frames per second taken from each video")
    ap.add_argument("--width", type=int, default=1280, help="Frame width used for reconstruction")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--max-splats", type=int, default=600_000)
    ap.add_argument("--sh-degree", type=int, default=1, choices=[0, 1])
    ap.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cuda:1")
    ap.add_argument("--sfm", choices=["colmap", "mapanything"], default="colmap",
                    help="Camera solver: COLMAP (CPU) or MapAnything (GPU, needs the mapanything package)")
    ap.add_argument("--keep-work", action="store_true", help="Keep frames and COLMAP files")
    ap.add_argument("--check", action="store_true", help="Report available dependencies and exit")
    mesh = ap.add_argument_group("Mesh export: turn a splat .ply into a real polygon mesh (.glb with vertex colours)")
    mesh.add_argument("--mesh", help="Splat .ply or .spz to convert (instead of building a scene)")
    mesh.add_argument("--mesh-out", help="Output .glb (default: next to the .ply, _mesh.glb)")
    mesh.add_argument("--mesh-up", default="-y", choices=["y", "-y", "z", "-z"], help="Up axis of the splat (TripoSplat: -y)")
    mesh.add_argument("--mesh-yaw", type=float, default=-90.0,
                      help="Turn about the up axis (degrees) so the front faces +z (TripoSplat: -90)")
    mesh.add_argument("--mesh-detail", default="high", choices=["low", "medium", "high", "auto"],
                      help="low 5k triangles + 1024 texture, medium 25k + 2048, high 100k + 2048, "
                           "auto = fewest triangles that keep the shape")
    mesh.add_argument("--mesh-faces", type=int, default=0, help="Exact triangle target instead of the preset's")
    mesh.add_argument("--mesh-texture", type=int, default=None,
                      help="Texture size instead of the preset's; 0 = colour per vertex, no texture")
    mesh.add_argument("--mesh-grid", type=int, default=256, help="Volume resolution; higher = finer detail and bigger file")
    cap = ap.add_argument_group("Venice capture (instead of --video): needs Venice Media Local with Agent Control on; "
                                "token from VENICE_MEDIA_LOCAL_TOKEN or ~/.config/venice-media-local/agent.json")
    cap.add_argument("--capture-image", help="Room image to film")
    cap.add_argument("--capture-prompt", help="Describe the room; Venice draws it first")
    cap.add_argument("--capture-view", action="append", default=[], help="Extra view images of the same room (skip auto views)")
    cap.add_argument("--capture-mode", choices=["first-last", "reference"], default="first-last")
    cap.add_argument("--capture-duration", default="5s")
    cap.add_argument("--venice-url", default=None, help="Venice Media Local address (default http://127.0.0.1:9876)")
    a = ap.parse_args()
    _prepare_cuda_build_env()
    if a.check:
        emit(event="check", **check())
        return 0
    if a.mesh:
        try:
            from .mesh import mesh_from_file

            ply = Path(a.mesh).expanduser().resolve()
            out = Path(a.mesh_out).expanduser().resolve() if a.mesh_out else ply.with_name(ply.stem.removesuffix("_native").removesuffix(".scene") + "_mesh.glb")
            emit(event="result", **mesh_from_file(ply, out, up=a.mesh_up, yaw=a.mesh_yaw, grid=a.mesh_grid, detail=a.mesh_detail, texture=a.mesh_texture, faces=a.mesh_faces, device=a.device,
                                                 progress=lambda msg, frac=None: emit(event="progress", message=msg, progress=frac)))
            return 0
        except Exception as exc:
            emit(event="error", error=str(exc), traceback=traceback.format_exc())
            return 1
    capture = bool(a.capture_image or a.capture_prompt)
    if not a.out or not (a.video or capture):
        ap.error("--out and either --video or --capture-image/--capture-prompt are required")
    try:
        from .pipeline import run_scene

        emit_progress = lambda msg, frac=None: emit(event="progress", message=msg, progress=frac)
        if capture:
            from .capture import capture_room

            a.video += [str(v) for v in capture_room(
                Path(a.out).expanduser().resolve() / f"{a.name}.capture", image=a.capture_image, prompt=a.capture_prompt,
                mode=a.capture_mode, duration=a.capture_duration, views=a.capture_view, url=a.venice_url,
                progress=emit_progress)]
            emit(event="videos", videos=a.video)

        result = run_scene(a.video, a.out, name=a.name, fps=a.fps, width=a.width, steps=a.steps,
                           max_splats=a.max_splats, sh_degree=a.sh_degree, keep_work=a.keep_work, device=a.device, sfm=a.sfm,
                           progress=emit_progress)
        emit(event="result", **result)
        return 0
    except Exception as exc:
        emit(event="error", error=str(exc), traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())

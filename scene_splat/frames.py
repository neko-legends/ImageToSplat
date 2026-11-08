"""Extract evenly spaced frames from videos with ffmpeg."""
import os
import shutil
import subprocess
from pathlib import Path


def ffmpeg_exe() -> str:
    """IMAGETOSPLAT_FFMPEG, then ffmpeg on PATH, then the imageio-ffmpeg bundled binary."""
    override = os.environ.get("IMAGETOSPLAT_FFMPEG")
    if override:
        return override
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - depends on the machine
        raise RuntimeError("ffmpeg not found: install it, add it to PATH, or set IMAGETOSPLAT_FFMPEG.") from exc


def extract_frames(video: Path, out_dir: Path, prefix: str, fps: float, width: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / f"{prefix}_%04d.jpg"
    subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-y", "-i", str(video),
         "-vf", f"fps={fps},scale={width}:-2", "-q:v", "2", str(pattern)],
        check=True,
    )
    return sorted(out_dir.glob(f"{prefix}_*.jpg"))

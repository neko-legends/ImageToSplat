"""Image or prompt -> room views (Sunburst edits) -> walkthrough video(s) via Venice Media Local."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import venice
from .inspect_video import inspect_clip


def _checked(make, out: Path, progress, retries=1):
    """Make a clip, inspect it (cuts, dissolves, no motion) and re-shoot once if it fails."""
    for attempt in range(retries + 1):
        clip = make(out if attempt == 0 else out.with_name(f"{out.stem}.retry{attempt}{out.suffix}"))
        report = inspect_clip(clip, clip.with_suffix(".sheet.jpg"))
        if report["ok"]:
            return clip
        progress(f"{clip.name} failed inspection ({', '.join(report['problems'])})"
                 + ("; filming it again" if attempt < retries else "; using it anyway"), None)
    return clip


def capture_room(work: Path, image=None, prompt=None, mode="first-last", duration="5s", resolution="1080p",
                 views=None, url=None, token=None, progress=lambda m, f=None: None) -> list[Path]:
    """Return the video file(s) to reconstruct from.

    mode "reference": one Seedance 2.5 reference-to-video spin through all views (one clip).
    mode "first-last": Flux 3 first/last-frame clips between neighbouring views, closing the loop (one clip per view).
    mode "object-spin": one Flux 3 image-to-video 360 degree orbit around an object (one clip, 10 s, square).
    `views` may list extra view images to use instead of generating them.
    """
    work.mkdir(parents=True, exist_ok=True)
    vml = venice.VeniceMediaLocal(url, token)
    if image:
        source = Path(image).expanduser().resolve()
    elif prompt:
        progress("Drawing the room", None)
        source = vml.generate_image(
            f"{prompt}. Photorealistic wide-angle interior photo at eye level, richly textured surfaces, sharp focus "
            "everywhere, no people.", work / "view1.png")
    else:
        raise ValueError("Give an image or a prompt for the room.")
    if mode == "object-spin":
        progress("Venice is filming a 360 degree orbit", None)
        return [_checked(lambda out: vml.video(venice.SPIN_VIDEO_MODEL, venice.object_spin_prompt(), out, source=source,
                                               duration="10s", resolution=resolution, aspect="1:1", progress=progress),
                         work / "spin360.mp4", progress)]
    room = [source] + [Path(v).expanduser().resolve() for v in views] if views else venice.room_views(vml, source, work, progress)

    if mode == "reference":
        progress("Venice is filming the walkthrough", None)
        return [_checked(lambda out: vml.video(venice.REFERENCE_VIDEO_MODEL, venice.walkthrough_prompt(len(room)), out,
                                               references=room[:9], duration=duration, resolution=resolution,
                                               progress=progress), work / "walkthrough.mp4", progress)]
    if mode == "first-last":
        progress(f"Venice is filming {len(room)} segments", None)
        with ThreadPoolExecutor(max_workers=len(room)) as pool:
            def segment(i, a, b):
                return _checked(lambda out: vml.video(venice.FIRST_LAST_VIDEO_MODEL, venice.first_last_prompt(), out,
                                                      source=a, end=b, duration=duration, resolution=resolution,
                                                      progress=progress), work / f"segment{i + 1}.mp4", progress)

            jobs = [pool.submit(segment, i, a, room[(i + 1) % len(room)]) for i, a in enumerate(room)]
            return [j.result() for j in jobs]
    raise ValueError(f"Unknown capture mode: {mode}")

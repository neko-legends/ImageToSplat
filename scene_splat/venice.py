"""Make walkthrough videos of one room with Venice Media Local (the desktop app holds the Venice key).

Connection: arguments, else VENICE_MEDIA_LOCAL_URL / VENICE_MEDIA_LOCAL_TOKEN (or VML_URL / VML_TOKEN),
else ~/.config/venice-media-local/agent.json ({"url": ..., "token": ...}). The token is the app's Agent
Control token, never the Venice API key.
"""
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

IMAGE_MODEL = "gpt-image-2-6-sunburst"
EDIT_MODEL = "gpt-image-2-6-sunburst-edit"
REFERENCE_VIDEO_MODEL = "seedance-2-5-reference-to-video-basic"
FIRST_LAST_VIDEO_MODEL = "flux-3-first-last-frame-to-video"
FROZEN = ("Frozen moment: nothing in the room moves, no people, the light stays exactly the same. "
          "One continuous take, no cuts, no transitions, no fades. No music. "
          "The lens never zooms: the same wide field of view from the first frame to the last.")

# Views of a room, in clockwise order as seen from the middle of the room. The first is the source image.
ROOM_VIEWS = [
    ("right wall", "the camera stands at the opposite side of the room and looks straight at the wall that is on the "
                   "right side of Image 1, showing everything along that wall with the floor and furniture in the foreground"),
    ("back wall", "the camera stands at the far end of the room and looks straight back toward where the camera of "
                  "Image 1 stood, showing what was behind that camera"),
    ("left wall", "the camera stands at the opposite side of the room and looks straight at the wall that is on the "
                  "left side of Image 1, showing everything along that wall with the floor and furniture in the foreground"),
]


def _connection(url=None, token=None):
    url = url or os.environ.get("VENICE_MEDIA_LOCAL_URL") or os.environ.get("VML_URL")
    token = token or os.environ.get("VENICE_MEDIA_LOCAL_TOKEN") or os.environ.get("VML_TOKEN")
    cfg = Path.home() / ".config" / "venice-media-local" / "agent.json"
    if (not url or not token) and cfg.is_file():
        data = json.loads(cfg.read_text())
        url, token = url or data.get("url"), token or data.get("token")
    url = (url or "http://127.0.0.1:9876").rstrip("/")
    if not url.endswith("/api/v1"):
        url += "/api/v1"
    if not token:
        raise RuntimeError("Venice Media Local token missing: enable Agent Control in Venice Media Local and pass its token "
                           "(or set VENICE_MEDIA_LOCAL_TOKEN).")
    return url, token


def data_url(path: Path) -> str:
    path = Path(path)
    mime = {".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}.get(path.suffix.lower(), "png")
    return f"data:image/{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def _save_data_url(url: str, out: Path) -> Path:
    header, b64 = url.split(",", 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(b64))
    return out


class VeniceMediaLocal:
    def __init__(self, url=None, token=None):
        self.url, self.token = _connection(url, token)

    def call(self, route, body=None, method="POST", timeout=900):
        req = urllib.request.Request(self.url + route, data=None if body is None else json.dumps(body).encode(), method=method,
                                     headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Venice Media Local {route}: HTTP {e.code} {e.read()[:300].decode(errors='replace')}") from None

    def _first_image(self, response, out: Path) -> Path:
        if isinstance(response, list):
            results = response
        else:
            results = response.get("results") or response.get("images") or [response.get("result") or response]
        for r in results:
            if isinstance(r, dict) and str(r.get("dataUrl", "")).startswith("data:"):
                mime = r["dataUrl"].split(";", 1)[0].split("/")[-1]
                return _save_data_url(r["dataUrl"], out.with_suffix("." + {"jpeg": "jpg"}.get(mime, mime)))
        raise RuntimeError("Venice Media Local returned no image")

    def generate_image(self, prompt, out: Path, model=IMAGE_MODEL, aspect="16:9") -> Path:
        return self._first_image(self.call("/generate-image", {"model": model, "prompt": prompt, "aspectRatio": aspect}), out)

    def edit_image(self, prompt, images, out: Path, model=EDIT_MODEL, aspect="16:9") -> Path:
        body = {"model": model, "prompt": prompt, "images": [data_url(p) for p in images], "aspectRatio": aspect}
        return self._first_image(self.call("/edit-image", body), out)

    def video(self, model, prompt, out: Path, source=None, end=None, references=None, duration="10s",
              resolution="1080p", aspect="16:9", progress=lambda msg, frac=None: None, poll=20, timeout=3600) -> Path:
        body = {"model": model, "prompt": prompt, "duration": duration, "resolution": resolution, "aspectRatio": aspect}
        if source:
            body["sourceImage"] = data_url(source)
        if end:
            body["endImage"] = data_url(end)
        if references:
            body["referenceImageUrls"] = [data_url(p) for p in references]
        queued = self.call("/queue-video", body)
        qid = queued.get("queueId") or queued.get("queue_id")
        if not qid:
            raise RuntimeError(f"Venice Media Local did not queue the video: {str(queued)[:300]}")
        t0 = time.time()
        errors_since = None
        while time.time() - t0 < timeout:
            time.sleep(poll)
            try:
                r = self.call("/retrieve-video", {"queueId": qid, "model": model})
                errors_since = None
            except (RuntimeError, OSError) as e:  # Venice Media Local restarting or briefly unavailable: keep the job
                errors_since = errors_since or time.time()
                if time.time() - errors_since > 600:
                    raise
                progress(f"Venice is not answering ({str(e)[:120]}); still waiting for {out.name}", None)
                continue
            status = str(r.get("status", "")).lower()
            result = r.get("result") or {}
            if result.get("dataUrl"):
                return _save_data_url(result["dataUrl"], out)
            if status in {"failed", "error", "cancelled", "canceled"}:
                raise RuntimeError(f"Venice video failed: {str(r)[:300]}")
            progress(f"Venice is making {out.name} ({int(time.time() - t0)} s)", None)
        raise TimeoutError("Venice video timed out")


def room_panorama(vml: VeniceMediaLocal, source: Path, out: Path) -> Path:
    """One wide panorama of the whole room from the source camera. Every other view is drawn from it, so they agree on
    where things are (views edited independently put the same furniture in different places)."""
    prompt = ("Image 1 shows one room. Make a full 360 degree equirectangular panorama (as shot by a 360 camera on a "
              "tripod) of this exact room from the camera position of Image 1, at eye level. The middle of the panorama "
              "shows exactly what Image 1 shows; the rest shows the remaining walls of the same room all the way around, so "
              "the left and right edges join seamlessly. Keep the same furniture, materials, colours and light. "
              "Photorealistic, sharp, no people.")
    return vml.edit_image(prompt, [source], out, aspect="21:9")


def room_views(vml: VeniceMediaLocal, source: Path, work: Path, progress=lambda m, f=None: None) -> list[Path]:
    """Source image plus three edited views of the same room, in clockwise order, all drawn from one panorama."""
    progress("Drawing a panorama of the room", None)
    pano = room_panorama(vml, source, work / "panorama.png")
    views = [source]
    for i, (label, where) in enumerate(ROOM_VIEWS):
        progress(f"Drawing the {label} view", None)
        prompt = ("Image 1 is a wide panorama of one room seen from where the camera of Image 2 stands; Image 2 is a "
                  f"normal photo of the same room. Make another normal photo of this exact room: "
                  f"{where.replace('Image 1', 'Image 2')}. Everything must be where the panorama shows it: same layout, "
                  "furniture, decor, materials, colours and light. Photorealistic interior photo at eye level with the "
                  "same wide-angle lens and field of view as Image 2, sharp, no people.")
        views.append(vml.edit_image(prompt, [pano, source], work / f"view{i + 2}.png"))
    return views


def walkthrough_prompt(n_views: int) -> str:
    tags = ", ".join(f"<Image {i}>" for i in range(1, n_views + 1))
    order = ", then ".join(f"facing the view in <Image {i}>" for i in range(2, n_views + 1))
    return (f"One continuous steady gimbal shot inside this one room. {tags} are views of the same room. The camera stands "
            "near the middle of the room at eye level and makes one complete 360-degree clockwise turn at a steady brisk "
            f"pace while drifting slightly: it starts facing the view in <Image 1>, then {order}, and finishes facing "
            f"<Image 1> again. {FROZEN}")


def first_last_prompt() -> str:
    return ("One continuous steady gimbal shot inside this room: the camera, near the middle of the room at eye level, "
            "turns right at a steady brisk pace and drifts slightly sideways until it shows exactly the end frame. " + FROZEN)


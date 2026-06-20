"""Neko Legends TripoSplat app with Spark.js in-browser viewer.
Usage: python run_gradio.py
"""
import argparse
import json
import os
import socket
import struct
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

gr = None


def _ensure_gradio():
    global gr
    if gr is None:
        import gradio as gradio

        gr = gradio
    return gr


# ----------------------------------------------------------------------------
# Runtime paths and lazy pipeline setup
# ----------------------------------------------------------------------------

IS_FROZEN = getattr(sys, "frozen", False)
APP_DIR = Path(sys.executable).resolve().parent if IS_FROZEN else Path(__file__).resolve().parent
RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR))
CKPT_ROOT = Path(os.environ.get("TRIPOSPLAT_CKPTS", str(APP_DIR / "ckpts"))).resolve()
TRIPOSPLAT_REPO_ID = "VAST-AI/TripoSplat"
APP_NAME = "ImageToSplat"
APP_VERSION = "dev"
AGENT_APP_ID = "image-to-splat"
AGENT_APP_NAME = "ImageToSplat"
AGENT_API_BIND_ADDRESS = "127.0.0.1"
DEFAULT_AGENT_API_PORT = 17340
AGENT_API_REGISTRY_FILE = "agent-api-registry.json"

PIPE = None
TORCH = None
PIPE_PATHS = {
    "ckpt_path": CKPT_ROOT / "diffusion_models/triposplat_fp16.safetensors",
    "decoder_path": CKPT_ROOT / "vae/triposplat_vae_decoder_fp16.safetensors",
    "dinov3_path": CKPT_ROOT / "clip_vision/dino_v3_vit_h.safetensors",
    "flux2_vae_encoder_path": CKPT_ROOT / "vae/flux2-vae.safetensors",
    "rmbg_path": CKPT_ROOT / "background_removal/birefnet.safetensors",
}
MODEL_FILES = [
    ("Diffusion model", "diffusion_models/triposplat_fp16.safetensors"),
    ("VAE decoder", "vae/triposplat_vae_decoder_fp16.safetensors"),
    ("DINO vision encoder", "clip_vision/dino_v3_vit_h.safetensors"),
    ("Flux VAE encoder", "vae/flux2-vae.safetensors"),
    ("Background remover", "background_removal/birefnet.safetensors"),
]

OUT_ROOT     = (APP_DIR / "gradio_outputs").resolve()
OUT_ROOT.mkdir(parents=True, exist_ok=True)
VIEWER_HTML  = (RESOURCE_DIR / "static/viewer/viewer.html").resolve()
EXAMPLES_DIR = (RESOURCE_DIR / "static/example_inputs").resolve()
STATIC_DIR   = (RESOURCE_DIR / "static").resolve()
APP_ICON     = (RESOURCE_DIR / "static/app-icon.png").resolve()
EXAMPLES = [
    str(EXAMPLES_DIR / "creature_butterfly.webp"),
    str(EXAMPLES_DIR / "building_stone_house.webp"),
    str(EXAMPLES_DIR / "vehicle_pirate_ship.webp"),
    str(EXAMPLES_DIR / "plant_water_lily.webp"),
]

PLACEHOLDER_HTML = (
    "<div class='viewer-placeholder'>"
    "<div><strong>3D viewer</strong><span>Upload an image, generate a splat, then orbit it here.</span></div>"
    "</div>"
)

NEKO_CSS = """
:root {
  --bg: #090806;
  --rail: #0d0b09;
  --surface: #15100c;
  --surface-2: #201711;
  --surface-3: #2b1d13;
  --border: rgba(255, 138, 42, 0.24);
  --text: #f6efe7;
  --muted: #b9a99c;
  --accent: #ff8a1c;
  --accent-2: #ffc46f;
  --danger: #da373c;
  --ok: #7ee08a;
  --shadow: 0 18px 60px rgb(0 0 0 / 0.28);
}

body,
.gradio-container {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

footer { display: none !important; }

.gradio-container {
  max-width: none !important;
  min-height: 100vh;
  padding: 0 !important;
}

.main {
  display: grid;
  gap: 0 !important;
}

.neko-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  min-height: 72px;
  padding: 14px 22px;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
}

.neko-brand {
  display: grid;
  grid-template-columns: 48px minmax(0, 1fr);
  align-items: center;
  gap: 12px;
  min-width: 0;
}

.neko-brand img {
  width: 48px;
  height: 48px;
  object-fit: contain;
  filter: drop-shadow(0 10px 28px rgb(0 0 0 / 0.35));
}

.neko-brand h1 {
  margin: 0;
  color: var(--text);
  font-size: 19px;
  line-height: 1.15;
  letter-spacing: 0;
}

.neko-brand p {
  margin: 5px 0 0;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.35;
}

.neko-status-pill {
  display: inline-grid;
  place-items: center;
  min-height: 32px;
  padding: 0 12px;
  border: 1px solid color-mix(in srgb, var(--accent) 52%, var(--border));
  border-radius: 999px;
  background: color-mix(in srgb, var(--surface-3) 78%, var(--accent) 10%);
  color: var(--text);
  font-size: 12px;
  font-weight: 800;
  white-space: nowrap;
}

.neko-notice {
  display: flex;
  align-items: center;
  min-height: 36px;
  padding: 7px 22px;
  border-bottom: 1px solid var(--border);
  background: color-mix(in srgb, var(--surface-2) 78%, var(--accent) 10%);
  color: var(--text);
  font-size: 13px;
}

#setup-modal {
  position: fixed !important;
  inset: 0 !important;
  z-index: 20 !important;
  display: grid !important;
  place-items: center !important;
  padding: 24px !important;
  border: 0 !important;
  background: rgba(0, 0, 0, 0.62) !important;
}

#setup-modal > .wrap,
#setup-modal .block,
#setup-modal .form {
  width: min(680px, calc(100vw - 48px)) !important;
}

.setup-card {
  display: grid;
  gap: 10px;
  padding: 20px;
  border: 1px solid rgba(255, 138, 42, 0.28);
  border-radius: 8px;
  background: #100c09;
  box-shadow: 0 22px 70px rgba(0, 0, 0, 0.48);
}

.setup-card .eyebrow {
  color: var(--muted);
  font-size: 12px;
  font-weight: 900;
  text-transform: uppercase;
}

.setup-card h2 {
  margin: 0;
  color: var(--text);
  font-size: 24px;
  line-height: 1.1;
}

.setup-card p {
  margin: 0;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.45;
}

.setup-actions {
  display: flex !important;
  gap: 10px !important;
}

.setup-status {
  color: var(--muted);
  font-size: 13px;
  line-height: 1.45;
}

.neko-layout {
  display: grid !important;
  grid-template-columns: minmax(360px, 0.82fr) minmax(440px, 1.18fr);
  gap: 0 !important;
  min-height: calc(100vh - 108px);
}

.neko-panel-left,
.neko-panel-right {
  min-width: 0;
  min-height: 0;
}

.neko-panel-left {
  padding: 22px;
  background: var(--bg);
}

.neko-panel-right {
  padding: 0 0 22px;
  border-left: 1px solid var(--border);
  background: var(--surface);
}

.neko-card,
.neko-panel-left .block,
.neko-panel-right .block {
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  background: var(--surface) !important;
  box-shadow: none !important;
}

.neko-panel-left .form,
.neko-panel-left .block {
  gap: 14px !important;
}

.neko-viewer-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  min-height: 56px;
  padding: 0 16px;
  border-bottom: 1px solid var(--border);
}

.neko-viewer-header h2 {
  margin: 0;
  color: var(--text);
  font-size: 18px;
  line-height: 1.2;
}

.neko-viewer-header span {
  color: var(--muted);
  font-size: 12px;
  font-weight: 800;
  text-transform: uppercase;
}

#viewer-wrap {
  margin: 14px !important;
}

#viewer-wrap iframe,
.viewer-placeholder {
  width: 100%;
  height: min(68vh, 620px);
  min-height: 460px;
  border: 0;
  border-radius: 8px;
  background: #0a0b0e;
}

.viewer-placeholder {
  display: grid;
  place-items: center;
  border: 1px dashed var(--border);
  color: var(--muted);
  text-align: center;
}

.viewer-placeholder div {
  display: grid;
  gap: 6px;
}

.viewer-placeholder strong {
  color: var(--text);
  font-size: 18px;
}

.viewer-placeholder span {
  font-size: 13px;
}

.export-grid {
  display: grid !important;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 150px), 1fr));
  gap: 10px !important;
  margin: 0 14px !important;
}

.download-button,
.lg.primary,
button.primary {
  border-radius: 8px !important;
}

button.primary {
  background: linear-gradient(180deg, #ffb15c, #ff7918) !important;
  border-color: var(--accent) !important;
  color: #120a04 !important;
  font-weight: 800 !important;
}

button.secondary,
.download-button {
  background: var(--surface-2) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
}

button.secondary:hover,
.download-button:hover {
  border-color: color-mix(in srgb, var(--accent) 45%, var(--border)) !important;
  color: var(--accent-2) !important;
}

label,
.label-wrap,
.prose,
.markdown,
.wrap,
.block {
  color: var(--text) !important;
}

input,
select,
textarea {
  background: var(--surface) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
}

.info-text {
  color: var(--muted);
  font-size: 13px;
  line-height: 1.4;
}

@media (max-width: 980px) {
  .neko-layout {
    grid-template-columns: 1fr;
  }

  .neko-panel-right {
    border-left: 0;
    border-top: 1px solid var(--border);
  }

  .neko-topbar {
    align-items: flex-start;
    flex-direction: column;
  }
}
"""


def _gr_file(path: Path) -> str:
    """Gradio serves any file under `allowed_paths` at `/gradio_api/file=<abspath>`."""
    return f"/gradio_api/file={path.as_posix()}"


def _viewer_iframe(ply_path: Path) -> str:
    ts = time.time()  # cache-bust so the iframe reloads each generation
    src = f"{_gr_file(VIEWER_HTML)}?ply={_gr_file(ply_path)}&ts={ts}"
    return (
        f"<iframe src='{src}' "
        "title='Spark.js Gaussian splat viewer'></iframe>"
    )


def _rgba_from_gaussian(gaussian) -> np.ndarray:
    opacity = gaussian.get_opacity.detach().cpu().numpy()
    f_dc = gaussian._features_dc.detach().cpu().numpy()
    c0 = 0.28209479177387814
    rgb = np.clip((f_dc[:, 0, :] * c0 + 0.5) * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(opacity[:, 0:1] * 255, 0, 255).astype(np.uint8)
    return np.concatenate([rgb, alpha], axis=1)


def _point_cloud_data(gaussian):
    xyz, _ = gaussian._transformed_xyz_rot()
    return xyz.astype(np.float32), _rgba_from_gaussian(gaussian)


def _pad4(data: bytes, pad: bytes = b"\x00") -> bytes:
    return data + pad * ((4 - len(data) % 4) % 4)


def _gltf_document(point_count: int, xyz: np.ndarray, rgba: np.ndarray, buffer_uri: str | None):
    position_bytes = xyz.astype("<f4", copy=False).tobytes()
    color_bytes = rgba.tobytes()
    bin_length = len(position_bytes) + len(color_bytes)
    buffer_def = {"byteLength": bin_length}
    if buffer_uri is not None:
        buffer_def["uri"] = buffer_uri

    return {
        "asset": {"version": "2.0", "generator": "Neko Legends TripoSplat"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "NekoSplat Point Cloud"}],
        "meshes": [{
            "name": "NekoSplat Point Cloud",
            "primitives": [{
                "attributes": {"POSITION": 0, "COLOR_0": 1},
                "mode": 0,
            }],
        }],
        "buffers": [buffer_def],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(position_bytes), "byteLength": len(color_bytes), "target": 34962},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": point_count,
                "type": "VEC3",
                "min": xyz.min(axis=0).astype(float).tolist(),
                "max": xyz.max(axis=0).astype(float).tolist(),
            },
            {
                "bufferView": 1,
                "componentType": 5121,
                "count": point_count,
                "type": "VEC4",
                "normalized": True,
            },
        ],
    }, position_bytes, color_bytes


def _save_glb(path: Path, xyz: np.ndarray, rgba: np.ndarray):
    document, position_bytes, color_bytes = _gltf_document(len(xyz), xyz, rgba, None)
    json_chunk = _pad4(json.dumps(document, separators=(",", ":")).encode("utf-8"), b" ")
    bin_chunk = _pad4(position_bytes + color_bytes)
    total_length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    with path.open("wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total_length))
        f.write(struct.pack("<I4s", len(json_chunk), b"JSON"))
        f.write(json_chunk)
        f.write(struct.pack("<I4s", len(bin_chunk), b"BIN\x00"))
        f.write(bin_chunk)


def _save_gltf_zip(path: Path, xyz: np.ndarray, rgba: np.ndarray):
    document, position_bytes, color_bytes = _gltf_document(len(xyz), xyz, rgba, "splat_points.bin")
    with ZipFile(path, "w", ZIP_DEFLATED) as zf:
        zf.writestr("splat_points.gltf", json.dumps(document, indent=2))
        zf.writestr("splat_points.bin", position_bytes + color_bytes)


def _save_obj(path: Path, xyz: np.ndarray, rgba: np.ndarray):
    rgb = rgba[:, :3].astype(np.float32) / 255.0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Neko Legends TripoSplat point-cloud OBJ export\n")
        f.write("# Vertex colors are stored as non-standard v x y z r g b values.\n")
        for point, color in zip(xyz, rgb):
            f.write(
                f"v {point[0]:.7g} {point[1]:.7g} {point[2]:.7g} "
                f"{color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n"
            )


def _write_fbx_array(f, values: np.ndarray, precision: int = 7, per_line: int = 24):
    flat = values.reshape(-1)
    fmt = f"{{:.{precision}g}}"
    for start in range(0, len(flat), per_line):
        chunk = ",".join(fmt.format(float(v)) for v in flat[start:start + per_line])
        f.write(f"\t\t\ta: {chunk}\n")


def _save_fbx(path: Path, xyz: np.ndarray, rgba: np.ndarray):
    colors = rgba.astype(np.float32) / 255.0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("; Neko Legends TripoSplat point-cloud FBX export\n")
        f.write("FBXHeaderExtension:  {\n\tFBXHeaderVersion: 1003\n\tFBXVersion: 7400\n}\n")
        f.write("GlobalSettings:  {\n\tVersion: 1000\n}\n")
        f.write("Objects:  {\n")
        f.write('\tGeometry: 1000, "Geometry::NekoSplatPoints", "Mesh" {\n')
        f.write(f"\t\tVertices: *{xyz.size} {{\n")
        _write_fbx_array(f, xyz)
        f.write("\t\t}\n")
        f.write("\t\tPolygonVertexIndex: *0 {\n\t\t\ta: \n\t\t}\n")
        f.write('\t\tLayerElementColor: 0 {\n\t\t\tVersion: 101\n\t\t\tName: ""\n')
        f.write('\t\t\tMappingInformationType: "ByVertice"\n\t\t\tReferenceInformationType: "Direct"\n')
        f.write(f"\t\t\tColors: *{colors.size} {{\n")
        _write_fbx_array(f, colors, precision=6)
        f.write("\t\t\t}\n\t\t}\n")
        f.write('\t\tLayer: 0 {\n\t\t\tVersion: 100\n\t\t\tLayerElement:  {\n\t\t\t\tType: "LayerElementColor"\n\t\t\t\tTypedIndex: 0\n\t\t\t}\n\t\t}\n')
        f.write("\t}\n")
        f.write('\tModel: 2000, "Model::NekoSplat Point Cloud", "Mesh" {\n\t\tVersion: 232\n\t}\n')
        f.write("}\n")
        f.write('Connections:  {\n\tC: "OO",1000,2000\n}\n')


def _safe_output_stem(value: str | None) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in (value or "").strip()
    ).strip("_")
    return cleaned or "neko_splat"


def _export_common_formats(gaussian, out_dir: Path, output_name: str | None = None):
    xyz, rgba = _point_cloud_data(gaussian)
    stem = _safe_output_stem(output_name)
    paths = {
        "prepared": out_dir / f"{stem}_prepared.webp",
        "ply": out_dir / f"{stem}_native.ply",
        "splat": out_dir / f"{stem}_native.splat",
        "glb": out_dir / f"{stem}_point_cloud.glb",
        "gltf": out_dir / f"{stem}_point_cloud_gltf.zip",
        "obj": out_dir / f"{stem}_point_cloud.obj",
        "fbx": out_dir / f"{stem}_point_cloud.fbx",
    }
    gaussian.save_ply(str(paths["ply"]))
    gaussian.save_splat(str(paths["splat"]))
    _save_glb(paths["glb"], xyz, rgba)
    _save_gltf_zip(paths["gltf"], xyz, rgba)
    _save_obj(paths["obj"], xyz, rgba)
    _save_fbx(paths["fbx"], xyz, rgba)
    return paths


def _download(path: Path):
    return _ensure_gradio().update(value=str(path), interactive=True)


def _missing_model_files():
    return [
        (label, rel_path, CKPT_ROOT / rel_path)
        for label, rel_path in MODEL_FILES
        if not (CKPT_ROOT / rel_path).exists()
    ]


def _setup_status_text() -> str:
    missing = _missing_model_files()
    if not missing:
        return (
            f"Setup complete. Model files are installed in `{CKPT_ROOT}`. "
            "You can generate splats now."
        )
    missing_lines = "\n".join(f"- `{path}`" for _, _, path in missing)
    return (
        "Setup is needed before generation. The app can download the expected "
        f"TripoSplat `.safetensors` files from `{TRIPOSPLAT_REPO_ID}` into `{CKPT_ROOT}`.\n\n"
        f"{missing_lines}"
    )


def open_setup():
    return _ensure_gradio().update(visible=True), _setup_status_text()


def close_setup():
    return _ensure_gradio().update(visible=False)


def download_missing_model_files(progress_callback=None) -> dict[str, Any]:
    missing = _missing_model_files()
    if not missing:
        return {
            "ok": True,
            "ckptRoot": str(CKPT_ROOT),
            "missing": [],
            "message": _setup_status_text(),
        }

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "The setup downloader needs huggingface_hub. The portable build includes it; "
            f"source runs can install it with pip. Details: {exc}"
        ) from exc

    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    total = len(missing)
    for index, (label, rel_path, _) in enumerate(missing, start=1):
        if progress_callback is not None:
            progress_callback(f"Downloading {label}", (index - 1) / total)
        hf_hub_download(
            repo_id=TRIPOSPLAT_REPO_ID,
            filename=rel_path,
            local_dir=str(CKPT_ROOT),
        )
    if progress_callback is not None:
        progress_callback("Setup complete", 1)
    return {
        "ok": True,
        "ckptRoot": str(CKPT_ROOT),
        "missing": [],
        "message": _setup_status_text(),
    }


def download_setup_files(progress=None):
    progress = progress or _ensure_gradio().Progress()
    try:
        download_missing_model_files(
            progress_callback=lambda message, value: progress(value, desc=message)
        )
    except Exception as exc:
        raise _ensure_gradio().Error(str(exc)) from exc
    return _ensure_gradio().update(visible=False), _setup_status_text()


def _port_available(port: int, host: str) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            return sock.connect_ex((host, port)) != 0
    except OSError:
        return False


def _server_port(requested: int | None = None, host: str | None = None) -> int:
    if requested:
        return requested
    requested_env = os.environ.get("TRIPOSPLAT_PORT")
    if requested_env:
        return int(requested_env)
    host = host or os.environ.get("TRIPOSPLAT_HOST", "127.0.0.1")
    for port in range(7860, 7875):
        if _port_available(port, host):
            return port
    return 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _shared_neko_legends_dir() -> Path | None:
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.environ.get("USERPROFILE")
        return Path(base) / "NekoLegends" if base else None
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "NekoLegends"
    base = os.environ.get("XDG_CONFIG_HOME")
    return Path(base or (Path.home() / ".config")) / "NekoLegends"


def _agent_api_registry_path() -> Path | None:
    root = _shared_neko_legends_dir()
    return root / AGENT_API_REGISTRY_FILE if root else None


def _agent_api_url(port: int) -> str:
    return f"http://{AGENT_API_BIND_ADDRESS}:{port}"


def _read_agent_api_registry() -> dict[str, Any]:
    path = _agent_api_registry_path()
    fallback = {"updatedAt": _now_iso(), "apps": []}
    if path is None or not path.exists():
        return fallback
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("apps"), list):
            return raw
    except Exception:
        pass
    return fallback


def _with_agent_api_registry_lock(path: Path, write_callback) -> None:
    lock_path = path.with_name(f"{path.name}.lock")
    for _ in range(4):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, str(os.getpid()).encode("ascii", errors="ignore"))
                write_callback()
            finally:
                os.close(fd)
                try:
                    lock_path.unlink()
                except OSError:
                    pass
            return
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > 2:
                    lock_path.unlink()
            except OSError:
                pass
            time.sleep(0.08)
    write_callback()


def _publish_agent_api_status(status: dict[str, Any] | None = None) -> None:
    path = _agent_api_registry_path()
    if path is None:
        return

    status = status or agent_api_status()
    now = _now_iso()
    entry = {
        "appId": AGENT_APP_ID,
        "appName": AGENT_APP_NAME,
        "defaultPort": DEFAULT_AGENT_API_PORT,
        "bindAddress": AGENT_API_BIND_ADDRESS,
        "port": status["port"],
        "enabled": status["enabled"],
        "url": status["url"],
        "openapiUrl": status["openapiUrl"],
        "busy": status["busy"],
        "activeJobId": status["activeJobId"],
        "lastSeen": now,
        "note": "Local Agent API.",
    }

    def write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        registry = _read_agent_api_registry()
        apps = registry.setdefault("apps", [])
        for index, app in enumerate(apps):
            if app.get("appId") == AGENT_APP_ID:
                apps[index] = entry
                break
        else:
            apps.append(entry)
        registry["updatedAt"] = now
        path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    try:
        _with_agent_api_registry_lock(path, write)
    except Exception:
        pass


def _read_registered_agent_api_port() -> int | None:
    for entry in _read_agent_api_registry().get("apps", []):
        if entry.get("appId") == AGENT_APP_ID:
            try:
                port = int(entry.get("port"))
                return port if port > 0 else None
            except (TypeError, ValueError):
                return None
    return None


def _resolve_agent_api_port(port: int | str | None = None) -> int:
    candidates = [
        port,
        os.environ.get("TRIPOSPLAT_AGENT_API_PORT"),
        _read_registered_agent_api_port(),
        DEFAULT_AGENT_API_PORT,
    ]
    for candidate in candidates:
        if candidate is None or candidate == "":
            continue
        try:
            parsed = int(candidate)
        except (TypeError, ValueError):
            continue
        if 0 < parsed <= 65535:
            return parsed
    raise ValueError("Agent API port must be between 1 and 65535.")


class AgentBusyError(RuntimeError):
    pass


class AgentServerState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.enabled = False
        self.port = _resolve_agent_api_port()
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.heartbeat_stop: threading.Event | None = None
        self.active_job: dict[str, Any] | None = None
        self.last_job: dict[str, Any] | None = None


AGENT_SERVER = AgentServerState()


def _model_status() -> dict[str, Any]:
    missing = [
        {"label": label, "relativePath": rel_path, "path": str(path)}
        for label, rel_path, path in _missing_model_files()
    ]
    return {
        "ready": not missing,
        "ckptRoot": str(CKPT_ROOT),
        "missing": missing,
    }


def agent_api_status() -> dict[str, Any]:
    with AGENT_SERVER.lock:
        enabled = AGENT_SERVER.enabled
        port = AGENT_SERVER.port
        active_job = dict(AGENT_SERVER.active_job) if AGENT_SERVER.active_job else None
        last_job = dict(AGENT_SERVER.last_job) if AGENT_SERVER.last_job else None
    return {
        "ok": True,
        "service": APP_NAME,
        "version": APP_VERSION,
        "enabled": enabled,
        "port": port,
        "url": _agent_api_url(port),
        "openapiUrl": f"{_agent_api_url(port)}/openapi.json",
        "busy": active_job is not None,
        "activeJobId": active_job.get("id") if active_job else None,
        "activeJob": active_job,
        "lastJob": last_job,
        "outputRoot": str(OUT_ROOT),
        "models": _model_status(),
    }


def _agent_openapi(port: int) -> dict[str, Any]:
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "ImageToSplat Agent API",
            "version": APP_VERSION,
        },
        "servers": [{"url": _agent_api_url(port)}],
        "paths": {
            "/health": {"get": {"summary": "Check API status"}},
            "/openapi.json": {"get": {"summary": "Fetch this OpenAPI document"}},
            "/status": {"get": {"summary": "Check active and last job status"}},
            "/models": {"get": {"summary": "Check model setup status"}},
            "/setup": {"post": {"summary": "Download missing TripoSplat model files"}},
            "/generate": {
                "post": {
                    "summary": "Start an image-to-splat generation job",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "imagePath": {"type": "string"},
                                        "paths": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "options": {"type": "object"},
                                        "outputDir": {"type": "string"},
                                        "outputName": {"type": "string"},
                                        "seed": {"type": "integer"},
                                        "steps": {"type": "integer"},
                                        "guidanceScale": {"type": "number"},
                                        "numGaussians": {"type": "integer"},
                                    },
                                }
                            }
                        },
                    },
                }
            },
        },
    }


def _copy_job(job: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(job, default=str))


def _set_active_job_fields(job_id: str, **fields: Any) -> None:
    with AGENT_SERVER.lock:
        if AGENT_SERVER.active_job and AGENT_SERVER.active_job.get("id") == job_id:
            AGENT_SERVER.active_job.update(fields)


def _finish_agent_job(job_id: str, **fields: Any) -> None:
    with AGENT_SERVER.lock:
        if not AGENT_SERVER.active_job or AGENT_SERVER.active_job.get("id") != job_id:
            return
        job = AGENT_SERVER.active_job
        job.update(fields)
        job["finishedAt"] = _now_iso()
        AGENT_SERVER.last_job = _copy_job(job)
        AGENT_SERVER.active_job = None
    _publish_agent_api_status()


def _agent_progress(job_id: str, message: str, progress: float | None) -> None:
    fields: dict[str, Any] = {"message": message}
    if progress is not None:
        fields["progress"] = round(float(progress), 3)
    _set_active_job_fields(job_id, **fields)
    _publish_agent_api_status()


def _nested_options(payload: dict[str, Any]) -> dict[str, Any]:
    options = payload.get("options")
    return options if isinstance(options, dict) else {}


def _payload_value(payload: dict[str, Any], options: dict[str, Any], *names: str, default=None):
    for name in names:
        if name in payload and payload[name] not in (None, ""):
            return payload[name]
        if name in options and options[name] not in (None, ""):
            return options[name]
    return default


def _agent_generate_options(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    options = _nested_options(payload)
    paths = payload.get("paths")
    image_path = _payload_value(payload, options, "imagePath", "image_path")
    if not image_path and isinstance(paths, list) and paths:
        image_path = paths[0]
    if not image_path:
        raise ValueError("Request body must include imagePath or paths[0].")
    return {
        "image": str(Path(str(image_path)).expanduser().resolve()),
        "seed": _payload_value(payload, options, "seed", default=42),
        "steps": _payload_value(payload, options, "steps", default=20),
        "guidance_scale": _payload_value(
            payload,
            options,
            "guidanceScale",
            "guidance_scale",
            default=3.0,
        ),
        "num_gaussians": _payload_value(
            payload,
            options,
            "numGaussians",
            "num_gaussians",
            default=262144,
        ),
        "output_dir": _payload_value(payload, options, "outputDir", "output_dir"),
        "output_name": _payload_value(payload, options, "outputName", "output_name"),
    }


def _run_agent_job(job_id: str, action: str, payload: dict[str, Any]) -> None:
    try:
        if action == "setup":
            result = download_missing_model_files(
                progress_callback=lambda message, value: _agent_progress(job_id, message, value)
            )
        elif action == "generate":
            request = _agent_generate_options(payload)
            result = run_generation_core(
                request["image"],
                seed=request["seed"],
                steps=request["steps"],
                guidance_scale=request["guidance_scale"],
                num_gaussians=request["num_gaussians"],
                output_dir=request["output_dir"],
                output_name=request["output_name"],
                job_id=job_id,
                progress_callback=lambda message, value: _agent_progress(job_id, message, value),
            )
        else:
            raise ValueError(f"Unsupported agent job action: {action}")
        _finish_agent_job(
            job_id,
            status="completed",
            progress=1,
            message="Completed",
            result=result,
        )
    except Exception as exc:
        _finish_agent_job(
            job_id,
            status="failed",
            message=str(exc),
            error=str(exc),
            traceback=traceback.format_exc(),
        )


def _start_agent_job(action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    job_id = f"splat-{uuid4().hex[:12]}"
    job = {
        "id": job_id,
        "action": action,
        "status": "running",
        "message": "Queued",
        "progress": 0,
        "createdAt": _now_iso(),
    }
    with AGENT_SERVER.lock:
        if AGENT_SERVER.active_job is not None:
            raise AgentBusyError("ImageToSplat is already running an agent job.")
        AGENT_SERVER.active_job = job
    _publish_agent_api_status()
    threading.Thread(
        target=_run_agent_job,
        args=(job_id, action, payload),
        daemon=True,
    ).start()
    return {"ok": True, "jobId": job_id, "statusUrl": f"{agent_api_status()['url']}/status"}


def _normalize_agent_path(path: str) -> str:
    clean = path.split("?", 1)[0]
    if clean.startswith("/api/v1/"):
        return clean[len("/api/v1"):]
    return clean


class AgentApiHandler(BaseHTTPRequestHandler):
    server_version = "ImageToSplatAgent/1.0"

    def log_message(self, _format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        if length > 2 * 1024 * 1024:
            raise ValueError("Agent request is too large.")
        raw = self.rfile.read(length)
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("Request body must be a JSON object.")
        return parsed

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        path = _normalize_agent_path(self.path)
        try:
            if path == "/health":
                _publish_agent_api_status()
                self._send_json(200, {"ok": True, "service": APP_NAME, "status": agent_api_status()})
            elif path == "/openapi.json":
                self._send_json(200, _agent_openapi(agent_api_status()["port"]))
            elif path == "/status":
                _publish_agent_api_status()
                self._send_json(200, agent_api_status())
            elif path == "/models":
                self._send_json(200, {"ok": True, "models": _model_status()})
            else:
                self._send_json(404, {"ok": False, "error": f"No agent endpoint for GET {path}"})
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})

    def do_POST(self) -> None:
        path = _normalize_agent_path(self.path)
        try:
            payload = self._read_json()
            if path == "/setup":
                self._send_json(200, _start_agent_job("setup", payload))
            elif path == "/generate":
                self._send_json(200, _start_agent_job("generate", payload))
            else:
                self._send_json(404, {"ok": False, "error": f"No agent endpoint for POST {path}"})
        except AgentBusyError as exc:
            self._send_json(409, {"ok": False, "error": str(exc), "status": agent_api_status()})
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})


def _agent_heartbeat_loop(stop_event: threading.Event) -> None:
    while not stop_event.wait(15):
        _publish_agent_api_status()


def start_agent_api_server(port: int | str | None = None) -> dict[str, Any]:
    resolved_port = _resolve_agent_api_port(port)
    with AGENT_SERVER.lock:
        if AGENT_SERVER.enabled and AGENT_SERVER.port == resolved_port:
            return agent_api_status()
    if AGENT_SERVER.enabled:
        stop_agent_api_server()

    server = ThreadingHTTPServer((AGENT_API_BIND_ADDRESS, resolved_port), AgentApiHandler)
    server.daemon_threads = True
    stop_event = threading.Event()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    heartbeat_thread = threading.Thread(
        target=_agent_heartbeat_loop,
        args=(stop_event,),
        daemon=True,
    )
    with AGENT_SERVER.lock:
        AGENT_SERVER.enabled = True
        AGENT_SERVER.port = resolved_port
        AGENT_SERVER.server = server
        AGENT_SERVER.thread = thread
        AGENT_SERVER.heartbeat_stop = stop_event
    thread.start()
    heartbeat_thread.start()
    _publish_agent_api_status()
    return agent_api_status()


def stop_agent_api_server() -> dict[str, Any]:
    with AGENT_SERVER.lock:
        server = AGENT_SERVER.server
        stop_event = AGENT_SERVER.heartbeat_stop
        AGENT_SERVER.enabled = False
        AGENT_SERVER.server = None
        AGENT_SERVER.thread = None
        AGENT_SERVER.heartbeat_stop = None
    if stop_event is not None:
        stop_event.set()
    if server is not None:
        server.shutdown()
        server.server_close()
    status = agent_api_status()
    _publish_agent_api_status(status)
    return status


def agent_api_status_markdown() -> str:
    status = agent_api_status()
    state = "enabled" if status["enabled"] else "off"
    busy = "busy" if status["busy"] else "idle"
    registry = _agent_api_registry_path()
    registry_line = f"\nRegistry: `{registry}`" if registry else ""
    return (
        f"Agent API is **{state}** at `{status['url']}` ({busy}).\n\n"
        f"OpenAPI: `{status['openapiUrl']}`\n"
        f"Models: {'ready' if status['models']['ready'] else 'setup needed'}"
        f"{registry_line}"
    )


def refresh_agent_api_controls():
    status = agent_api_status()
    return agent_api_status_markdown(), _ensure_gradio().update(value=status["port"])


def enable_agent_api_control(port):
    try:
        status = start_agent_api_server(port)
        return agent_api_status_markdown(), _ensure_gradio().update(value=status["port"])
    except Exception as exc:
        raise _ensure_gradio().Error(f"Unable to start Agent API: {exc}") from exc


def disable_agent_api_control():
    status = stop_agent_api_server()
    return agent_api_status_markdown(), _ensure_gradio().update(value=status["port"])


def _launch_agent_api_from_env() -> bool:
    return _env_truthy("TRIPOSPLAT_AGENT_API")


def _serve_agent_api_forever(port: int | str | None = None) -> int:
    status = start_agent_api_server(port)
    print(json.dumps(status, indent=2), flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        stop_agent_api_server()
    return 0


def _run_headless(args: argparse.Namespace) -> int:
    if not args.inputs:
        print("No input image was provided for --headless.", file=sys.stderr)
        return 2
    result = run_generation_core(
        args.inputs[0],
        seed=args.seed,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        num_gaussians=args.num_gaussians,
        output_dir=args.output_dir,
        output_name=args.output_name,
        progress_callback=lambda message, value: print(
            json.dumps({"event": "progress", "message": message, "progress": value}),
            flush=True,
        ),
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Neko Legends ImageToSplat")
    parser.add_argument("inputs", nargs="*", help="Input image paths for headless generation.")
    parser.add_argument("--headless", action="store_true", help="Generate without launching the Gradio UI.")
    parser.add_argument("--serve-agent-api", action="store_true", help="Run only the local Agent API.")
    parser.add_argument("--agent-api", action="store_true", help="Start the local Agent API beside the Gradio UI.")
    parser.add_argument("--agent-api-port", type=int, default=None, help="Agent API port. Default: 17340 or registry value.")
    parser.add_argument("--host", default=None, help="Gradio host. Default: TRIPOSPLAT_HOST or 127.0.0.1.")
    parser.add_argument("--port", type=int, default=None, help="Gradio port. Default: TRIPOSPLAT_PORT or first free 7860-7874.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser for the Gradio UI.")
    parser.add_argument("--output-dir", default=None, help="Headless output directory.")
    parser.add_argument("--output-name", default=None, help="Headless output file stem.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--num-gaussians", type=int, default=262144)
    return parser


def _main() -> int:
    args = _cli_parser().parse_args()
    if args.no_browser:
        os.environ["TRIPOSPLAT_OPEN_BROWSER"] = "0"

    if args.headless or _env_truthy("TRIPOSPLAT_HEADLESS"):
        return _run_headless(args)
    if args.serve_agent_api:
        return _serve_agent_api_forever(args.agent_api_port)
    if args.agent_api or _launch_agent_api_from_env():
        start_agent_api_server(args.agent_api_port)

    demo = build_demo()
    demo.launch(
        server_name=args.host or os.environ.get("TRIPOSPLAT_HOST", "127.0.0.1"),
        server_port=_server_port(args.port, args.host),
        allowed_paths=[
            str(STATIC_DIR),
            str(OUT_ROOT),
            str(EXAMPLES_DIR),
        ],
        css=NEKO_CSS,
        inbrowser=os.environ.get("TRIPOSPLAT_OPEN_BROWSER", "1") != "0",
    )
    return 0


def _get_pipe():
    global PIPE, TORCH
    if PIPE is not None:
        return PIPE

    missing = [path for _, _, path in _missing_model_files()]
    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        raise RuntimeError(
            "TripoSplat model weights are missing. Open Setup and download the required files.\n\n"
            f"{missing_list}"
        )

    import torch
    from triposplat import TripoSplatPipeline

    TORCH = torch
    device = os.environ.get("TRIPOSPLAT_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    pipe_paths = {key: str(path) for key, path in PIPE_PATHS.items()}
    PIPE = TripoSplatPipeline(**pipe_paths, device=device)
    return PIPE


def _load_input_image(image):
    if image is None:
        raise ValueError("Please provide an input image.")
    if isinstance(image, (str, os.PathLike, Path)):
        from PIL import Image

        image_path = Path(image).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Input image was not found: {image_path}")
        with Image.open(image_path) as opened:
            return opened.convert("RGBA")
    return image


def _coerce_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _coerce_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _path_map(paths: dict[str, Path]) -> dict[str, str]:
    return {key: str(path) for key, path in paths.items()}


def run_generation_core(
    image,
    *,
    seed: int = 42,
    steps: int = 20,
    guidance_scale: float = 3.0,
    num_gaussians: int = 262144,
    output_dir: str | Path | None = None,
    output_name: str | None = None,
    job_id: str | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    """Run TripoSplat once and write every supported export format."""
    input_image = _load_input_image(image)
    seed = _coerce_int(seed, 42, 0, 2**31 - 1)
    steps = _coerce_int(steps, 20, 1, 50)
    guidance_scale = _coerce_float(guidance_scale, 3.0, 1.0, 10.0)
    num_gaussians = _coerce_int(num_gaussians, 262144, 1, 262144)

    def report(message: str, progress: float | None = None) -> None:
        if progress_callback is not None:
            progress_callback(message, progress)

    report("Loading TripoSplat", 0.0)
    pipe = _get_pipe()

    report("Preprocessing image", 0.1)
    t0 = time.time()
    prepared = pipe.preprocess_image(input_image)

    report("Generating splat", 0.25)
    gen = TORCH.Generator(device=pipe._device).manual_seed(seed)
    cond = pipe.encode_image(prepared, generator=gen)
    out = pipe.sample_latent(
        cond,
        steps=steps,
        guidance_scale=guidance_scale,
        generator=gen,
        show_progress=True,
    )

    report("Decoding gaussians", 0.82)
    gaussian = pipe.decode_latent(out["latent"], num_gaussians=num_gaussians)
    gen_dt = time.time() - t0

    out_dir = Path(output_dir).expanduser().resolve() if output_dir else OUT_ROOT / (job_id or uuid4().hex[:12])
    out_dir.mkdir(parents=True, exist_ok=True)
    export_paths = _export_common_formats(gaussian, out_dir, output_name)
    prepared.save(export_paths["prepared"])

    report("Finished", 1.0)
    gaussian_count = int(gaussian.get_xyz.shape[0])
    return {
        "ok": True,
        "jobId": job_id,
        "outputDir": str(out_dir),
        "gaussianCount": gaussian_count,
        "generationSeconds": round(gen_dt, 3),
        "paths": _path_map(export_paths),
        "settings": {
            "seed": seed,
            "steps": steps,
            "guidanceScale": guidance_scale,
            "numGaussians": num_gaussians,
        },
    }


# ----------------------------------------------------------------------------
# Event handlers
# ----------------------------------------------------------------------------

def generate(image, seed: int, steps: int, guidance_scale: float,
             num_gaussians: int,
             progress=None):
    """Run the full pipeline (preprocess + encode + sample + decode)."""
    ui = _ensure_gradio()
    progress = progress or ui.Progress(track_tqdm=True)
    if image is None:
        raise ui.Error("Please upload an image first.")

    def gradio_progress(message: str, value: float | None) -> None:
        if value is not None:
            progress(value, desc=message)

    result = run_generation_core(
        image,
        seed=int(seed),
        steps=int(steps),
        guidance_scale=float(guidance_scale),
        num_gaussians=int(num_gaussians),
        progress_callback=gradio_progress,
    )
    export_paths = {key: Path(path) for key, path in result["paths"].items()}

    info = (f"{result['gaussianCount']:,} gaussians  ·  "
            f"generation: {result['generationSeconds']:.1f}s  ·  native preview: PLY  ·  "
            "common exports are point-cloud conversions")
    return (
        str(export_paths["prepared"]),
        _viewer_iframe(export_paths["ply"]),
        _download(export_paths["ply"]),
        _download(export_paths["splat"]),
        _download(export_paths["glb"]),
        _download(export_paths["gltf"]),
        _download(export_paths["obj"]),
        _download(export_paths["fbx"]),
        info,
    )


# ----------------------------------------------------------------------------
# Gradio UI
# ----------------------------------------------------------------------------

def build_demo():
    ui = _ensure_gradio()
    with ui.Blocks(title="Neko Legends Splat Forge") as demo:
        ui.HTML(
            "<header class='neko-topbar'>"
            "<div class='neko-brand'>"
            f"<img src='{_gr_file(APP_ICON)}' alt='Neko Legends'>"
            "<div><h1>Neko Legends Splat Forge</h1>"
            "<p>Image upload, TripoSplat generation, Spark.js 3D preview, and common 3D point-cloud exports.</p></div>"
            "</div>"
            "<span class='neko-status-pill'>Local 3D App</span>"
            "</header>"
            "<div class='neko-notice'>Native .ply/.splat keep Gaussian splat fidelity. glTF, GLB, OBJ, and FBX exports are point-cloud conversions for broader tool compatibility.</div>"
        )

        with ui.Group(visible=bool(_missing_model_files()), elem_id="setup-modal") as setup_modal:
            with ui.Column(elem_classes=["setup-card"]):
                ui.HTML(
                    "<p class='eyebrow'>First launch</p>"
                    "<h2>Setup TripoSplat files</h2>"
                    "<p>Download the model `.safetensors` files inside the app, then generate splats without hunting through folders.</p>"
                )
                setup_status = ui.Markdown(value=_setup_status_text(), elem_classes=["setup-status"])
                with ui.Row(elem_classes=["setup-actions"]):
                    setup_download_btn = ui.Button("Download Setup Files", variant="primary")
                    setup_later_btn = ui.Button("Later", variant="secondary")

        with ui.Row(elem_classes=["neko-layout"]):
            with ui.Column(scale=1, elem_classes=["neko-panel-left"]):
                setup_btn = ui.Button("Setup", variant="secondary")
                with ui.Accordion("AI Agent Control", open=False):
                    agent_port_in = ui.Number(
                        label="Agent API port",
                        value=agent_api_status()["port"],
                        precision=0,
                    )
                    agent_status = ui.Markdown(
                        value=agent_api_status_markdown(),
                        elem_classes=["info-text"],
                    )
                    with ui.Row(elem_classes=["setup-actions"]):
                        agent_enable_btn = ui.Button("Enable API", variant="primary")
                        agent_disable_btn = ui.Button("Disable", variant="secondary")
                        agent_refresh_btn = ui.Button("Refresh", variant="secondary")
                image_in = ui.Image(label="Input image", type="pil", image_mode="RGBA", height=320)

                ui.Examples(
                    examples=[[p] for p in EXAMPLES],
                    inputs=[image_in],
                    label="Examples (click to load)",
                    examples_per_page=10,
                    cache_examples=False,
                )

                with ui.Accordion("Sampling settings", open=False):
                    seed_in = ui.Number(label="Seed", value=42, precision=0)
                    steps_in = ui.Slider(label="Inference steps", minimum=1, maximum=50, step=1, value=20)
                    cfg_in = ui.Slider(label="Guidance scale", minimum=1.0, maximum=10.0, step=0.5, value=3.0)
                    num_g_in = ui.Dropdown(
                        label="Number of gaussians",
                        choices=["32768", "65536", "131072", "262144"],
                        value="262144",
                    )

                run_btn = ui.Button("Generate Splat", variant="primary")
                prepared_out = ui.Image(label="Preprocessed input", interactive=False, height=240)
                info_out = ui.Markdown(elem_classes=["info-text"])

            with ui.Column(scale=2, elem_classes=["neko-panel-right"]):
                ui.HTML("<div class='neko-viewer-header'><h2>3D Viewer</h2><span>Spark.js orbit preview</span></div>")
                viewer_out = ui.HTML(value=PLACEHOLDER_HTML, label="Spark.js viewer", elem_id="viewer-wrap")
                with ui.Row(elem_classes=["export-grid"]):
                    ply_out = ui.DownloadButton(label="PLY", value=None, interactive=False)
                    splat_out = ui.DownloadButton(label="SPLAT", value=None, interactive=False)
                    glb_out = ui.DownloadButton(label="GLB", value=None, interactive=False)
                    gltf_out = ui.DownloadButton(label="glTF ZIP", value=None, interactive=False)
                    obj_out = ui.DownloadButton(label="OBJ", value=None, interactive=False)
                    fbx_out = ui.DownloadButton(label="FBX", value=None, interactive=False)

        run_btn.click(
            fn=generate,
            inputs=[image_in, seed_in, steps_in, cfg_in, num_g_in],
            outputs=[prepared_out, viewer_out, ply_out, splat_out, glb_out, gltf_out, obj_out, fbx_out, info_out],
        )
        setup_btn.click(fn=open_setup, outputs=[setup_modal, setup_status])
        setup_later_btn.click(fn=close_setup, outputs=[setup_modal])
        setup_download_btn.click(fn=download_setup_files, outputs=[setup_modal, setup_status])
        agent_enable_btn.click(
            fn=enable_agent_api_control,
            inputs=[agent_port_in],
            outputs=[agent_status, agent_port_in],
        )
        agent_disable_btn.click(
            fn=disable_agent_api_control,
            outputs=[agent_status, agent_port_in],
        )
        agent_refresh_btn.click(
            fn=refresh_agent_api_controls,
            outputs=[agent_status, agent_port_in],
        )
    return demo


if __name__ == "__main__":
    raise SystemExit(_main())

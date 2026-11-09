"""Gaussian splat -> real polygon mesh (.glb with vertex colours) that any 3D tool or three.js GLTFLoader can open.

The splat is rendered (colour + depth) from many directions around the object, the depth maps are fused into a
signed distance volume (TSDF, with space carving from the silhouettes), and marching cubes extracts the surface.
Each vertex takes the average colour of the views that see it head-on. Shading and highlights stay baked into the
colours, as in the splat.
"""
import math
import time
from pathlib import Path

import numpy as np

UPS = {"y": (0, 1, 0), "-y": (0, -1, 0), "z": (0, 0, 1), "-z": (0, 0, -1)}


def load_ply(path: Path) -> dict:
    """Standard 3D Gaussian splat .ply (x y z, f_dc_*, opacity, scale_*, rot_*) as numpy columns."""
    with open(path, "rb") as f:
        names, n = [], 0
        while True:
            line = f.readline().decode(errors="replace").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                if line.split()[1] != "float":
                    raise ValueError(f"Unsupported .ply property type: {line}")
                names.append(line.split()[-1])
            elif line == "end_header":
                break
        data = np.frombuffer(f.read(n * 4 * len(names)), dtype=np.float32).reshape(n, len(names))
    return {k: data[:, i] for i, k in enumerate(names)}


def _write_unlit_glb(mesh, out: Path):
    """Write .glb with an unlit material (KHR_materials_unlit): the colours already contain the light and shading of
    the source image, so lighting them again in three.js, Blender or a game engine would wash them out."""
    import json
    import struct

    glb = mesh.export(file_type="glb")
    json_len = struct.unpack_from("<I", glb, 12)[0]
    doc = json.loads(glb[20:20 + json_len])
    rest = glb[20 + json_len:]
    for m in doc.get("materials") or []:
        m.setdefault("extensions", {})["KHR_materials_unlit"] = {}
        m["pbrMetallicRoughness"] = {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 1}
    if not doc.get("materials"):
        doc["materials"] = [{"extensions": {"KHR_materials_unlit": {}},
                             "pbrMetallicRoughness": {"metallicFactor": 0, "roughnessFactor": 1}}]
        for m in doc["meshes"]:
            for prim in m["primitives"]:
                prim["material"] = 0
    doc["extensionsUsed"] = sorted(set(doc.get("extensionsUsed", [])) | {"KHR_materials_unlit"})
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js + rest
    Path(out).write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body)


def mesh_from_file(path: Path, out: Path, up="-y", **kw) -> dict:
    """.ply (3DGS) or .spz splat -> .glb mesh."""
    path = Path(path)
    if path.suffix.lower() == ".spz":
        from .spz import read_spz

        d = read_spz(path)
        return splat_to_mesh(d["means"], d["quats_wxyz"], np.exp(d["log_scales"]), d["opacities"], d["sh0"], out, up=up, **kw)
    return mesh_from_ply(path, out, up=up, **kw)


def mesh_from_ply(ply: Path, out: Path, up="-y", **kw) -> dict:
    d = load_ply(Path(ply))
    col = lambda *k: np.stack([d[x] for x in k], 1)
    return splat_to_mesh(col("x", "y", "z"), col("rot_0", "rot_1", "rot_2", "rot_3"), np.exp(col("scale_0", "scale_1", "scale_2")),
                         1 / (1 + np.exp(-d["opacity"])), col("f_dc_0", "f_dc_1", "f_dc_2"), out, up=up, **kw)


def splat_to_mesh(means, quats_wxyz, scales, opacities, sh0, out: Path, up="-y", yaw=0.0, grid=256, views=128, size=512,
                  texture=0, faces=0, device="cuda", progress=lambda msg, frac=None: None) -> dict:
    """means/quats/scales (linear)/opacities (0..1)/sh0 (DC coefficients). `up` is the splat's up axis; the .glb is
    written y-up (then turned `yaw` degrees about y, so the front faces +z as glTF expects) with the object standing on
    y = 0, centred in x and z.
    texture: 0 = colour per vertex; N = simplify the mesh, unwrap it (xatlas) and bake an N x N JPEG texture.
    faces: simplify to about this many triangles (0 = keep all; textured meshes default to 100k)."""
    import torch
    import torch.nn.functional as F
    import trimesh
    from gsplat import rasterization
    from skimage.measure import marching_cubes

    t0 = time.time()
    target_faces = faces
    t = lambda a: torch.as_tensor(np.array(a), dtype=torch.float32, device=device)
    means, quats, scales, op, sh = t(means), F.normalize(t(quats_wxyz), dim=-1), t(scales), t(opacities), t(sh0)[:, None]
    solid = op > 0.3
    c = means[solid].median(0).values
    r = float((means[solid] - c).norm(dim=1).quantile(0.995))
    fov = math.radians(40)
    f = size / 2 / math.tan(fov / 2)
    dist = r / math.sin(fov / 2) * 1.05
    K = torch.tensor([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1]], device=device)
    U = torch.tensor(UPS[up], device=device, dtype=torch.float32)

    # views spread evenly over a sphere (Fibonacci lattice)
    i = torch.arange(views, device=device) + 0.5
    phi, th = torch.acos(1 - 2 * i / views), math.pi * (1 + 5 ** 0.5) * i
    dirs = torch.stack([torch.cos(th) * torch.sin(phi), torch.sin(th) * torch.sin(phi), torch.cos(phi)], 1)
    shots = []
    progress("Rendering the splat from all sides", 0.0)
    for dv in dirs:
        eye = c + dist * dv
        fwd = F.normalize(c - eye, dim=0)
        ref = U if abs(float(fwd @ U)) < 0.95 else torch.tensor([1.0, 0, 0], device=device)
        right = F.normalize(torch.linalg.cross(fwd, ref), dim=0)
        R = torch.stack([right, torch.linalg.cross(fwd, right), fwd])
        w2c = torch.eye(4, device=device)
        w2c[:3, :3], w2c[:3, 3] = R, -R @ eye
        with torch.no_grad():
            img, alpha, _ = rasterization(means, quats, scales, op, sh, w2c[None], K[None], size, size, sh_degree=0,
                                          render_mode="RGB+ED")
        shots.append((w2c, img[0, ..., :3].clamp(0, 1), img[0, ..., 3], alpha[0, ..., 0]))

    def project(P, w2c):
        p = P @ w2c[:3, :3].T + w2c[:3, 3]
        z = p[:, 2].clamp_min(1e-6)
        u, v = (p[:, 0] * f / z + size / 2).long(), (p[:, 1] * f / z + size / 2).long()
        idx = torch.where((u >= 0) & (u < size) & (v >= 0) & (v < size) & (p[:, 2] > 0))[0]
        return idx, u[idx], v[idx], p[idx, 2]

    progress("Fusing depth into a volume", 0.4)
    lo = c - r * 1.05
    vs = 2 * r * 1.05 / (grid - 1)
    ax = torch.arange(grid, device=device, dtype=torch.float32) * vs
    X = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3) + lo
    tsdf, wsum = torch.zeros(len(X), device=device), torch.zeros(len(X), device=device)
    trunc = 3 * vs
    for w2c, _, dep, al in shots:
        idx, u, v, z = project(X, w2c)
        sdf = dep[v, u] - z
        empty = al[v, u] < 0.5  # the silhouette shows background here: the voxel is outside the object
        val = torch.where(empty, torch.ones_like(sdf), (sdf / trunc).clamp(-1, 1))
        use = empty | (sdf > -trunc)
        ii = idx[use]
        tsdf[ii] = (tsdf[ii] * wsum[ii] + val[use]) / (wsum[ii] + 1)
        wsum[ii] += 1
    tsdf[wsum == 0] = -1  # never seen from outside: inside the object
    del X
    progress("Extracting the surface", 0.6)
    verts, faces, _, _ = marching_cubes(tsdf.reshape(grid, grid, grid).cpu().numpy(), 0.0, spacing=(vs, vs, vs))
    mesh = trimesh.Trimesh(verts + lo.cpu().numpy(), faces, process=True)
    parts = mesh.split(only_watertight=False)
    biggest = max(len(p.faces) for p in parts)
    mesh = trimesh.util.concatenate([p for p in parts if len(p.faces) > 0.01 * biggest])  # drop small floaters
    if (mesh.vertex_normals * (mesh.vertices - mesh.vertices.mean(0))).sum(1).mean() < 0:
        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[:, ::-1], process=False)  # normals point outwards

    if texture and not target_faces:
        target_faces = 100_000
    if target_faces and len(mesh.faces) > target_faces * 1.05:
        import fast_simplification

        progress("Simplifying the mesh", 0.7)
        sv, sf = fast_simplification.simplify(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32),
                                              target_reduction=1 - target_faces / len(mesh.faces))
        mesh = trimesh.Trimesh(sv, sf, process=True)

    def place(vertices):
        """y-up, front +z, standing on y = 0, centred in x and z."""
        Uv = np.array(UPS[up], dtype=np.float64)
        v = vertices
        if up != "y":
            a = np.cross(Uv, [0, 1, 0])
            if np.linalg.norm(a) < 1e-9:  # up is -y: rotate 180 degrees about x
                Rm = np.diag([1.0, -1.0, -1.0])
            else:
                a /= np.linalg.norm(a)
                A = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                Rm = np.eye(3) + A + A @ A  # 90 degrees: sin = 1, 1 - cos = 1
            v = v @ Rm.T
        if yaw:
            cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            v = v @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]).T
        lo_, hi_ = v.min(0), v.max(0)
        return v - np.array([(lo_[0] + hi_[0]) / 2, lo_[1], (lo_[2] + hi_[2]) / 2])

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if texture:
        progress("Unwrapping the mesh", 0.75)
        tex, uv, vmap, tris = _bake_texture(mesh, texture, dirs, c, dist, U, means, quats, scales, op, sh, fov, vs,
                                            device, progress)
        _write_textured_glb(place(mesh.vertices[vmap]), uv, tris, tex, out)
        return {"glb": str(out), "vertices": int(len(vmap)), "faces": int(len(tris)), "texture": texture,
                "sizeBytes": out.stat().st_size, "seconds": round(time.time() - t0, 1)}

    progress("Colouring the mesh", 0.8)
    V, Nrm = t(mesh.vertices), t(mesh.vertex_normals)
    col, wc = torch.zeros(len(V), 3, device=device), torch.zeros(len(V), device=device)
    for w2c, rgb, dep, _ in shots:
        idx, u, v, z = project(V, w2c)
        visible = (dep[v, u] - z).abs() < 1.5 * vs
        cam = -w2c[:3, :3].T @ w2c[:3, 3]
        w = (F.normalize(cam - V[idx], dim=1) * Nrm[idx]).sum(1).clamp_min(0) ** 2 * visible
        col[idx] += rgb[v, u] * w[:, None]
        wc[idx] += w
    col = (col / wc.clamp_min(1e-6)[:, None]).clamp(0, 1) ** 2.2  # glTF vertex colours are linear

    vv = place(mesh.vertices)
    rgba = (np.c_[col.cpu().numpy(), np.ones(len(vv))] * 255).round().astype(np.uint8)
    _write_unlit_glb(trimesh.Trimesh(vv, mesh.faces, vertex_colors=rgba, process=False), out)
    return {"glb": str(out), "vertices": int(len(vv)), "faces": int(len(mesh.faces)), "sizeBytes": out.stat().st_size,
            "seconds": round(time.time() - t0, 1)}


def _bake_texture(mesh, size, dirs, c, dist, U, means, quats, scales, op, sh, fov, vs, device, progress, render=1024):
    """xatlas unwrap, then every texel takes the colour of the splat renders that see its surface point head-on."""
    import torch
    import torch.nn.functional as F
    import xatlas
    from gsplat import rasterization

    vmap, tris, uv = xatlas.parametrize(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
    P = torch.tensor(mesh.vertices[vmap], dtype=torch.float32, device=device)
    T = torch.tensor(tris.astype(np.int64), device=device)
    UVp = torch.tensor(uv, dtype=torch.float32, device=device) * size  # texel units, row = v * size

    # rasterize triangles in UV space: per texel, the 3D point and face normal
    a, b, cc = UVp[T[:, 0]], UVp[T[:, 1]], UVp[T[:, 2]]
    lo = torch.minimum(torch.minimum(a, b), cc).floor().clamp(0, size - 1).long()
    hi = torch.maximum(torch.maximum(a, b), cc).ceil().clamp(0, size - 1).long()
    wh = (hi - lo + 1)
    fn = F.normalize(torch.linalg.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]), dim=1)
    pos = torch.zeros(size * size, 3, device=device)
    nrm = torch.zeros(size * size, 3, device=device)
    filled = torch.zeros(size * size, dtype=torch.bool, device=device)
    for s0 in range(0, len(T), 20000):
        sl = slice(s0, s0 + 20000)
        n = (wh[sl, 0] * wh[sl, 1])
        tid = torch.repeat_interleave(torch.arange(s0, s0 + len(n), device=device), n)
        k = torch.arange(len(tid), device=device) - torch.repeat_interleave(torch.cumsum(n, 0) - n, n)
        x = lo[tid, 0] + k % wh[tid, 0]
        y = lo[tid, 1] + k // wh[tid, 0]
        q = torch.stack([x, y], 1).float() + 0.5
        A, B, C = a[tid], b[tid], cc[tid]
        den = (B[:, 1] - C[:, 1]) * (A[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (A[:, 1] - C[:, 1])
        den = torch.where(den.abs() < 1e-12, torch.full_like(den, 1e-12), den)
        w0 = ((B[:, 1] - C[:, 1]) * (q[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (q[:, 1] - C[:, 1])) / den
        w1 = ((C[:, 1] - A[:, 1]) * (q[:, 0] - C[:, 0]) + (A[:, 0] - C[:, 0]) * (q[:, 1] - C[:, 1])) / den
        w2 = 1 - w0 - w1
        eps = -0.02
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        tid, x, y, w0, w1, w2 = tid[inside], x[inside], y[inside], w0[inside], w1[inside], w2[inside]
        ti = T[tid]
        idx = y * size + x
        pos[idx] = P[ti[:, 0]] * w0[:, None] + P[ti[:, 1]] * w1[:, None] + P[ti[:, 2]] * w2[:, None]
        nrm[idx] = fn[tid]
        filled[idx] = True
    tex_idx = torch.where(filled)[0]
    X, N = pos[tex_idx], nrm[tex_idx]

    progress("Baking the texture", 0.8)
    f = render / 2 / math.tan(fov / 2)
    K = torch.tensor([[f, 0, render / 2], [0, f, render / 2], [0, 0, 1]], device=device)
    col = torch.zeros(len(X), 3, device=device)
    wc = torch.zeros(len(X), device=device)
    for dv in dirs:
        eye = c + dist * dv
        fwd = F.normalize(c - eye, dim=0)
        ref = U if abs(float(fwd @ U)) < 0.95 else torch.tensor([1.0, 0, 0], device=device)
        right = F.normalize(torch.linalg.cross(fwd, ref), dim=0)
        R = torch.stack([right, torch.linalg.cross(fwd, right), fwd])
        w2c = torch.eye(4, device=device)
        w2c[:3, :3], w2c[:3, 3] = R, -R @ eye
        with torch.no_grad():
            img, _, _ = rasterization(means, quats, scales, op, sh, w2c[None], K[None], render, render, sh_degree=0,
                                      render_mode="RGB+ED")
        rgb, dep = img[0, ..., :3].clamp(0, 1), img[0, ..., 3]
        p = X @ R.T + w2c[:3, 3]
        z = p[:, 2].clamp_min(1e-6)
        u, v = (p[:, 0] * f / z + render / 2).long(), (p[:, 1] * f / z + render / 2).long()
        ok = torch.where((u >= 0) & (u < render) & (v >= 0) & (v < render) & (p[:, 2] > 0))[0]
        vis = (dep[v[ok], u[ok]] - z[ok]).abs() < 1.5 * vs
        facing = (F.normalize(eye - X[ok], dim=1) * N[ok]).sum(1).clamp_min(0)
        w = facing ** 4 * vis
        col[ok] += rgb[v[ok], u[ok]] * w[:, None]
        wc[ok] += w
    img = torch.zeros(size * size, 3, device=device)
    have = torch.zeros(size * size, device=device)
    seen = wc > 0
    img[tex_idx[seen]] = col[seen] / wc[seen, None]
    have[tex_idx[seen]] = 1
    # pad charts outwards so texture filtering at seams never picks up empty texels
    img, have = img.T.reshape(1, 3, size, size), have.reshape(1, 1, size, size)
    for _ in range(16):
        s_ = F.avg_pool2d(img * have, 3, 1, 1)
        h_ = F.avg_pool2d(have, 3, 1, 1)
        grow = (have == 0) & (h_ > 0)
        img = torch.where(grow, s_ / h_.clamp_min(1e-6), img)
        have = torch.where(grow, torch.ones_like(have), have)
    tex = (img[0].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()
    return tex, uv, vmap, tris


def _write_textured_glb(vertices, uv, tris, tex, out: Path, quality=90):
    """Minimal glTF 2.0 binary: positions, UVs, indices, one unlit material with a JPEG base-colour texture."""
    import io
    import json
    import struct

    from PIL import Image

    jpg = io.BytesIO()
    Image.fromarray(tex).save(jpg, "JPEG", quality=quality)
    pos = np.ascontiguousarray(vertices, dtype=np.float32)
    blobs = [pos.tobytes(), np.ascontiguousarray(uv, dtype=np.float32).tobytes(),
             np.ascontiguousarray(tris, dtype=np.uint32).tobytes(), jpg.getvalue()]
    views, offset, binary = [], 0, b""
    for i, blob in enumerate(blobs):
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(blob)}
        if i < 2:
            view["target"] = 34962
        elif i == 2:
            view["target"] = 34963
        views.append(view)
        blob += b"\0" * (-len(blob) % 4)
        binary += blob
        offset += len(blob)
    doc = {
        "asset": {"version": "2.0", "generator": "ImageToSplat scene_splat.mesh"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0, "roughnessFactor": 1},
                       "extensions": {"KHR_materials_unlit": {}}}],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": 3, "mimeType": "image/jpeg"}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(pos), "type": "VEC3",
             "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": len(uv), "type": "VEC2"},
            {"bufferView": 2, "componentType": 5125, "count": int(np.size(tris)), "type": "SCALAR"},
        ],
    }
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(binary), 0x004E4942) + binary
    Path(out).write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body)

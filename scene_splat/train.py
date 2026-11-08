"""Train 3D Gaussian splats with gsplat from an undistorted COLMAP model and export .spz."""
import math
import time
from pathlib import Path

import numpy as np


def _ssim(x, y):
    import torch
    import torch.nn.functional as F

    k = torch.tensor([math.exp(-(i - 5) ** 2 / (2 * 1.5 ** 2)) for i in range(11)], device=x.device)
    k = (k / k.sum())[:, None] @ (k / k.sum())[None, :]
    w = k.expand(3, 1, 11, 11)
    mu_x, mu_y = F.conv2d(x, w, groups=3), F.conv2d(y, w, groups=3)
    sxx = F.conv2d(x * x, w, groups=3) - mu_x ** 2
    syy = F.conv2d(y * y, w, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, w, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))).mean()


def to_three_frame(means, quats_wxyz, shN):
    """Rotate 180 degrees about x: COLMAP (y down, z forward) -> three.js (y up)."""
    means = means * np.array([1.0, -1.0, -1.0], dtype=means.dtype)
    w, x, y, z = quats_wxyz.T
    quats = np.stack([-x, w, -z, y], 1)  # (0,1,0,0) * q
    shN = shN.copy()
    if shN.shape[1] >= 3:  # degree-1 basis is (y, z, x): y and z terms change sign
        shN[:, 0:2, :] *= -1
    if shN.shape[1] > 3:
        raise ValueError("Frame rotation is implemented for SH degree <= 1")
    return means, quats, shN


def train_splats(data: Path, out_dir: Path, steps: int = 20000, max_splats: int = 600_000, sh_degree: int = 1,
                 holdout: int = 8, device: str = "cuda", progress=lambda msg, frac=None: None) -> dict:
    import pycolmap
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from gsplat import rasterization
    from gsplat.strategy import MCMCStrategy

    from .spz import write_spz

    out_dir.mkdir(parents=True, exist_ok=True)
    rec = pycolmap.Reconstruction(data / "sparse")
    views = []
    for im in sorted(rec.images.values(), key=lambda i: i.name):
        cam = rec.cameras[im.camera_id]
        p = cam.params
        fx, fy, cx, cy = (p[0], p[1], p[2], p[3]) if len(p) >= 4 else (p[0], p[0], p[1], p[2])
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :4] = im.cam_from_world().matrix()
        rgb = torch.from_numpy(np.asarray(Image.open(data / "images" / im.name).convert("RGB"), dtype=np.float32) / 255.0)
        views.append(dict(name=im.name, K=K.to(device), w2c=torch.from_numpy(w2c).to(device), rgb=rgb, W=cam.width, H=cam.height))
    test_ids = set(range(0, len(views), holdout)) if holdout else set()
    train_ids = [i for i in range(len(views)) if i not in test_ids]

    pts = torch.tensor(np.array([p.xyz for p in rec.points3D.values()], dtype=np.float32), device=device)
    cols = torch.tensor(np.array([p.color for p in rec.points3D.values()], dtype=np.float32) / 255.0, device=device)
    with torch.no_grad():
        d = [torch.cdist(pts[s:s + 4096], pts).topk(4, largest=False).values[:, 1:].mean(1) for s in range(0, len(pts), 4096)]
        dist = torch.cat(d).clamp_min(1e-7)
    n = len(pts)
    C0 = 0.28209479177387814
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(pts),
        "scales": torch.nn.Parameter(torch.log(dist)[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]], device=device).repeat(n, 1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.5, device=device))),
        "sh0": torch.nn.Parameter(((cols - 0.5) / C0)[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros((n, (sh_degree + 1) ** 2 - 1, 3), device=device)),
    })
    centers = torch.stack([-(v["w2c"][:3, :3].T @ v["w2c"][:3, 3]) for v in views])
    scene_scale = float((centers - centers.mean(0)).norm(dim=1).max()) * 1.1
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([params[k]], lr=lr, eps=1e-15) for k, lr in lrs.items()}
    sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1.0 / steps))
    strategy = MCMCStrategy(cap_max=max_splats, refine_stop_iter=int(steps * 0.8))
    strategy.check_sanity(params, opts)
    state = strategy.initialize_state()

    def render(v, deg):
        img, _, info = rasterization(
            params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), torch.cat([params["sh0"], params["shN"]], 1),
            v["w2c"][None], v["K"][None], v["W"], v["H"], sh_degree=deg, packed=False)
        return img[0].clamp(0, 1), info

    t0 = time.time()
    rng = np.random.default_rng(0)
    for step in range(steps):
        v = views[train_ids[rng.integers(len(train_ids))]]
        img, info = render(v, min(step // 1000, sh_degree))
        gt = v["rgb"].to(device)
        loss = 0.8 * (img - gt).abs().mean() + 0.2 * (1 - _ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        loss = loss + 0.01 * torch.sigmoid(params["opacities"]).mean() + 0.01 * torch.exp(params["scales"]).mean()
        loss.backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        sched.step()
        strategy.step_post_backward(params, opts, state, step, info, lr=sched.get_last_lr()[0])
        if step % 500 == 0:
            progress(f"Training splats {step}/{steps}", step / steps)

    psnrs = []
    with torch.no_grad():
        for i in sorted(test_ids):
            img, _ = render(views[i], sh_degree)
            psnrs.append(float(-10 * torch.log10(((img - views[i]["rgb"].to(device)) ** 2).mean())))

    P = {k: v.detach().cpu().numpy() for k, v in params.items()}
    quats = F.normalize(params["quats"], dim=-1).detach().cpu().numpy()
    keep = 1 / (1 + np.exp(-P["opacities"])) >= 0.005
    means, quats, shN = to_three_frame(P["means"][keep], quats[keep], P["shN"][keep])
    spz_path = out_dir / "scene.spz"
    count = write_spz(spz_path, means, P["scales"][keep], quats, P["opacities"][keep], P["sh0"][keep, 0], shN, sh_degree)
    return {
        "spz": str(spz_path),
        "splats": int(count),
        "sizeBytes": spz_path.stat().st_size,
        "heldOutPsnr": round(float(np.mean(psnrs)), 2) if psnrs else None,
        "trainSeconds": round(time.time() - t0, 1),
        "steps": steps,
    }

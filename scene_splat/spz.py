"""Write Gaussian splats as .spz (Niantic's compressed format, v3, as read by Spark).

Inputs are numpy arrays in 3DGS/gsplat conventions:
  means (N,3), log_scales (N,3), quats_wxyz (N,4), opacity_logits (N,), sh0 (N,3), shN (N,K,3)
"""
import gzip

import numpy as np

SPZ_MAGIC = 0x5053474E
C0 = 0.28209479177387814
SH_VECS = {0: 0, 1: 3, 2: 8, 3: 15}


def _u8(x):
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def _quat_smallest_three(q_xyzw):
    q = q_xyzw / np.linalg.norm(q_xyzw, axis=1, keepdims=True)
    largest = np.abs(q).argmax(1)
    negate = q[np.arange(len(q)), largest] < 0
    comp = largest.astype(np.uint64)
    # pack the three other components in index order, like SpzWriter.setQuat
    others = np.array([[j for j in range(4) if j != k] for k in range(4)])[largest]  # (N,3)
    for c in range(3):
        v = q[np.arange(len(q)), others[:, c]]
        negbit = ((v < 0) ^ negate).astype(np.uint64)
        mag = np.floor(511 * (np.abs(v) / np.sqrt(0.5)) + 0.5).astype(np.uint64)
        comp = (comp << np.uint64(10)) | (negbit << np.uint64(9)) | mag
    comp = comp.astype(np.uint32)
    return np.stack([(comp >> s) & 255 for s in (0, 8, 16, 24)], 1).astype(np.uint8)


def write_spz(path, means, log_scales, quats_wxyz, opacity_logits, sh0, shN=None, sh_degree=0,
              fractional_bits=12, antialias=False):
    n = len(means)
    sh_degree = int(sh_degree)
    header = np.zeros(16, np.uint8)
    header[0:4] = np.frombuffer(np.uint32(SPZ_MAGIC).tobytes(), np.uint8)
    header[4:8] = np.frombuffer(np.uint32(3).tobytes(), np.uint8)
    header[8:12] = np.frombuffer(np.uint32(n).tobytes(), np.uint8)
    header[12], header[13], header[14] = sh_degree, fractional_bits, 1 if antialias else 0

    fixed = np.clip(np.round(means * (1 << fractional_bits)), -8388607, 8388607).astype(np.int32)
    pos = np.stack([(fixed >> s) & 255 for s in (0, 8, 16)], -1).astype(np.uint8).reshape(n, 9)
    alpha = _u8(255 / (1 + np.exp(-opacity_logits)))
    rgb = _u8((sh0 * 0.15 + 0.5) * 255)
    scale = _u8((log_scales + 10) * 16)
    rot = _quat_smallest_three(quats_wxyz[:, [1, 2, 3, 0]])
    parts = [header, pos.ravel(), alpha, rgb.ravel(), scale.ravel(), rot.ravel()]
    k = SH_VECS[sh_degree]
    if k:
        sh = shN[:, :k, :].reshape(n, k * 3)  # coefficient-major, rgb interleaved
        v = np.round(sh * 128) + 128
        bits = np.where(np.arange(k * 3) < 9, 5, 4)
        bucket = (1 << (8 - bits)).astype(np.float64)
        parts.append(_u8(np.floor((v + bucket / 2) / bucket) * bucket).ravel())
    with gzip.open(path, "wb", compresslevel=9) as f:
        f.write(np.concatenate(parts).tobytes())
    return n


def read_spz(path) -> dict:
    """Read a v2/v3 .spz back into 3DGS arrays: means, log_scales, quats_wxyz, opacities (0..1), sh0 (DC), sh_degree."""
    with gzip.open(path, "rb") as f:
        raw = np.frombuffer(f.read(), np.uint8)
    magic, version, n = (int(np.frombuffer(raw[i:i + 4].tobytes(), np.uint32)[0]) for i in (0, 4, 8))
    if magic != SPZ_MAGIC or version not in (2, 3):
        raise ValueError(f"Not a supported .spz file (magic {magic:#x}, version {version})")
    sh_degree, frac = int(raw[12]), int(raw[13])
    o = 16
    pos = raw[o:o + 9 * n].reshape(n, 3, 3).astype(np.int32)
    o += 9 * n
    fixed = pos[..., 0] | (pos[..., 1] << 8) | (pos[..., 2] << 16)
    fixed = np.where(fixed & 0x800000, fixed - (1 << 24), fixed)
    means = fixed.astype(np.float32) / (1 << frac)
    alpha = raw[o:o + n].astype(np.float32) / 255
    o += n
    rgb = raw[o:o + 3 * n].reshape(n, 3).astype(np.float32)
    o += 3 * n
    log_scales = raw[o:o + 3 * n].reshape(n, 3).astype(np.float32) / 16 - 10
    o += 3 * n
    if version == 3:
        b = raw[o:o + 4 * n].reshape(n, 4).astype(np.uint32)
        o += 4 * n
        comp = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24)
        largest = (comp >> 30).astype(np.int64)
        q = np.zeros((n, 4), np.float32)
        others = np.array([[j for j in range(4) if j != k] for k in range(4)])[largest]
        for c in range(3):
            chunk = (comp >> (10 * (2 - c))) & 1023
            v = (chunk & 511).astype(np.float32) / 511 * np.sqrt(0.5)
            v = np.where(chunk & 512, -v, v)
            q[np.arange(n), others[:, c]] = v
        q[np.arange(n), largest] = np.sqrt(np.clip(1 - (q ** 2).sum(1), 0, 1))
        quats_xyzw = q
    else:
        xyz = raw[o:o + 3 * n].reshape(n, 3).astype(np.float32) / 127.5 - 1
        o += 3 * n
        quats_xyzw = np.c_[xyz, np.sqrt(np.clip(1 - (xyz ** 2).sum(1), 0, 1))]
    sh0 = (rgb / 255 - 0.5) / 0.15
    return {"means": means, "log_scales": log_scales, "quats_wxyz": quats_xyzw[:, [3, 0, 1, 2]], "opacities": alpha,
            "sh0": sh0, "sh_degree": sh_degree}

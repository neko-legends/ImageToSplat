"""Structure-from-motion for one or more videos of the same scene (pycolmap).

Frames are named <prefix>_NNNN.jpg, one prefix per video. Pairs are matched within each video (sliding window)
and across videos (subsampled all-to-all), so separate clips of one room land in one model.
"""
import itertools
from pathlib import Path


def build_pairs(names: list[str], window: int = 15) -> list[tuple[str, str]]:
    seqs: dict[str, list[str]] = {}
    for n in sorted(names):
        seqs.setdefault(n.rsplit("_", 1)[0], []).append(n)
    pairs: set[tuple[str, str]] = set()
    for s in seqs.values():
        for i, a in enumerate(s):
            for b in s[i + 1:i + 1 + window]:
                pairs.add((a, b))
    keys = sorted(seqs)
    for x, y in itertools.combinations(keys, 2):
        for a, b in itertools.product(seqs[x][::3], seqs[y][::3]):
            pairs.add((a, b))
    return sorted(pairs)


def reconstruct(work: Path, progress=lambda msg, frac=None: None):
    import pycolmap

    images = work / "images"
    db = work / "database.db"
    sparse = work / "sparse"
    db.unlink(missing_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)

    progress("Finding features", None)
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.sift.max_num_features = 8192
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    pycolmap.extract_features(db, images, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=extraction)

    names = sorted(p.name for p in images.glob("*.jpg"))
    pairs = build_pairs(names)
    pair_file = work / "pairs.txt"
    pair_file.write_text("\n".join(f"{a} {b}" for a, b in pairs))
    progress(f"Matching {len(pairs)} frame pairs", None)
    pycolmap.match_image_pairs(db, pairing_options=pycolmap.ImportedPairingOptions(match_list_path=pair_file))

    progress("Solving camera positions", None)
    models = pycolmap.incremental_mapping(db, images, sparse)
    if not models:
        raise RuntimeError("Could not work out camera positions. The video needs more texture and overlap between frames.")
    best = max(models.values(), key=lambda m: m.num_reg_images())
    stats = {
        "frames": len(names),
        "registered": best.num_reg_images(),
        "points": best.num_points3D(),
        "reprojectionError": round(best.compute_mean_reprojection_error(), 3),
        "models": len(models),
    }
    progress(f"Placed {stats['registered']}/{stats['frames']} frames", None)
    undistorted = work / "undistorted"
    best_dir = sparse / "best"
    best_dir.mkdir(exist_ok=True)
    best.write(best_dir)
    pycolmap.undistort_images(undistorted, best_dir, images)
    return undistorted, stats

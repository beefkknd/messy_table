"""Label-free, foreground-constrained SIFT and affine correspondence evidence."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from common import sha256

LONG_SIDE_LIMIT = 384
MAX_KEYPOINTS = 256
EROSION_RADIUS = 2
RATIO = 0.80
NORMALIZED_LONG_SIDE = 224
RANSAC_THRESHOLD = 4.0
RANSAC_MAX_ITERS = 2000
RANSAC_CONFIDENCE = 0.99
RANSAC_REFINE_ITERS = 10
RANSAC_SEED = 0

SETTINGS = {
    "crop": "Original native RGB detector box, converted to grayscale with Pillow RGB→L",
    "max_working_long_side": LONG_SIDE_LIMIT,
    "resize": "Only downsample; preserve aspect ratio; INTER_AREA for image and INTER_NEAREST for mask",
    "sam_mask": "Use only raw usable SAM mask; no RGB fallback for this cue",
    "mask_erosion_radius": EROSION_RADIUS,
    "mask_erosion": "Ellipse 5x5, one iteration, BORDER_CONSTANT value0",
    "sift": {"nfeatures": MAX_KEYPOINTS, "contrastThreshold": .02, "edgeThreshold": 10, "sigma": 1.6},
    "descriptor": "SIFT float32 L2",
    "reciprocal_ratio_exclusive": RATIO,
    "ransac_seed_per_edge": RANSAC_SEED,
    "affine": {
        "method": "cv2.estimateAffine2D RANSAC",
        "normalized_long_side": NORMALIZED_LONG_SIDE,
        "threshold_px": RANSAC_THRESHOLD, "max_iters": RANSAC_MAX_ITERS,
        "confidence": RANSAC_CONFIDENCE, "refine_iters": RANSAC_REFINE_ITERS,
    },
    "coordinate_convention": "Native crop pixel centers via inverse resize, then scaled by 224/native long side",
    "geometry_metrics": "Inlier XY extent and convex hull area divided by each native crop dimension/area",
    "opencv_threads": 1,
}


@dataclass
class LocalCrop:
    status: str
    metadata: dict
    native_xy: np.ndarray | None = None
    normalized_xy: np.ndarray | None = None
    descriptors: np.ndarray | None = None


def runtime():
    cv2.setNumThreads(1)
    return (
        cv2.SIFT_create(**SETTINGS["sift"]),
        cv2.BFMatcher(cv2.NORM_L2, crossCheck=False),
    )


def sam_foreground(record, cache):
    path = record["mask_path"]
    if record["status"] != "usable":
        return None
    if not path or record["fallback_original_rgb"]:
        raise ValueError("Usable SAM mask has no usable raw image")
    filename = Path(cache) / path
    if sha256(filename) != record["mask_sha256"]:
        raise ValueError(f"Raw foreground mask SHA mismatch: {filename}")
    with Image.open(filename) as image:
        if image.mode != "L":
            raise ValueError("Expected native single-channel SAM mask")
        pixels = np.array(image)
    if not ((pixels == 0) | (pixels == 255)).all():
        raise ValueError("Expected binary SAM mask")
    mask = pixels == 255
    if float(mask.mean()) != record["raw_mask_fraction"]:
        raise ValueError("Native SAM mask and recorded fraction differ")
    return mask


def crop_features(crop, sam_record, cache, sift):
    width, height = crop.size
    meta = {
        "native_width": width, "native_height": height,
        "sam_status": sam_record["status"], "sam_fallback_original_rgb": sam_record["fallback_original_rgb"],
        "sam_raw_mask_fraction": sam_record["raw_mask_fraction"], "sam_mask_sha256": sam_record["mask_sha256"],
        "sam_mask_path": sam_record["mask_path"], "keypoints_detected": 0, "keypoints_kept": 0,
    }
    mask = sam_foreground(sam_record, cache)
    if mask is None:
        return LocalCrop("sam_mask_unusable", meta)
    if mask.shape != (height, width):
        raise ValueError("Raw SAM mask and native detector crop shape differ")
    gray = np.array(crop.convert("L"))
    scale = min(1.0, LONG_SIDE_LIMIT / max(width, height))
    ww, hh = max(1, round(width * scale)), max(1, round(height * scale))
    if (ww, hh) != (width, height):
        gray = cv2.resize(gray, (ww, hh), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), (ww, hh), interpolation=cv2.INTER_NEAREST).astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * EROSION_RADIUS + 1,) * 2)
    interior = cv2.erode(
        mask.astype(np.uint8) * 255, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0
    )
    meta.update({
        "working_width": ww, "working_height": hh, "working_mask_fraction": float(mask.mean()),
        "eroded_mask_fraction": float(np.count_nonzero(interior) / interior.size),
    })
    if not interior.any():
        return LocalCrop("eroded_mask_empty", meta)
    try:
        keypoints, desc = sift.detectAndCompute(gray, interior)
    except cv2.error as exc:
        meta["error"] = str(exc)
        return LocalCrop("sift_error", meta)
    meta["keypoints_detected"] = len(keypoints)
    if not keypoints or desc is None:
        return LocalCrop("no_keypoints", meta)
    if desc.dtype != np.float32 or desc.shape != (len(keypoints), 128) or not np.isfinite(desc).all():
        raise ValueError("Unexpected SIFT descriptor values")
    # OpenCV can retain more than nfeatures at equal response. Cap deterministically
    # and verify explicitly that rounded centers lie inside the eroded mask.
    picked = []
    for i, point in enumerate(keypoints):
        x, y = point.pt
        ix, iy = min(ww - 1, max(0, int(round(x)))), min(hh - 1, max(0, int(round(y))))
        if interior[iy, ix]:
            picked.append(i)
    if len(picked) > MAX_KEYPOINTS:
        picked = sorted(picked, key=lambda i: (-keypoints[i].response, i))[:MAX_KEYPOINTS]
    meta["keypoints_kept"] = len(picked)
    if len(picked) < 2:
        return LocalCrop("too_few_keypoints", meta)
    work_xy = np.asarray([keypoints[i].pt for i in picked], dtype=np.float64)
    native_xy = ((work_xy + 0.5) * [width / ww, height / hh] - 0.5).astype(np.float32)
    norm_xy = (native_xy * (NORMALIZED_LONG_SIDE / max(width, height))).astype(np.float32)
    return LocalCrop("available", meta, native_xy, norm_xy, np.ascontiguousarray(desc[picked]))


def ratio_matches(matcher, source, target):
    valid = {}
    for nearest in matcher.knnMatch(source, target, k=2):
        if len(nearest) != 2:
            continue
        best, second = nearest
        if best.distance < RATIO * second.distance:
            valid[best.queryIdx] = (best.trainIdx, float(best.distance), float(best.distance / second.distance))
    return valid


def support_geometry(points, width, height):
    if len(points) == 0:
        return {"spread_x": 0.0, "spread_y": 0.0, "hull_fraction": 0.0}
    extent = np.ptp(points, axis=0)
    area = float(cv2.contourArea(cv2.convexHull(points))) if len(points) >= 3 else 0.0
    return {
        "spread_x": float(extent[0] / width), "spread_y": float(extent[1] / height),
        "hull_fraction": area / (width * height),
    }


def geometric_fit(a_xy, b_xy):
    if len(a_xy) < 3:
        return None, np.zeros(len(a_xy), dtype=bool), "insufficient_reciprocal_matches"
    cv2.setRNGSeed(RANSAC_SEED)
    affine, flags = cv2.estimateAffine2D(
        a_xy, b_xy, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD,
        maxIters=RANSAC_MAX_ITERS, confidence=RANSAC_CONFIDENCE, refineIters=RANSAC_REFINE_ITERS,
    )
    if affine is None or flags is None:
        return None, np.zeros(len(a_xy), dtype=bool), "affine_unavailable"
    if affine.shape != (2, 3) or flags.size != len(a_xy):
        raise ValueError("Malformed affine fit output dimensions")
    if not np.isfinite(affine).all():
        # Repeated orientation keypoints can form a degenerate three-point
        # sample. OpenCV sometimes returns NaN/Inf with inlier flags instead
        # of None. Successful comparison with no valid fit is measured zero.
        return None, np.zeros(len(a_xy), dtype=bool), "affine_nonfinite_measured_zero"
    return affine, flags.ravel().astype(bool), "affine_measured"


def measure(a, b, matcher):
    result = {
        "status": "local_unavailable", "local_available": False, "crop_status": {"A": a.status, "B": b.status},
        "reciprocal_matches": 0, "inliers": 0, "inlier_fraction": None,
        "geometry": {"A": None, "B": None}, "affine_224_A_to_B": None, "matches": [],
    }
    if a.status != "available" or b.status != "available":
        return result
    try:
        ab = ratio_matches(matcher, a.descriptors, b.descriptors)
        ba = ratio_matches(matcher, b.descriptors, a.descriptors)
    except cv2.error as exc:
        result["error"] = str(exc)
        return result
    reciprocal = [
        (i, j, distance, ratio, ba[j][1], ba[j][2])
        for i, (j, distance, ratio) in sorted(ab.items())
        if j in ba and ba[j][0] == i
    ]
    a_ids = [row[0] for row in reciprocal]
    b_ids = [row[1] for row in reciprocal]
    try:
        affine, inlier, status = geometric_fit(a.normalized_xy[a_ids], b.normalized_xy[b_ids])
    except cv2.error as exc:
        result.update({"error": str(exc), "reciprocal_matches": len(reciprocal)})
        return result
    result.update({
        "status": status, "local_available": True,
        "ratio_matches_A_to_B": len(ab), "ratio_matches_B_to_A": len(ba),
        "reciprocal_matches": len(reciprocal), "inliers": int(inlier.sum()),
        "inlier_fraction": float(inlier.mean()) if len(inlier) else None,
        "geometry": {
            "A": support_geometry(a.native_xy[a_ids][inlier], a.metadata["native_width"], a.metadata["native_height"]),
            "B": support_geometry(b.native_xy[b_ids][inlier], b.metadata["native_width"], b.metadata["native_height"]),
        },
        "affine_224_A_to_B": affine.tolist() if affine is not None else None,
        "matches": [
            {
                "A_keypoint": i, "B_keypoint": j, "descriptor_distance_A_to_B": d_ab,
                "descriptor_ratio_A_to_B": r_ab, "descriptor_distance_B_to_A": d_ba, "descriptor_ratio_B_to_A": r_ba,
                "A_native_xy": a.native_xy[i].tolist(), "B_native_xy": b.native_xy[j].tolist(),
                "A_normalized_224_xy": a.normalized_xy[i].tolist(), "B_normalized_224_xy": b.normalized_xy[j].tolist(),
                "inlier": bool(flag),
            }
            for (i, j, d_ab, r_ab, d_ba, r_ba), flag in zip(reciprocal, inlier, strict=True)
        ],
    })
    return result

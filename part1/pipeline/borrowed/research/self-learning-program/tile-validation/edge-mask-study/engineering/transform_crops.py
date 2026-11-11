"""Native-pixel transforms with identical detector-box crop coordinates."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from common import sha256
RGB_FILL = 128
GRABCUT_ITERATIONS = 3
CV_SEED = 0
MASK_MIN = 0.02
MASK_MAX = 0.98
VARIANTS = ("rgb", "grayscale_rgb", "canny_rgb", "grabcut_masked_rgb", "sam_masked_rgb")
SETTINGS = {
    "grayscale_rgb": {"conversion": "Pillow RGB to L, then repeat L to 3 channels", "blur": False},
    "canny_rgb": {
        "conversion": "Pillow RGB to L, binary white edges on black replicated in RGB",
        "gaussian_kernel": [5, 5], "gaussian_sigma": 1.1, "low": 50, "high": 150,
        "gaussian_border": "BORDER_DEFAULT", "aperture_size": 3, "L2gradient": True, "empty_fallback": False,
    },
    "grabcut": {
        "padding_fraction_per_side": 0.20, "padding_round": "ceil then clip to native frame",
        "iterations": GRABCUT_ITERATIONS, "opencv_rng_seed_per_crop": CV_SEED, "opencv_threads": 1,
        "initialization": "GC_INIT_WITH_RECT; FGD and PR_FGD are foreground",
    },
    "sam": {
        "resolution": 1024, "confidence_floor_exclusive": 0.0, "prompts": "One original-detector XYXY rectangle per box",
        "points": False, "masks_per_prompt": 1, "native_mask_geometry": True,
    },
    "masking": {
        "background_rgb": [RGB_FILL] * 3,
        "foreground_fraction_lower_exclusive": MASK_MIN,
        "foreground_fraction_upper_exclusive": MASK_MAX,
        "failure_fallback": "Exact original RGB crop; retain status and raw mask when available",
    },
}


@dataclass
class MaskResult:
    mask: np.ndarray | None
    status: str
    confidence: float | None = None
    error: str | None = None


def initialize_opencv():
    cv2.setNumThreads(1)


def native_crop(rgb, box):
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= rgb.shape[1] and 0 <= y0 < y1 <= rgb.shape[0]):
        raise ValueError(f"Invalid original-pixel detector box {box}")
    return np.ascontiguousarray(rgb[y0:y1, x0:x1])


def replicate(channel):
    return cv2.cvtColor(channel, cv2.COLOR_GRAY2RGB)


def grayscale_and_edges(crop):
    luminance = np.asarray(Image.fromarray(crop).convert("L"))
    blurred = cv2.GaussianBlur(luminance, (5, 5), 1.1, borderType=cv2.BORDER_DEFAULT)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3, L2gradient=True)
    return replicate(luminance), replicate(edges), float(np.count_nonzero(edges) / edges.size)


def padded_region(box, width, height):
    x0, y0, x1, y1 = box
    px = (x1 - x0 + 4) // 5
    py = (y1 - y0 + 4) // 5
    return max(x0 - px, 0), max(y0 - py, 0), min(x1 + px, width), min(y1 + py, height)


def grabcut(rgb, box):
    x0, y0, x1, y1 = box
    l, t, r, b = padded_region(box, rgb.shape[1], rgb.shape[0])
    image = cv2.cvtColor(rgb[t:b, l:r], cv2.COLOR_RGB2BGR)
    canvas = np.zeros(image.shape[:2], dtype=np.uint8)
    foreground = np.zeros((1, 65), dtype=np.float64)
    background = np.zeros((1, 65), dtype=np.float64)
    rect = (x0 - l, y0 - t, x1 - x0, y1 - y0)
    cv2.setRNGSeed(CV_SEED)
    try:
        cv2.grabCut(image, canvas, rect, background, foreground, GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_RECT)
    except cv2.error as exc:
        return MaskResult(None, "grabcut_error", error=str(exc))
    crop = canvas[y0 - t:y1 - t, x0 - l:x1 - l]
    return MaskResult((crop == cv2.GC_FGD) | (crop == cv2.GC_PR_FGD), "candidate")


def grabcut_view(path, expected_sha256, width, height, boxes):
    """Isolated per-view job; each worker keeps its OpenCV thread count at one."""
    import time

    started = time.monotonic()
    initialize_opencv()
    path = Path(path)
    if sha256(path) != expected_sha256:
        raise ValueError(f"Source image SHA changed: {path}")
    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB"))
    if rgb.shape[:2] != (height, width):
        raise ValueError(f"Native image shape differs: {path}")
    masks = [grabcut(rgb, box) for box in boxes]
    return masks, time.monotonic() - started


def masked_rgb(crop, result):
    mask = result.mask
    fraction = None
    status = result.status
    if mask is not None:
        if mask.shape != crop.shape[:2] or mask.dtype != bool:
            raise ValueError("Mask changed the native crop geometry or has wrong dtype")
        fraction = float(mask.mean())
        status = (
            "empty" if fraction == 0
            else "full" if fraction == 1
            else "near_empty" if fraction <= MASK_MIN
            else "near_full" if fraction >= MASK_MAX
            else "usable"
        )
    effective = (
        np.where(mask[:, :, None], crop, RGB_FILL).astype(np.uint8)
        if status == "usable"
        else crop
    )
    return effective, {
        "status": status, "raw_mask_fraction": fraction,
        "fallback_original_rgb": status != "usable", "mask_confidence": result.confidence,
        "error": result.error,
    }


def sam2_masks(model, rgb, boxes, device="mps"):
    if not boxes:
        return []
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    try:
        result = model.predict(
            bgr, bboxes=boxes, imgsz=1024, conf=0.0, device=device, verbose=False, quantize=32
        )[0]
    except RuntimeError as exc:
        return [MaskResult(None, "sam_error", error=str(exc)) for _ in boxes]
    rows = [MaskResult(None, "missing_prompt_mask") for _ in boxes]
    if result.masks is None:
        return rows
    masks = result.masks.data
    if tuple(masks.shape[1:]) != rgb.shape[:2] or masks.dtype.is_floating_point:
        raise ValueError("SAM2 masks are not binary at the source image geometry")
    prompt_ids = result.boxes.cls.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()
    if len(set(prompt_ids)) != len(prompt_ids) or any(
        index != int(index) or not (0 <= int(index) < len(boxes)) for index in prompt_ids
    ):
        raise ValueError("SAM2 output did not preserve unique in-range detector prompt IDs")
    for prompt, mask, confidence in zip(prompt_ids, masks, confidences, strict=True):
        index = int(prompt)
        x0, y0, x1, y1 = boxes[index]
        crop = mask[y0:y1, x0:x1].cpu().numpy().astype(bool, copy=True)
        rows[index] = MaskResult(crop, "candidate", float(confidence))
    return rows

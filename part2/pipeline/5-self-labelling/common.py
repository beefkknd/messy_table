"""Local paths, integrity, and runtime isolation for the detector pilot."""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PROGRAM = HERE.parent
PREVIOUS = HERE.parents[1] / "engineering"
A_CHECKPOINT = PREVIOUS / "round2/train/A/checkpoint.pt"
WEIGHTS = ROOT / "everyday-identity/assets/models/yolo26m.pt"
WEIGHTS_SHA = "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7"
ULTRALYTICS = "8.4.202"


def local_path(path):
    return (ROOT / path).resolve(strict=True)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_text(path):
    return Path(path).read_text(encoding="utf-8")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def initialize_runtime():
    os.environ["YOLO_CONFIG_DIR"] = str(HERE / ".runtime")
    os.environ["MPLCONFIGDIR"] = str(HERE / ".runtime/matplotlib")
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    (HERE / ".runtime/matplotlib").mkdir(parents=True, exist_ok=True)


def pin_ultralytics():
    import ultralytics

    if ultralytics.__version__ != ULTRALYTICS:
        raise RuntimeError(f"Expected Ultralytics {ULTRALYTICS}, got {ultralytics.__version__}")
    ultralytics.settings.update({"sync": False})

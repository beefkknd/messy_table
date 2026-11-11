"""Paths, native-source loading, and atomic artifacts for official reproduction."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import types

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVAL = ROOT / 'evaluation'
DATA = ROOT / 'data/messytable-full/MessyTable'
LABELS = DATA / 'labels/test.json'
VIT = EVAL / 'messytable-vit-p3de-100'
ASNET = EVAL / 'messytable-asnet-ladder'
VIT_REPO = VIT / 'assets/ViT-P3DE'
ASNET_REPO = ASNET / 'assets/MessyTable'


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_native_dataset(repository, config):
    # Compile only these unchanged upstream definitions. Their other module
    # imports require the Linux metric extension, which runs in Docker.
    namespace = {
        'np': np, 'cv2': cv2, 'torch': torch, 'Dataset': Dataset,
        'transforms': transforms, 'json': json, 'os': __import__('os'),
    }
    selections = [('utils.py', {'crop_feat', 'scalar_clip'}),
                  ('data.py', {'MessyTableDatasetFeatures'})]
    for name, wanted in selections:
        path = repository / 'src' / name
        parsed = ast.parse(path.read_text())
        nodes = [node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and node.name in wanted]
        assert {node.name for node in nodes} == wanted
        source = ast.Module(body=nodes, type_ignores=[])
        exec(compile(source, str(path), 'exec'), namespace)
    return namespace['MessyTableDatasetFeatures'](config, str(LABELS))


def load_module(name, path):
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

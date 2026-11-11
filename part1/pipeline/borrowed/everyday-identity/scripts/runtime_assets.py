"""Resolve hash-verified persistent local assets for the common-object branch."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def manifest():
    path=ROOT/'assets/manifest.json'
    if not path.exists():raise FileNotFoundError('Missing preserved assets/manifest.json; see README replay instructions.')
    return json.loads(path.read_text())


@lru_cache(maxsize=None)
def checked(relative):
    path=ROOT/relative
    record=next((r for r in manifest()['assets'] if r['file']==relative),None)
    if record is None or not path.is_file():raise FileNotFoundError(f'Preserved inference asset missing: {path}')
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=record['sha256']:raise ValueError(f'Inference asset hash mismatch: {path}')
    return path


def model_file(name):return checked('assets/models/'+name)


def dinov2_source():
    for name in ['config.json','preprocessor_config.json','model.safetensors']:checked('assets/dinov2-small/'+name)
    return str(ROOT/'assets/dinov2-small')


def dinov2_weights():return checked('assets/dinov2-small/model.safetensors')
def dinov2_commit():return manifest()['dinov2']['revision']


def foundyou_repository():
    root=ROOT/'vendor/FoundYou'
    for name in ['foundyou.safetensors','sam2_hiera_small.pt']:checked('vendor/FoundYou/pretrain/'+name)
    for name,digest in manifest()['foundyou']['source_hashes'].items():
        path=root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'FoundYou source hash mismatch: {path}')
    return root

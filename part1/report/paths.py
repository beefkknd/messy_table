"""Where the Part 1 report build reads its inputs and writes the static site.

Small frozen results live in this repo under part1/results/. Large inputs (the MessyTable label
file, six photos, three score arrays and the ground-truth vector) stay outside git. They are found
under the workspace named by MESSYTABLE_WORKSPACE, and every one is checked against the sha256
recorded in external-inputs.json before the build uses it.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

REPORT = Path(__file__).resolve().parent
PART1 = REPORT.parent
RESULTS = PART1 / 'results'
SAMPLE = RESULTS / 'sample100'
STEP2 = SAMPLE / 'step2'
PART_ONE_RESULTS = RESULTS / 'part-one'
SITE = PART1 / 'site'
ASSETS = SITE / 'assets'
EXTERNAL_MANIFEST = REPORT / 'external-inputs.json'
WORKSPACE_ENV = 'MESSYTABLE_WORKSPACE'
LABELS_REL = 'data/messytable-full/MessyTable/labels/test.json'
IMAGES_REL = 'data/messytable-full/MessyTable/images'
SCORES_REL = 'evaluation/messytable-official-reproduction/sample100/step2/scores'
GT_REL = 'evaluation/messytable-official-reproduction/sample100/vit/gt_inst.npy'


@dataclass(frozen=True)
class Sources:
    """Every input the build reads, resolved to concrete paths."""
    workspace: Path
    labels: Path
    images: Path
    scores: Path
    gt_inst: Path
    score_freeze: Path
    pair_index: Path
    features: Path
    dense: Path
    views: Path
    native_links: Path


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def find_workspace() -> Path:
    """MESSYTABLE_WORKSPACE if set, else the folder that contains this repo."""
    configured = os.environ.get(WORKSPACE_ENV)
    candidate = Path(configured).expanduser() if configured else PART1.parents[1]
    if not (candidate / LABELS_REL).is_file():
        raise SystemExit(
            f'MessyTable inputs not found under {candidate}.\n'
            f'Set {WORKSPACE_ENV} to the folder that contains {LABELS_REL} and {SCORES_REL}. '
            'See part1/README.md for how to obtain them.')
    return candidate


def verify_external(workspace: Path) -> None:
    """Fail fast if any large input is missing or differs from the recorded file."""
    problems = []
    for entry in json.loads(EXTERNAL_MANIFEST.read_text()):
        path = workspace / entry['path']
        if not path.is_file():
            problems.append(f'missing: {entry["path"]}')
        elif sha256(path) != entry['sha256']:
            problems.append(f'sha256 mismatch: {entry["path"]}')
    if problems:
        raise SystemExit('External inputs do not match external-inputs.json:\n  ' + '\n  '.join(problems))


def sources(verify: bool = True) -> Sources:
    workspace = find_workspace()
    if verify:
        verify_external(workspace)
    return Sources(
        workspace=workspace, labels=workspace / LABELS_REL, images=workspace / IMAGES_REL,
        scores=workspace / SCORES_REL, gt_inst=workspace / GT_REL,
        score_freeze=STEP2 / 'scores/freeze.json', pair_index=SAMPLE / 'vit/pair-index.json',
        features=SAMPLE / 'vit/features', dense=STEP2 / 'dense', views=STEP2 / 'views',
        native_links=PART_ONE_RESULTS / 'flow-native-links.json')

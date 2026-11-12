"""Copy Part Two's pipeline scripts byte-for-byte from the research workspace into part2/pipeline/<stage>/.

Writes pipeline/MANIFEST.json: for every file its repo path, original workspace path, sha256, size, stage and
any note; plus the Part Three scripts deliberately left out. The copies are an archive for provenance; they import
each other through the original workspace layout and run from there, not from this repo.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pipeline_files as pf

PART2 = Path(__file__).resolve().parent
OUT = PART2 / 'pipeline'


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def workspace() -> Path:
    configured = os.environ.get('MESSYTABLE_WORKSPACE')
    root = Path(configured).expanduser() if configured else PART2.parents[1]
    if not (root / 'evaluation/messytable-part-two').is_dir():
        raise SystemExit(f'Research workspace not found under {root}. Set MESSYTABLE_WORKSPACE.')
    return root


def main() -> None:
    root = workspace()
    if OUT.exists():
        shutil.rmtree(OUT)
    entries, seen = [], set()
    for stage, files in pf.STAGES.items():
        for original in files:
            source = root / original
            target = OUT / stage / Path(original).name
            if target in seen:
                raise SystemExit(f'Name clash in {stage}: {Path(original).name}')
            seen.add(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            digest = sha256(target)
            if digest != sha256(source):
                raise SystemExit(f'Copy mismatch: {original}')
            entries.append({'repo_path': str(target.relative_to(PART2.parent)), 'original_path': original,
                            'sha256': digest, 'bytes': target.stat().st_size, 'stage': stage,
                            **({'note': pf.NOTES[original]} if original in pf.NOTES else {})})
    (OUT / 'MANIFEST.json').write_text(json.dumps({
        'rule': 'Every first-party script/config whose output Part Two uses; identity-matching (Part Three) code excluded.',
        'files': entries,
        'excluded_part_three': pf.EXCLUDED_PART_THREE,
        'not_included': 'Model weights, MessyTable images/labels, prediction folders and virtual environments.',
    }, indent=2) + '\n')
    print('archived', len(entries), 'files into', OUT)


if __name__ == '__main__':
    main()

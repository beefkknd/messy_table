#!/usr/bin/env python3
"""Release Development100 scoring truth after a complete cohort-matched run freeze."""

import argparse
from pathlib import Path

from prepare import LABELS, OUT, ROOT, annotation_stats, digest, load, save, truth_rows, verify


def validate_run(path, cohort_path, cohort):
    freeze_path = path / 'freeze.json'
    freeze = load(freeze_path)
    required = {'inference-plan.json', 'predictions.json'}
    if set(freeze) != required:
        raise ValueError(f'Incomplete prediction freeze in {path}')
    for name in required:
        if digest(path / name) != freeze[name]:
            raise ValueError(f'Prediction artifact changed after freeze: {path / name}')
    if load(path / 'inference-plan.json')['manifest_sha256'] != digest(cohort_path):
        raise ValueError(f'Predictions use a different cohort: {path}')
    rows = load(path / 'predictions.json')['scenes']
    by_id = {row['id']: row for row in rows}
    if len(rows) != len(cohort) or by_id.keys() != cohort.keys():
        raise ValueError(f'Incomplete or duplicate prediction scene IDs: {path}')
    for case_id, item in cohort.items():
        expected = {view['id']: view['sha256'] for view in item['views']}
        prediction = by_id[case_id]
        actual = {key: value['sha256'] for key, value in prediction['views'].items()}
        if prediction['setup'] != item['setup'] or expected != actual:
            raise ValueError(f'Prediction source views differ in {case_id}: {path}')
    return {'run': path.relative_to(ROOT).as_posix(), 'freeze_sha256': digest(freeze_path), **freeze}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True,
                        help='Frozen prediction directories for the matched pretrained control and all compared arms')
    args = parser.parse_args()
    paths = [path.resolve() for path in args.runs]
    if len(paths) < 5 or len(paths) != len(set(paths)):
        parser.error('Provide at least five distinct frozen runs: original checkpoint and all four pilot arms.')
    output = OUT / 'scoring/development100.json'
    if output.exists():
        parser.error('Development100 is already available to the scorer. Preserve the first reveal.')
    verify(OUT)
    cohort_path = OUT / 'cohorts/development100.json'
    rows = load(cohort_path)['scenes']
    cohort = {row['id']: row for row in rows}
    validated = [validate_run(path, cohort_path, cohort) for path in paths]
    source = LABELS['val']
    source_digest = digest(source)
    if source_digest != load(OUT / 'audit.json')['source_sha256'][source.relative_to(ROOT).as_posix()]:
        raise ValueError('Native validation annotations changed since cohort freeze')
    truth = truth_rows(rows)
    save(output, {'version': 1, 'cohort': 'development100', 'cohort_sha256': digest(cohort_path),
                  'source_annotation_sha256': source_digest, 'first_reveal_runs': validated,
                  'statistics': annotation_stats(rows, truth), 'scenes': truth})
    save(OUT / 'development100-reveal-freeze.json', {str(output.relative_to(ROOT)): digest(output),
         str(Path(__file__).resolve().relative_to(ROOT)): digest(Path(__file__))})


if __name__ == '__main__':
    main()

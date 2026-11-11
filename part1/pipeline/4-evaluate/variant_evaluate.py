"""Run the original metric runtime for every frozen arm, then select the top3."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import time

from variant_common import OUT, ROOT, SAMPLE, freeze, read, save, sha, verify


def score(method):
    destination = OUT / 'metrics' / (method + '.json')
    if destination.exists():
        return method
    command = [
        'docker', 'run', '--rm', '--platform', 'linux/amd64',
        '-v', str(ROOT) + ':/workspace', 'messytable-published-metrics:20261111', 'python',
        '/workspace/evaluation/messytable-official-reproduction/runtime/score_variant_native.py',
        '--method', method,
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    (OUT / 'metrics' / (method + '.log')).write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(method + ': ' + completed.stderr[-1500:])
    assert destination.exists()
    return method


def main():
    while not (OUT / 'scores/freeze.json').exists():
        time.sleep(2)
    verify(OUT / 'scores')
    plan = read(OUT / 'scores/plan.json')
    (OUT / 'metrics').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(score, method): method for method in plan['methods']}
        for count, future in enumerate(as_completed(pending), 1):
            method = future.result()
            row = read(OUT / 'metrics' / (method + '.json'))
            print(f"{count}/{len(pending)} {method}: AP {100*row['AP']:.2f} FPR95 {100*row['FPR95']:.2f} IPAA100 {100*row['IPAA100']:.2f}", flush=True)
    results = {method: read(OUT / 'metrics' / (method + '.json')) for method in plan['methods']}
    for name, model in (('vit_reference', 'vit'), ('asnet_reference', 'asnet')):
        old = read(SAMPLE / model / 'native-metrics.json')
        actual = results[name]
        assert actual['AP'] == old['AP'] and actual['FPR95'] == old['FPR95']
        assert actual['IPAA_counts'] == old['IPAA_counts'], name
    ordered = sorted(results, key=lambda name: (-results[name]['AP'], results[name]['FPR95'], -results[name]['IPAA100'], name))
    own = [name for name in ordered if not name.endswith('_reference')]
    summary = {
        'training': False, 'selection_sha256': sha(SAMPLE / 'selection.json'),
        'scenes': 100, 'setups': 100, 'images': 900, 'image_pairs': 3600,
        'candidate_object_pairs': plan['candidates'], 'positive_object_pairs': 78335,
        'rank_by': 'AP descending,then FPR95 ascending,then IPAA100 descending.',
        'top_three_own_variants': own[:3], 'ranking_including_references': ordered,
        'reference_metrics_reproduced_exactly': True, 'methods': results,
        'IPAA_limit': 'Distances for new scorers use fixed transforms and the author p95 normalization and0.70 threshold; no per-method calibration was fitted.',
        'selection_limit': 'Winners selected on this benchmark after inspecting all fixed arms. Performance outside the sample remains unmeasured.',
        'source_scores_freeze_sha256': sha(OUT / 'scores/freeze.json'),
    }
    save(OUT / 'results.json', summary)
    files = [OUT / 'results.json', OUT / 'plan.json']
    files.extend(OUT / 'metrics' / (name + '.json') for name in results)
    freeze(OUT / 'metrics', [OUT / 'metrics' / (name + '.json') for name in results])
    save(OUT / 'completion.json', {'complete': True, 'training': False,
                                  'result_sha256': sha(OUT / 'results.json'),
                                  'metric_freeze_sha256': sha(OUT / 'metrics/freeze.json')})
    print('TOP3', [(name, results[name]['AP'], results[name]['FPR95'], results[name]['IPAA100']) for name in own[:3]], flush=True)


if __name__ == '__main__':
    main()

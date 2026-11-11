"""Save the sample comparison with an explicit full-publication verification status."""
from common import HERE, ROOT, read, save, sha

TARGETS = {
    'asnet': {
        'name': 'ASNet ResNet-18, original released checkpoint',
        'AP': .524, 'FPR95': .209, 'IPAA100': .170,
        'source': 'https://arxiv.org/html/2007.14878',
        'table': 'Table1 ASNet, no epipolar soft constraint',
    },
    'vit': {
        'name': 'ViT-P3DE* Tiny, released TripletNet_cam_pos checkpoint',
        'AP': .828, 'FPR95': .049, 'IPAA100': .435,
        'source': 'https://ijcai.org/proceedings/2023/149',
        'table': 'Table2 ViT-P3DE* ViT-T, no separate epipolar soft constraint',
    },
}


def main():
    folder = HERE / 'sample100'
    selection = read(folder / 'selection.json')
    rows = {}
    for model, target in TARGETS.items():
        path = folder / model / 'native-metrics.json'
        result = read(path)
        assert result['image_pairs'] == selection['image_pairs'] == 3600
        assert result['candidate_object_pairs'] == selection['candidate_object_pairs']
        assert result['positive_object_pairs'] == selection['positive_object_pairs']
        assert sum(item['image_pairs'] for item in result['scene_outcomes'].values()) == 3600
        assert len(result['scene_outcomes']) == 100
        assert len({item['setup'] for item in result['scene_outcomes'].values()}) == 100
        measured = {key: result[key] for key in ('AP', 'FPR95')}
        measured['IPAA100'] = result['IPAA']['100']
        rows[model] = {
            'name': target['name'], 'published_full_test': target,
            'sample': measured,
            'sample_minus_published_percentage_points': {key: 100 * (measured[key] - target[key]) for key in measured},
            'inference_cpu_mps_and_batch_checks': read(folder / model / 'smoke.json'),
            'elapsed_feature_extraction_seconds': read(folder / model / 'progress.json')['elapsed_seconds'],
            'elapsed_distance_seconds': read(folder / model / 'distance-plan.json')['elapsed_seconds'],
            'elapsed_native_metric_seconds': result['elapsed_seconds'],
            'metrics_sha256': sha(path),
        }
    image_id = __import__('subprocess').check_output(
        ['docker', 'image', 'inspect', 'messytable-published-metrics:20261111', '--format', '{{.Id}}'],
        text=True).strip()
    result = {
        'training': False, 'selection': selection, 'models': rows,
        'metric_environment': {'docker_image_id': image_id, 'architecture': 'linux/amd64',
                               'native_author_KMSolver': True, 'native_author_metric_functions': True},
        'feature_environment': 'Modern PyTorch and torchvision on MPS, unmodified upstream crop/dataset functions and released model definitions. CPU/MPS and scene-batch-size checks passed. Per-model plans record exact versions.',
        'full_published_test_verification': 'Pending. Only the fixed 100-scene sample was run to completion.',
        'interpretation': 'Sample reproduces the ranking and approximate magnitude of the published model results. Population and test-distance95th-percentile normalization differ from the full test split.',
        'scope_after_user_steering': 'User requested a subset. Full-split jobs were not started. Initial12-scene throughput pilot is outside this sample comparison.',
    }
    save(folder / 'comparison.json', result)
    lines = [
        'OFFICIAL MESSYTABLE PROTOCOL — FIXED100-SCENE SAMPLE\n',
        '\n100 scenes from 100 distinct camera setups, 900 photos, 3,600 image pairs.\n',
        f"{selection['annotated_appearances']:,} annotated appearances, {selection['candidate_object_pairs']:,} object-pair candidates, {selection['positive_object_pairs']:,} true object pairs.\n",
        'Difficulty mix 30 easy / 50 medium / 20 hard. Selection fixed by SHA256 order before predictions.\n',
        '\nMetric values in percent\n',
        'Model | population | AP | FPR95 (lower is better) | IPAA100\n',
    ]
    for model in ('asnet', 'vit'):
        for population, values in (('published full test', rows[model]['published_full_test']),
                                   ('our 100-scene sample', rows[model]['sample'])):
            lines.append(f"{model} | {population} | {100*values['AP']:.2f} | {100*values['FPR95']:.2f} | {100*values['IPAA100']:.2f}\n")
    lines.extend([
        '\nAP measures how well true object pairs rank above false pairs across thresholds.\n',
        'FPR95 measures the fraction of negative object pairs accepted at 95% true-pair recall.\n',
        'IPAA100 measures the fraction of camera-view pairs for which every object association and absence decision is correct under the authors matching procedure.\n',
        '\nOriginal author metric functions and shipped Linux KMSolver ran in Python 3.7 Docker.\n',
        'Original authored crop/dataset and model definitions ran on modern Mac PyTorch/MPS. CPU and batch-size checks passed. Native released weights loaded strictly. No training.\n',
        '\nThis sample supports the ranking and approximate scale of the papers results. Exact published full-test numbers remain unverified.\n',
        'The prior 100-pair fixed-budget recall experiment measures a different population and metric.\n',
        'Full official test includes 1,861 scenes and 66,996 image pairs. The full run was paused in favor of this requested sample.\n',
        '\nSources\n', TARGETS['asnet']['source']+'\n', TARGETS['vit']['source']+'\n',
    ])
    (folder / 'RESULTS.txt').write_text(''.join(lines))
    files = [folder / name for name in ('selection.json', 'comparison.json', 'RESULTS.txt')]
    files.extend(folder / model / name for model in ('vit', 'asnet')
                 for name in ('plan.json', 'inference-freeze.json', 'distance-freeze.json', 'native-metrics.json'))
    save(folder / 'completion.json', {'complete': True, 'training': False, 'scope': '100-scene sample only',
                                     'sha256': {str(path.relative_to(folder)): sha(path) for path in files}})
    print(''.join(lines))


if __name__ == '__main__':
    main()

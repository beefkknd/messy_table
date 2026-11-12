"""Audit localization metrics and their matching assumptions using saved boxes.

No detector inference, training, changes to frozen scores, or Part One edits.
"""
from collections import Counter, defaultdict, deque
from itertools import combinations
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FULL = ROOT/'evaluation/messytable-detection-full-test'
PROGRAM = ROOT/'research/self-learning-program'
ARMS = ['vanilla', 'fine_tuned', 'fine_tuned_tiles', 'fine_tuned_tiles_G']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def overlap(detections, targets):
    a = np.asarray([d['box_xyxy'] for d in detections], dtype=float).reshape(-1, 4)
    b = np.asarray(list(targets.values()), dtype=float).reshape(-1, 4)
    intersection = np.maximum(np.minimum(a[:, None, 2:], b[None, :, 2:])
                              - np.maximum(a[:, None, :2], b[None, :, :2]), 0).prod(axis=2)
    union = ((a[:, 2:]-a[:, :2]).prod(axis=1)[:, None]
             + (b[:, 2:]-b[:, :2]).prod(axis=1)[None, :] - intersection)
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def mapping(matrix, cutoff, policy='sum_iou', reverse=False):
    valid = matrix >= cutoff
    weights = np.where(valid, matrix, 0.)
    if policy == 'cardinality_first':
        weights = np.where(valid, matrix+min(matrix.shape)+1, 0.)
    if reverse:
        rows, cols = linear_sum_assignment(weights[::-1, ::-1], maximize=True)
        rows, cols = matrix.shape[0]-1-rows, matrix.shape[1]-1-cols
    else:
        rows, cols = linear_sum_assignment(weights, maximize=True)
    return {int(a): int(b) for a, b in zip(rows, cols) if valid[a, b]}


def forced_and_possible(matrix, chosen, cutoff=.5):
    """Coverage bounds among all maximum-cardinality eligible assignments.

    A non-isolated GT can be matched in some maximum matching. GT vertices
    reachable by an alternating path from an unmatched GT can be exposed in
    another maximum matching, and hence are not necessarily covered.
    """
    valid = matrix >= cutoff
    matched = set(chosen.values())
    possible = set(np.flatnonzero(valid.any(axis=0)).tolist())
    exposed = set(range(matrix.shape[1]))-matched
    queue = deque(exposed)
    while queue:
        column = queue.popleft()
        for prediction in np.flatnonzero(valid[:, column]).tolist():
            partner = chosen.get(prediction)
            if partner is not None and partner not in exposed:
                exposed.add(partner)
                queue.append(partner)
    return matched-exposed, possible


def graph_checks():
    """Exhaustively check coverage bounds against all matchings of 3×3 graphs."""
    for bits in range(512):
        edges = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3,3)
        all_sets = []
        def enumerate_matches(row, used):
            if row == 3:
                all_sets.append(set(used))
                return
            enumerate_matches(row+1, used)
            for column in np.flatnonzero(edges[row]).tolist():
                if column not in used:
                    enumerate_matches(row+1, used|{column})
        enumerate_matches(0, set())
        maximum = max(map(len, all_sets))
        optimum = [s for s in all_sets if len(s)==maximum]
        chosen = mapping(np.where(edges,.8,0.),.5,policy='cardinality_first')
        assert len(chosen)==maximum
        forced, possible=forced_and_possible(np.where(edges,.8,0.),chosen)
        assert possible==set.union(*optimum) and forced==set.intersection(*optimum)
    example=np.array([[1.,.5,0.],[0.,1.,.5],[.5,0.,0.]])
    assert len(mapping(example,.5))==2
    assert len(mapping(example,.5,policy='cardinality_first'))==3
    return {'all_512_three_by_three_eligibility_graphs_checked':True,
            'explicit_iou_sum_vs_match_count_counterexample_checked':True}


def g_gate(det):
    x1, y1, x2, y2 = det['box_xyxy']
    short, long = sorted((x2-x1, y2-y1))
    if det['confidence'] < .8 or short < 8 or long/short > 12 or short*long/(1920*1080) > .25:
        return False
    l, t, r, b = det['source_tile_xyxy']
    distances = ([x1-l] if l > 0 else [])+([y1-t] if t > 0 else [])
    distances += ([r-x2] if r < 1920 else [])+([b-y2] if b < 1080 else [])
    return not distances or min(distances) >= 12


def run_cohort(scene_rows, predictions, truths, expected):
    total = {arm: Counter() for arm in expected}
    per_setup = {arm: defaultdict(Counter) for arm in expected}
    examples = []
    seen = set()
    for index, scene in enumerate(scene_rows):
        sid = scene['id']
        pred = predictions(scene)
        annotated, views = {}, {arm: {} for arm in expected}
        for camera, raw in pred['views'].items():
            assert (sid, camera) not in seen
            seen.add((sid, camera))
            target = truths(scene, camera)
            ids = list(map(int, target))
            assert len(ids) == len(set(ids))
            annotated[camera] = set(ids)
            if 'fine_tuned_tiles_G' in expected:
                base = raw['arms']['fine_tuned']
                original_raw = raw['arms']['fine_tuned_tiles']
                gated = raw['arms']['fine_tuned_tiles_G']
                assert original_raw[:len(base)] == gated[:len(base)] == base
                assert gated == base+[d for d in original_raw[len(base):] if g_gate(d)]
            for arm in expected:
                detections = raw['arms'][arm]
                assert all(d['confidence'] > .25 for d in detections)
                matrix = overlap(detections, target)
                old = mapping(matrix, .5)
                reversed_old = mapping(matrix, .5, reverse=True)
                coverage = mapping(matrix, .5, policy='cardinality_first')
                old75 = mapping(matrix, .75)
                coverage75 = mapping(matrix, .75, policy='cardinality_first')
                assert len(coverage) >= len(old) and len(coverage75) >= len(old75)
                forced, possible = forced_and_possible(matrix, coverage)
                unmatched = sorted(set(range(len(detections)))-set(old))
                eligible_unmatched = int((matrix[unmatched] >= .5).any(axis=1).sum())
                counts = Counter(
                    images=1, ground_truth=len(target), detections=len(detections),
                    localized=len(old), unmapped=len(detections)-len(old),
                    missing=len(target)-len(old), localized_at_iou75=len(old75),
                    unmapped_with_eligible_target=eligible_unmatched,
                    unmapped_without_eligible_target=len(unmatched)-eligible_unmatched,
                    cardinality_first_localized=len(coverage),
                    cardinality_first_localized_iou75=len(coverage75),
                    cardinality_first_unmapped=len(detections)-len(coverage),
                    reversed_sum_iou_localized=len(reversed_old),
                    cardinality_gain_frames=int(len(coverage)>len(old)),
                    reversed_count_change_frames=int(len(reversed_old)!=len(old)),
                    reversed_id_change_frames=int(set(reversed_old.values())!=set(old.values())),
                    empty_prediction_frames=int(not detections), empty_target_frames=int(not target),
                    exact_iou50_edges=int(np.count_nonzero(matrix == .5)),
                    exact_iou75_edges=int(np.count_nonzero(matrix == .75)),
                    potentially_covered_gt=len(possible), necessarily_covered_gt=len(forced),
                    maximum_cardinality_ambiguous_gt=len(possible-forced),
                )
                for detection in detections:
                    l, t, r, b = detection['box_xyxy']
                    counts['degenerate_boxes'] += int(l >= r or t >= b)
                    counts['out_of_frame_boxes'] += int(l < 0 or t < 0 or r > 1920 or b > 1080)
                assert counts['localized']+counts['unmapped'] == counts['detections']
                assert counts['localized']+counts['missing'] == counts['ground_truth']
                assert eligible_unmatched+counts['unmapped_without_eligible_target'] == counts['unmapped']
                total[arm].update(counts)
                per_setup[arm][scene['setup']].update(counts)
                views[arm][camera] = {
                    'original': {ids[j] for j in old.values()},
                    'reversed': {ids[j] for j in reversed_old.values()},
                    'cardinality': {ids[j] for j in coverage.values()},
                    'forced': {ids[j] for j in forced}, 'possible': {ids[j] for j in possible},
                }
                if (len(coverage)>len(old) or set(reversed_old.values())!=set(old.values())) and len(examples)<12:
                    examples.append({'scene': sid, 'camera': camera, 'arm': arm,
                                     'original_matches': old, 'cardinality_matches': coverage,
                                     'original_count': len(old), 'cardinality_count': len(coverage),
                                     'reversed_count': len(reversed_old)})
        for a, b in combinations(annotated, 2):
            shared = annotated[a] & annotated[b]
            for arm in expected:
                local = views[arm]
                count = Counter(shared_objects=len(shared), view_pairs=1)
                for policy, field in [('original','shared_localized_both'),
                                      ('reversed','reversed_shared_localized_both'),
                                      ('cardinality','cardinality_first_shared_localized_both'),
                                      ('forced','coverage_bound_lower_shared'),
                                      ('possible','coverage_bound_upper_shared')]:
                    count[field] = len(shared & local[a][policy] & local[b][policy])
                total[arm].update(count)
                per_setup[arm][scene['setup']].update(count)
        if (index+1) % 300 == 0:
            print('Audited', index+1, '/', len(scene_rows), 'scenes', flush=True)
    records = {}
    for arm, counts in total.items():
        source = expected[arm]
        for key in ('images','ground_truth','detections','localized','unmapped','localized_at_iou75',
                    'shared_objects','shared_localized_both'):
            assert counts[key] == source[key], (arm,key,counts[key],source[key])
        records[arm] = {'counts': dict(counts), 'metrics': {
            'recall': counts['localized']/counts['ground_truth'],
            'precision': counts['localized']/counts['detections'],
            'unmapped_fraction': counts['unmapped']/counts['detections'],
            'miss_rate': counts['missing']/counts['ground_truth'],
            'both_view_coverage': counts['shared_localized_both']/counts['shared_objects'],
            'cardinality_first_recall': counts['cardinality_first_localized']/counts['ground_truth'],
            'cardinality_first_precision': counts['cardinality_first_localized']/counts['detections'],
            'cardinality_first_both_view_coverage': counts['cardinality_first_shared_localized_both']/counts['shared_objects'],
        }}
    return {'arms': records, 'unique_photos': len(seen), 'examples': examples}, per_setup


def main():
    graph_tests=graph_checks()
    # Snapshot the committed/reporting boundary requested by the user.
    part_one = ROOT/'evaluation/messytable-part-one'
    boundary = {str(p.relative_to(ROOT)): sha(p) for p in sorted(part_one.rglob('*'))
                if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts}
    manifest = read(FULL/'images.json')['scenes']
    frozen = read(FULL/'prediction-freeze.json')
    protocol = read(FULL/'protocol.json')
    original = read(FULL/'results.json')
    assert sha(FULL/'prediction-freeze.json') == original['prediction_freeze_sha256']
    assert sha(FULL/'images.json') == frozen['manifest_sha256']
    assert sha(ROOT/'data/messytable-full/MessyTable/labels/test.json') == protocol['native_test_sha256']
    labels = read(ROOT/'data/messytable-full/MessyTable/labels/test.json')['scenes']
    image_meta = {scene['id']: {v['id']:v for v in scene['views']} for scene in manifest}
    def load(scene):
        path = FULL/'predictions'/(scene['id']+'.json')
        assert sha(path) == frozen['scenes'][path.name]
        value = read(path)
        assert value['id'] == scene['id'] and value['setup'] == scene['setup']
        assert set(value['views']) == set(image_meta[scene['id']])
        for camera, data in value['views'].items():
            assert data['sha256'] == image_meta[scene['id']][camera]['sha256']
        return value
    truth = lambda scene, camera: {oid:o['pos'] for oid,o in labels[scene['id']]['cameras'][camera]['instances'].items()}
    full, setups = run_cohort(manifest, load, truth, {arm:v['counts'] for arm,v in original['arms'].items()})
    assert full['unique_photos'] == 16749
    for arm, item in full['arms'].items():
        for name, source in [('recall','detection_recall'),('precision','detection_precision'),('both_view_coverage','both_view_coverage')]:
            assert item['metrics'][name] == original['arms'][arm]['metrics'][source]
        assert abs(item['metrics']['precision']+item['metrics']['unmapped_fraction']-1)<1e-12
    # Independently reconstruct the published setup-bootstrap intervals.
    setup_ids = sorted(setups['vanilla'])
    assert len(setup_ids) == 188
    rng = np.random.default_rng(20261111)
    draws = rng.integers(0, 188, size=(20000, 188))
    fields = ['localized','ground_truth','detections','shared_localized_both','shared_objects']
    pooled = {a: np.array([[setups[a][s][k] for k in fields] for s in setup_ids],dtype=float)[draws].sum(axis=1) for a in ARMS}
    for arm, baseline in [('fine_tuned','vanilla'),('fine_tuned_tiles','fine_tuned'),('fine_tuned_tiles_G','fine_tuned')]:
        contrast = original['paired_setup_bootstrap'][arm+'_minus_'+baseline]
        for name, num, den in [('detection_recall',0,1),('detection_precision',0,2),('both_view_coverage',3,4)]:
            delta=pooled[arm][:,num]/pooled[arm][:,den]-pooled[baseline][:,num]/pooled[baseline][:,den]
            actual=100*np.quantile(delta,[.025,.975])
            assert np.allclose(actual,contrast[name]['setup_bootstrap_ci95_percentage_points'],rtol=0,atol=1e-10)
    # Historical displayed figures use their original frozen cohort and keys.
    old_manifest = read(PROGRAM/'data/cohorts/original100.json')['scenes']
    old_truth = {r['id']:r for r in read(PROGRAM/'data/scoring/original100.json')['scenes']}
    sources = {'vanilla':PROGRAM/'engineering/inference/pretrained768/original100/predictions.json',
               'fine_tuned':PROGRAM/'engineering/round2/inference/A/original100/predictions.json',
               'fine_tuned_tiles':PROGRAM/'step0-tiles/engineering/inference/original100/predictions.json'}
    old_predictions = {arm:{r['id']:r for r in read(p)['scenes']} for arm,p in sources.items()}
    def old_load(scene):
        return {'views': {v['id']:{'arms':{arm:[d for d in runs[scene['id']]['views'][v['id']]['detections'] if d['confidence']>.25]
                                                 for arm,runs in old_predictions.items()}} for v in scene['views']}}
    def old_key(scene,camera):
        item=old_truth[scene['id']]['views'][camera]
        if 'objects_xywh_historical' in item:
            return {oid:[b[0],b[1],b[0]+b[2],b[1]+b[3]] for oid,b in item['objects_xywh_historical'].items()}
        return item['objects_xyxy']
    old_counts = read(PROGRAM/'data/round2-detector-comparison.json')['cohorts']['original100']['fixed_0_25']['arms']['pretrained768']['counts']
    tiles = read(PROGRAM/'step0-tiles/data/detector-comparison.json')['cohorts']['original100']['arms']
    historical,_ = run_cohort(old_manifest,old_load,old_key,{'vanilla':old_counts,'fine_tuned':tiles['full_frame']['counts'],'fine_tuned_tiles':tiles['tiled_merged']['counts']})
    teacher = read(PROGRAM/'stronger-teacher/data/teacher-quality.json')
    tc,tm=teacher['counts'],teacher['metrics']
    for iou in [50,75]:
        for prefix in ['single','ladder','supplement']:
            assert tc[prefix+'_mapped_'+str(iou)]/tc[prefix+'_seeds']==tm[prefix+'_precision_iou'+str(iou)]
    for prefix in ['single','ladder']:
        assert tc[prefix+'_mapped_50']/tc['gt_endpoints']==tm[prefix+'_recall_iou50']
    assert tc['ladder_mapped_50']-tc['single_mapped_50']==tc['newly_mapped_50']-tc['previously_mapped_lost_50']
    for path,expected in boundary.items():
        assert sha(ROOT/path)==expected, 'Part One changed during audit: '+path
    result={'passed':True,'new_inference':False,'new_training':False,
            'software':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__},
            'graph_tests':graph_tests,
            'full_test':full,'historical100':historical,
            'teacher_aggregate_arithmetic_verified':True,'bootstrap_reconstruction_verified':True,
            'part_one_files_unchanged':True,'part_one_boundary_sha256':boundary,
            'source_results_sha256':sha(FULL/'results.json'),'source_predictions_freeze_sha256':sha(FULL/'prediction-freeze.json'),
            'audit_code_sha256':sha(Path(__file__)),
            'definitions':{'recall':'Matched ground-truth appearances / all annotated appearances; micro pooled',
                           'precision':'Matched predictions / all predictions; class-agnostic localization, fixed confidence >.25 and IoU >=.50',
                           'unmapped':'All predictions minus matched predictions under the frozen assignment policy',
                           'missing':'All ground-truth appearances minus matched ground-truth appearances',
                           'matching':'Frozen policy maximizes sum of IoU. Sensitivity maximizes match count first, IoU second.',
                           'tie_probe':'Reverse both prediction and annotation order under the same IoU-sum objective; observed sensitivity, not exhaustive bounds.',
                           'coverage_bounds':'Lower/upper bounds over all maximum-cardinality eligible assignments. Upper bound need not be jointly attainable across all links.'}}
    save(HERE/'metric-audit.json',result)
    print(json.dumps({'passed':True,'full_test':full['arms'],'historical':historical['arms']},indent=2),flush=True)


if __name__=='__main__':
    main()

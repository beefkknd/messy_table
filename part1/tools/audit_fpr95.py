"""Audit all frozen FPR95 cutoffs in the exact preserved author runtime.

No inference, score changes or training. Check ties after author normalization,
including any extra floating-point ties introduced by that normalization.
"""
from pathlib import Path
import hashlib
import json
import math
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SAMPLE = ROOT/'evaluation/messytable-official-reproduction/sample100'
SWEEP = SAMPLE/'step2'
NATIVE = ROOT/'evaluation/messytable-vit-p3de-100/assets/ViT-P3DE/src/utils.py'
IMAGE = 'messytable-published-metrics:20261111'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    results = json.loads((SWEEP/'results.json').read_text())
    freeze = json.loads((SWEEP/'scores/freeze.json').read_text())
    for name, saved in results['methods'].items():
        path = SWEEP/'scores'/(name+'.npy')
        assert sha(path) == freeze[path.name] == saved['score_sha256']
        assert sha(NATIVE) == saved['metric_code_sha256']
    code = """
import ast,json,math
from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler
p=Path('/workspace/evaluation/messytable-vit-p3de-100/assets/ViT-P3DE/src/utils.py')
nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('scale_data','FPR_95')]
scope={'np':np,'MinMaxScaler':MinMaxScaler}
exec(compile(ast.Module(body=nodes),str(p),'exec'),scope)
base=Path('/workspace/evaluation/messytable-official-reproduction/sample100')
truth=np.load(base/'vit/gt_inst.npy')
labels=truth.reshape(-1).astype(bool)
positives=int(labels.sum()); negatives=len(labels)-positives
needed=int(math.ceil(.95*positives)); methods={}
names=__NAMES__
for name in names:
    distance=np.load(base/'step2/scores'/(name+'.npy'))
    scores=(1-scope['scale_data'](distance)).reshape(-1)
    assert len(scores)==len(labels) and np.isfinite(scores).all()
    positive_scores=scores[labels]
    kth=len(positive_scores)-needed
    cutoff=float(np.partition(positive_scores,kth)[kth])
    before=scores>cutoff; tied=scores==cutoff
    pb=int(np.count_nonzero(before&labels)); nb=int(np.count_nonzero(before&~labels))
    pt=int(np.count_nonzero(tied&labels)); nt=int(np.count_nonzero(tied&~labels))
    assert pb<needed<=pb+pt
    tie_distances=distance.reshape(-1)[tied]
    methods[name]={
      'author_reproduced_fpr95':float(scope['FPR_95'](distance,truth)),
      'cutoff_normalized_similarity':cutoff,
      'cutoff_distance_min':float(tie_distances.min()),
      'cutoff_distance_max':float(tie_distances.max()),
      'positive_before_cutoff':pb,'negative_before_cutoff':nb,
      'positive_at_cutoff':pt,'negative_at_cutoff':nt,'candidates_at_cutoff':pt+nt,
      'recall_before_tied_block':pb/positives,
      'recall_accepting_entire_tied_block':(pb+pt)/positives,
      'fpr_before_tied_block':nb/negatives,
      'fpr_accepting_entire_tied_block':(nb+nt)/negatives,
      'arbitrary_row_order_fpr95_min':nb/negatives,
      'arbitrary_row_order_fpr95_max':(nb+nt)/negatives,
    }
print(json.dumps({'candidate_pairs':len(labels),'positive_pairs':positives,'negative_pairs':negatives,
 'required_positive_count_for_95_recall':needed,'methods':methods}))
""".replace('__NAMES__', repr(list(results['methods'])))
    completed = subprocess.run([
        'docker','run','--rm','--platform','linux/amd64','-v',str(ROOT)+':/workspace:ro',
        IMAGE,'python','-c',code,
    ], capture_output=True, text=True, check=True)
    native = json.loads(completed.stdout)
    methods = native['methods']
    for name, row in methods.items():
        assert row['author_reproduced_fpr95'] == results['methods'][name]['FPR95'], name
        mixed = row['positive_at_cutoff'] > 0 and row['negative_at_cutoff'] > 0
        changes = mixed and round(100*row['arbitrary_row_order_fpr95_min'], 2) != round(100*row['arbitrary_row_order_fpr95_max'], 2)
        row.update({
            'status': 'tied_cutoff' if changes else 'minor_cutoff_tie' if mixed else 'valid',
            'has_mixed_cutoff_tie': mixed,
            'display_changes_with_tie_order_at_2_decimals': changes,
            'report_fpr95': None if changes else row['author_reproduced_fpr95'],
            'source_score_sha256': freeze[name+'.npy'],
        })
    invalid = [name for name, row in methods.items() if row['status'] == 'tied_cutoff']
    guarded_order = sorted(methods, key=lambda name: (
        -results['methods'][name]['AP'],
        math.inf if methods[name]['report_fpr95'] is None else methods[name]['report_fpr95'],
        -results['methods'][name]['IPAA100'], name))
    assert results['ranking_including_references'] == guarded_order
    image_id = subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],
                              capture_output=True, text=True, check=True).stdout.strip()
    audit = {
        'source_results_sha256': sha(SWEEP/'results.json'),
        'source_score_freeze_sha256': sha(SWEEP/'scores/freeze.json'),
        'source_labels_sha256': sha(SAMPLE/'vit/gt_inst.npy'),
        'native_metric_code_sha256': sha(NATIVE), 'audit_code_sha256': sha(Path(__file__)),
        'native_runtime_image': IMAGE, 'native_runtime_image_id': image_id,
        'normalization': 'Exact upstream scale_data + 1-distance inside the pinned native runtime',
        **native, 'invalid_fpr95_methods': invalid,
        'minor_cutoff_tie_methods': [name for name, row in methods.items() if row['status'] == 'minor_cutoff_tie'],
        'display_policy': 'N/A and omit from FPR95 chart when cutoff tie ordering changes the reported percentage at two decimal places; annotate smaller ties without hiding their audit.',
        'native_rerun_matches_saved_values': True,
        'AP_and_IPAA_changed': False, 'ranking_unchanged_with_guarded_tiebreak': True,
        'interpretation': [
            'The original code counts a prefix of sorted rows, including a partial tied block.',
            'Different tie permutations can produce different FPR values for the same scores.',
            'Accepting an entire tied block gives a deterministic threshold operating point; report it separately.',
            'N/A flags a non-unique threshold interpretation that changes the displayed result, not a failed reproduction.',
        ],
    }
    (HERE/'fpr95-tie-audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'invalid_methods': invalid, 'minor_ties': audit['minor_cutoff_tie_methods'],
                      'all_native_values_reproduced': True,
                      'diagnostics': {n: methods[n] for n in invalid}}, indent=2), flush=True)


if __name__ == '__main__':
    main()

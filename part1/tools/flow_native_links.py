"""Replay the author's native KMSolver for the leading blend; export every view pair's predicted links."""
import json, sys
import numpy as np
sys.path.insert(0, '/proj/evaluation/messytable-vit-p3de-100/assets/ViT-P3DE/src')
import utils
R = '/proj/evaluation/messytable-official-reproduction/sample100'
n = 'blend_roma_vit_reference_dense_first'
d = np.load(R + '/step2/scores/%s.npy' % n).reshape(-1)
gt = np.load(R + '/vit/gt_inst.npy').reshape(-1)
pairs = json.load(open(R + '/vit/pair-index.json'))
p95 = float(np.percentile(d, 95)); c = w = 0; links = {}
for p in pairs:
    D = d[p['start']:p['stop']].reshape(len(p['main_bbox_id']), -1) / p95
    G = gt[p['start']:p['stop']].reshape(D.shape)
    P = utils.apply_km(D, .70).astype(bool)
    c += int((P & (G == 1)).sum()); w += int((P & (G == 0)).sum())
    i, j = np.nonzero(P)
    links['%s/%d-%d' % (p['scene'], p['camera_A'], p['camera_B'])] = [[p['main_bbox_id'][a], p['sec_bbox_id'][b]] for a, b in zip(i, j)]
json.dump({'method': n, 'correct': c, 'wrong': w, 'p95': p95, 'links': links}, open('/proj/evaluation/messytable-part-one/flow-native-links.json', 'w'))
print(n, 'correct', c, 'wrong', w)

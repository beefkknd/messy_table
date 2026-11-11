"""The preserved official-protocol sample and fixed Step 2 sweep inputs."""
import sys

from common import DATA, EVAL, HERE, LABELS, ROOT, read, save, sha

SAMPLE = HERE / 'sample100'
OUT = SAMPLE / 'step2'
ROMA = EVAL / 'messytable-roma-100'
OLD = EVAL / 'messytable-annotated-step2'
EDGES = ROOT / 'research/self-learning-program/tile-validation/edge-mask-study/engineering'
LOCAL = ROOT / 'research/self-learning-program/tile-validation/staged-verifier/engineering'
VIEWPOINT = ROOT / 'research/self-learning-program/tile-validation/viewpoint-study'
sys.path[:0] = [str(EVAL), str(ROMA), str(ROOT / 'everyday-identity/scripts')]
SELECTION_SHA = '35fc135826c641d68134336e361b88f924450486f43455a4c58b93388eb19189'

# The reused image-transform modules call this hash helper by its older name.
import common
common.sha256 = sha


def inputs():
    selection = read(SAMPLE / 'selection.json')
    assert sha(SAMPLE / 'selection.json') == SELECTION_SHA
    assert sha(LABELS) == selection['source_sha256']['test.json']
    all_scenes = read(LABELS)
    scenes = {row['scene']: all_scenes['scenes'][row['scene']] for row in selection['scenes']}
    return selection, scenes, all_scenes['intrinsics']


def view_path(scene, camera):
    return DATA / 'images' / f'{scene}-0{camera}.jpg'


def annotations(scene):
    return {cam: [{'box_xyxy': item['pos']} for item in record['instances'].values()]
            for cam, record in scene['cameras'].items()}


def freeze(folder, files):
    save(folder / 'freeze.json', {str(path.relative_to(folder)): sha(path) for path in files})


def verify(folder, manifest='freeze.json'):
    for name, digest in read(folder / manifest).items():
        assert sha(folder / name) == digest, (folder, name)


def main():
    selection, scenes, _ = inputs()
    OUT.mkdir(exist_ok=True)
    plan = {
        'training': False, 'selection_sha256': SELECTION_SHA,
        'scenes': selection['sample_scenes'], 'setups': selection['sample_setups'],
        'images': selection['images'], 'image_pairs': selection['image_pairs'],
        'candidate_object_pairs': selection['candidate_object_pairs'],
        'positive_object_pairs': selection['positive_object_pairs'],
        'box_source': 'Same native human annotation boxes and endpoint order as cached ASNet/ViT sample baselines.',
        'rank_primary': 'AP descending, then FPR95 ascending, then IPAA100 descending.',
        'methods': [
            'Frozen ASNet and ViT-P3DE controls',
            'DINO tight,1.25x,1.5x,2x crops; mean,max,second-highest crop scores',
            'DINO SAM foreground,GrabCut foreground,grayscale,Canny',
            'DINO RGB/SAM quarter-turns,per-edge max and shared-view rotation selected by appearance/anchors',
            'DINO calibrated epipolar soft penalty and padded-box feasibility gate',
            'Foreground SIFT with held-out homography validation, native-patch control and rectified-patch DINO rescoring',
            'RoMa v2 full-scene box/SAM coverage,minimum and symmetric-mean evidence',
            'Previously used50/50 and75/25 dense/appearance percentile blends with DINO context,SAM and ASNet',
            'Fixed50/50 and75/25 RoMa/ViT percentile blends',
        ],
        'native_scoring': 'Original AP,FPR95,IPAA functions and shipped KMSolver in the pinned Python3.7 Docker runtime.',
        'distance_maps': 'DINO1-cosine,RoMa1-coverage,rank fusion1-combined percentile. Native ASNet/ViT distances unchanged.',
        'IPAA_interpretation': 'Same author p95 normalization and0.70 threshold for all arms. New score distances have no fitted threshold calibration, so AP/FPR are the primary ranking evidence.',
        'geometry': 'Use native camera calibration, no instance identities. Reuse fixed soft penalty0.10,free residual0.25 and prior padded-box rule.',
        'perspective': 'Reuse prior foreground SIFT, reciprocal ratio,holdout homography and common-surface checks. Shortlist top2 per lane and direction from RGB,gray,Canny,GrabCut,SAM. Keep all Cartesian edges in final AP population.',
        'gate_distance': 'Incompatible box geometry receives DINO distance2.0. Every candidate stays in the metric population.',
        'dense_compute': '3600 RoMa pairs; reuse each source-view SAM mask across the8 paired views. Save object-level evidence, no full warp archive.',
        'policy_freeze': 'No training or label-dependent choice of weights,geometry settings,methods or sample membership. Select top3 after full sweep.',
        'interpretation': 'Top3 are exploratory winners on this reused sample. Future generalization evaluation needs data outside this selection.',
        'script_sha256': sha(__file__),
    }
    if (OUT / 'plan.json').exists():
        assert read(OUT / 'plan.json') == plan
    else:
        save(OUT / 'plan.json', plan)
    print({key: plan[key] for key in ('scenes','images','image_pairs','candidate_object_pairs')})


if __name__ == '__main__':
    main()

"""Run released ViT-P3DE ViT-Tiny on the frozen RoMa candidate graph."""
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torchvision import transforms
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVAL = HERE.parent
VENDOR = HERE / 'assets/ViT-P3DE'
WEIGHTS = HERE / 'assets/TripletNet_cam_pos.pth'
CONFIG = VENDOR / 'models/tripletnet_1gpu_all_pairs_vit_tiny_p3de/train.yaml'
ROMA = EVAL / 'messytable-roma-100'
CAMERAS = ROOT / 'research/self-learning-program/data/cohorts/original100.json'
TRUTH = ROOT / 'data/messytable-100/answer-key.json'
sys.path[:0] = [str(HERE / 'assets/python'), str(VENDOR / 'src'), str(EVAL),
                str(EVAL / 'messytable-foundyou-native-retrieval-100')]
import timm
from model import TripletNet_cam_pos
from score_extension import ADDED, load_extension, score_methods
from score_foundyou_native_retrieval import global_greedy, prediction_rows, read, save, sha


def load_model(device):
    config = yaml.safe_load(CONFIG.read_text())
    assert config['features_net'] == 'FeaturesDeitTiny_P3DE'
    assert config['zoomout_ratio'] == [2] and config['grid'] == 60
    assert config['sum'] and config['scale'] == .1
    config['device'] = device

    def local_backbone(repo, name, pretrained):
        # The complete released state dict supplies all parameters.
        assert repo == 'facebookresearch/deit:main' and name == 'deit_tiny_patch16_224'
        return timm.create_model(name, pretrained=False)

    with patch.object(torch.hub, 'load', side_effect=local_backbone):
        model = TripletNet_cam_pos(config)
    state = torch.load(WEIGHTS, weights_only=True, map_location='cpu')
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


def crop(image, box):
    # Upstream crop_feat(..., zoomout_ratio=2): inclusive end, clamp to frame.
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) / 2, (y2 - y1) / 2
    return image[int(max(0, y1-dy)):int(min(height, y2+1+dy)),
                 int(max(0, x1-dx)):int(min(width, x2+1+dx)), :]


PREPROCESS = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
                                 transforms.ToTensor()])


@torch.inference_mode()
def encode(model, views, graph, device):
    result = {}
    for view in views:
        side = view['id']
        image = cv2.imread(str(ROOT / view['path']))
        assert image.shape == (1080, 1920, 3)
        camera = int(view['camera_id'])
        assert 1 <= camera <= 9
        boxes = np.asarray([d['box_xyxy'] for d in graph['detections'][side]], dtype=np.int64).reshape(-1, 4)
        assert ((boxes >= 0) & (boxes <= [1920,1080,1920,1080])).all()
        assert (boxes[:, 2:] > boxes[:, :2]).all()
        # The upstream evaluator uses OpenCV BGR directly with ToTensor.
        parts = []
        for start in range(0, len(boxes), 48):
            batch = boxes[start:start+48]
            images = torch.stack([PREPROCESS(crop(image, box)) for box in batch])
            cameras = torch.full((len(batch),), camera, dtype=torch.int64, device=device)
            positions = torch.as_tensor(batch, dtype=torch.float64)
            # MPS lacks float64. Detector coordinates are integers <= 1920,
            # so the upstream center-bin calculation agrees in float32.
            if device.type == 'mps':
                bins = (positions[:, :2] + positions[:, 2:]) / 120
                assert torch.equal(bins.long(), bins.float().long())
            features = model.forward_once(images.to(device), cameras, positions.float().to(device))['feat']
            parts.append(features.cpu().numpy())
        result[side] = np.concatenate(parts) if parts else np.empty((0, 192), dtype=np.float32)
        assert result[side].shape == (len(boxes), 192) and np.isfinite(result[side]).all()
    return result


def main():
    assert not (HERE / 'scores-summary.json').exists(), 'Preserve completed results'
    start = time.monotonic()
    _, graphs, _, _, _ = load_extension()
    roma_plan = read(ROMA / 'plan.json')
    roma_reference = read(ROMA / 'scores-summary.json')
    assert sorted(graphs) == roma_plan['cases'] == list(range(1, 101))
    graph_hashes = read(ROMA / 'candidate-freeze.json')
    for case in graphs:
        folder = (EVAL / 'messytable-foundyou-native-retrieval-100' / f'case-{case:03}/graph.json'
                  if case in ADDED else EVAL / 'messytable-foundyou-native-retrieval/graphs' / f'case-{case:03}.json')
        assert sha(folder) == graph_hashes[str(case)]
    views = {int(scene['id'].split('-')[1]): scene['views'] for scene in read(CAMERAS)['scenes']}
    assert views.keys() == graphs.keys()
    for case, pair in views.items():
        assert sorted(view['id'] for view in pair) == ['A', 'B']
        for view in pair:
            assert view['path'] == f"data/messytable-100/case-{case:03}/{view['id']}.jpg"
            assert sha(ROOT / view['path']) == view['sha256'] == roma_plan['input_image_sha256'][view['path']]
    box_count = sum(len(graph['detections'][side]) for graph in graphs.values() for side in 'AB')
    edge_count = sum(len(graph['edges']) for graph in graphs.values())
    assert box_count == roma_reference['observations']['detector_boxes'] == 3276
    assert edge_count == roma_reference['observations']['candidate_edges'] == 7115
    budgets = roma_plan['budgets']
    assert budgets == [100, 200, 411, 431]
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_num_threads(8)
    plan = {
        'training': False, 'cohort': 'Original100', 'pairs': sorted(graphs), 'boxes': box_count,
        'shortlist_edges': edge_count, 'budgets': budgets, 'device': str(device),
        'checkpoint_url': 'https://drive.google.com/file/d/1KS7Yafe1MTNTMt2HF4rrXyyKNdL9S2Vo/view',
        'checkpoint_sha256': sha(WEIGHTS),
        'upstream_commit': subprocess.check_output(['git','-C',str(VENDOR),'rev-parse','HEAD'],text=True).strip(),
        'upstream_config_sha256': sha(CONFIG), 'upstream_model_sha256': sha(VENDOR / 'src/model.py'),
        'camera_manifest_sha256': sha(CAMERAS), 'roma_candidate_freeze_sha256': sha(ROMA / 'candidate-freeze.json'),
        'roma_score_sha256': sha(ROMA / 'scores-summary.json'),
        'preprocessing': 'OpenCV BGR, official 2x crop, bilinear 224x224, ToTensor, original camera ID and box position.',
        'score': 'Negative L2 distance of native embeddings with learned camera and position tokens. No separate ESC.',
        'assignment': 'Same ASNet shortlist, global greedy one-to-one and fixed budgets as RoMa.',
        'versions': {name:importlib.metadata.version(name) for name in ('torch','torchvision','timm','opencv-python')},
        'code_sha256': sha(__file__),
    }
    save(HERE / 'plan.json', plan)
    model = load_model(device)
    # Compare model ports across CPU and MPS on identical detector crops.
    sample_case = 1
    cpu = encode(model, views[sample_case], graphs[sample_case], torch.device('cpu'))
    model.to(device)
    trial = encode(model, views[sample_case], graphs[sample_case], device)
    max_delta = max(float(np.max(np.abs(cpu[s] - trial[s]))) for s in 'AB')
    assert all(np.allclose(cpu[s], trial[s], atol=2e-4, rtol=1e-4) for s in 'AB'), max_delta
    print(f'Strict checkpoint load and CPU/{device} agreement passed (max {max_delta:.3g}).', flush=True)
    matrices = {}
    features = {}
    for case in sorted(graphs):
        graph = graphs[case]
        encoded = trial if case == sample_case else encode(model, views[case], graph, device)
        features.update({f'{case:03}_{side}': encoded[side] for side in 'AB'})
        shape = tuple(len(encoded[side]) for side in 'AB')
        matrix = np.full(shape, np.nan, dtype=np.float32)
        a, b = np.asarray(graph['edges'], dtype=int).reshape(-1, 2).T
        matrix[a, b] = -np.linalg.norm(encoded['A'][a] - encoded['B'][b] + 1e-6, axis=1)
        matrices[case] = matrix
        if case % 25 == 0:
            print(f'Encoded {case}/100 pairs in {time.monotonic()-start:.1f}s', flush=True)
    np.savez_compressed(HERE / 'embeddings.npz', **features)
    ranked = global_greedy(matrices)
    assert len(ranked) >= max(budgets)
    predictions = {}
    for budget in budgets:
        links = {}
        for link in ranked[:budget]:
            links.setdefault(link['case'], []).append({k:v for k,v in link.items() if k != 'case'})
        predictions[f'vit_p3de_tiny__global_B{budget}'] = prediction_rows(graphs, links)
    save(HERE / 'rankings.json', ranked)
    save(HERE / 'predictions.json', predictions)
    freeze = {name:sha(HERE / name) for name in ('plan.json','embeddings.npz','rankings.json','predictions.json')}
    save(HERE / 'prediction-freeze.json', freeze)
    assert sha(TRUTH) == roma_reference['answer_key_sha256']
    truths = {}
    for truth in read(TRUTH)['cases']:
        truth['same_vehicle_ids'] = truth['shared_object_ids']
        for scene in truth['scenes'].values():
            scene['vehicles'] = scene['objects']
        truths[truth['case']] = truth
    # Confirm identical evaluator behavior by replaying the RoMa baseline.
    reference_method = 'roma_mask__global_B431'
    control = read(ROMA / 'predictions.json')[reference_method]
    assert all(row['detections'] == graphs[row['case']]['detections'] for row in control)
    replay, _ = score_methods({reference_method:control}, truths, sorted(graphs))
    recorded = roma_reference['methods'][reference_method]
    assert all(replay[reference_method][group][key] == value
               for group, row in recorded.items() for key,value in row.items() if key != 'zero_evidence_proposals')
    results, scored = score_methods(predictions, truths, sorted(graphs))
    assert all(row['overall']['expected_links'] == 1696 and row['overall']['detected_in_both'] == 606
               for row in results.values())
    save(HERE / 'scored-predictions.json', scored)
    summary = {
        'methods': results, 'scored_pairs': 100, 'true_links': 1696, 'detected_in_both': 606,
        'roma_control_reproduced': True, 'cpu_device_embedding_max_abs_delta': max_delta,
        'answer_key_sha256': sha(TRUTH), 'prediction_freeze_sha256': sha(HERE / 'prediction-freeze.json'),
        'wall_seconds': time.monotonic()-start,
    }
    save(HERE / 'scores-summary.json', summary)
    for name, row in results.items():
        print(name, row['overall'], flush=True)


if __name__ == '__main__':
    main()

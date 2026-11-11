"""Local, full-precision inference for the released RoMa v2 checkpoint."""
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import importlib
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ASSETS = HERE / 'assets'
VENDOR = ASSETS / 'RoMaV2'
DINO = ASSETS / 'dinov3'
WEIGHTS = ASSETS / 'romav2.0.1.pt'
CHECKSUM = '1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7'
RESOLUTION = 640

sys.path.insert(0, str(VENDOR / 'src'))
from romav2 import RoMaV2


def set_device(model, device):
    device = torch.device(device)
    for name, module in tuple(sys.modules.items()):
        if name.startswith('romav2.') and isinstance(getattr(module, 'device', None), torch.device):
            module.device = device
    return model.to(device) if model is not None else None


def build_model(device):
    set_device(None, 'cpu')
    with WEIGHTS.open('rb') as file:
        assert hashlib.file_digest(file, 'sha256').hexdigest() == CHECKSUM
    original_load = torch.hub.load
    original_load_state = torch.nn.Module.load_state_dict
    loaded = []

    def local_descriptor(*, repo_or_dir, model, **kwargs):
        assert repo_or_dir.endswith('adc254450203739c8149213a7a69d8d905b4fcfa')
        assert model == 'dinov3_vitl16' and kwargs['pretrained'] is False
        return original_load(str(DINO), model, source='local', pretrained=False)

    def local_weights(url, **kwargs):
        assert url.endswith('/v2.0.1/romav2.0.1.pt')
        return torch.load(WEIGHTS, map_location='cpu', weights_only=True)

    def audit_load(module, state, *args, **kwargs):
        report = original_load_state(module, state, *args, **kwargs)
        if isinstance(module, RoMaV2):
            assert set(state) == set(module.state_dict())
            assert not report.missing_keys and not report.unexpected_keys
            loaded.append(len(state))
        return report

    default = RoMaV2.Cfg()
    cfg = replace(default, setting='base',
                  descriptor=replace(default.descriptor, enable_amp=False),
                  matcher=replace(default.matcher, enable_amp=False))
    with (patch.object(torch.hub, 'load', local_descriptor),
          patch.object(torch.hub, 'load_state_dict_from_url', local_weights),
          patch.object(torch.nn.Module, 'load_state_dict', audit_load)):
        model = RoMaV2(cfg).float().eval().requires_grad_(False)
    assert len(loaded) == 1
    model.bidirectional = True
    return set_device(model, device), loaded[0]


def input_tensor(path, resolution=RESOLUTION):
    with Image.open(path) as image:
        pixels = np.array(image.convert('RGB'))
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)[None].float() / 255
    return F.interpolate(tensor, size=(resolution, resolution), mode='bicubic',
                         align_corners=False, antialias=True)


@torch.inference_mode()
def predict(model, images, device):
    # The release hardcodes AMP in two inference heads. Use FP32 throughout on
    # both devices, with the same CPU resize and author forward path.
    with patch.object(torch, 'autocast', lambda *args, **kwargs: nullcontext()):
        outputs = model(*(tensor.to(device) for tensor in images))
    result = {}
    for direction in ('AB', 'BA'):
        warp = outputs['warp_' + direction][0].float().cpu().numpy()
        overlap = outputs['confidence_' + direction][0, ..., 0].sigmoid().float().cpu().numpy()
        assert np.isfinite(warp).all() and np.isfinite(overlap).all()
        assert warp.shape == (*overlap.shape, 2)
        result['warp_' + direction] = warp
        result['overlap_' + direction] = overlap
    return result

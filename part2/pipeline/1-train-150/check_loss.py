"""Verify both YOLO26 branches against the stock detection loss on fixed tensors."""
import argparse
import json
from pathlib import Path

from common import initialize_runtime, pin_ultralytics, sha256, WEIGHTS, WEIGHTS_SHA, write_json

initialize_runtime()

import torch
from ultralytics import YOLO
from ultralytics.utils.loss import E2ELoss

from partial_loss import PartialDetectionLoss
from train_detector import new_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    pin_ultralytics()
    assert sha256(WEIGHTS) == WEIGHTS_SHA
    torch.manual_seed(923)
    torch.set_num_threads(8)
    device = args.device
    model = new_model(YOLO(str(WEIGHTS)).model, "one2one").to(device)
    stock = E2ELoss(model)
    partial = E2ELoss(model, loss_fn=PartialDetectionLoss)
    feats = [torch.zeros((2, 1, n, n), device=device) for n in (8, 4, 2)]
    anchor_count = sum(feat.shape[-1] ** 2 for feat in feats)
    boxes = torch.full((2, 4, anchor_count), 2.0, device=device, requires_grad=True)
    logits = torch.zeros((2, 1, anchor_count), device=device, requires_grad=True)
    preds = {"feats": feats, "boxes": boxes, "scores": logits}
    batch = {
        "batch_idx": torch.tensor([0, 1], device=device),
        "cls": torch.zeros((2, 1), device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]] * 2, device=device),
        "incomplete": torch.tensor([False, True], device=device),
    }
    results = {}
    for branch in ("one2many", "one2one"):
        reference = getattr(stock, branch)
        adapter = getattr(partial, branch)
        targets_stock, loss_stock, _ = reference.get_assigned_targets_and_loss(preds, batch)
        targets_partial, loss_partial, _ = adapter.get_assigned_targets_and_loss(preds, batch)
        foreground = targets_stock[0]
        grad_stock = torch.autograd.grad(loss_stock[1], logits, retain_graph=True)[0][:, 0]
        grad_partial = torch.autograd.grad(loss_partial[1], logits, retain_graph=True)[0][:, 0]
        assert torch.equal(foreground, targets_partial[0])
        assert torch.equal(loss_stock[[0, 2]], loss_partial[[0, 2]])
        assert torch.equal(grad_stock[0], grad_partial[0])
        assert foreground[1].any()
        assert torch.equal(grad_stock[1, foreground[1]], grad_partial[1, foreground[1]])
        assert (grad_partial[1, foreground[1]] != 0).all()
        assert (grad_stock[1, ~foreground[1]] > 0).all()
        assert (grad_partial[1, ~foreground[1]] == 0).all()
        complete_batch = {**batch, "incomplete": torch.zeros_like(batch["incomplete"])}
        _, loss_complete, _ = adapter.get_assigned_targets_and_loss(preds, complete_batch)
        assert torch.equal(loss_stock, loss_complete)
        results[branch] = {
            "trusted_gradient_identical": True,
            "assigned_positive_anchors_preserved_on_partial": int(foreground[1].sum()),
            "unknown_negative_gradients_removed_on_partial": int((~foreground[1]).sum()),
            "box_losses_identical": True,
            "all_complete_loss_identical": True,
        }
    print(json.dumps(results, indent=2))
    if args.out:
        write_json(args.out, results)


if __name__ == "__main__":
    main()

"""Unknown-region loss for the pinned Ultralytics detector."""
from ultralytics.utils.loss import v8DetectionLoss


def mask_unknown(bce, targets, incomplete):
    """Keep full BCE on trusted frames, and positive targets on partial images."""
    return bce * (~incomplete[:, None, None] | (targets > 0))


class PartialDetectionLoss(v8DetectionLoss):
    """Preserve stock assignment, box regression, and BCE normalization."""

    def get_assigned_targets_and_loss(self, preds, batch, **assign_kwargs):
        # The pinned base method calls self.bce exactly once, after task-aligned
        # assignment. Install the mask for this call only.
        incomplete = batch["incomplete"]
        stock_bce = self.bce

        def masked_bce(logits, targets):
            return mask_unknown(stock_bce(logits, targets), targets, incomplete)

        self.bce = masked_bce
        try:
            return super().get_assigned_targets_and_loss(preds, batch, **assign_kwargs)
        finally:
            self.bce = stock_bce

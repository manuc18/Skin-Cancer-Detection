"""
src/training/losses.py
Loss functions for imbalanced binary classification.

Focal Loss motivation (EDA finding 1):
  ~3.5% positive rate means standard BCE is dominated by easy negatives.
  Focal Loss down-weights well-classified easy examples (gamma > 0),
  focusing training on hard examples and rare positives.

  Lin et al. (2017): FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
    - gamma=0: reduces to BCE
    - gamma=2: the recommended setting from the original paper
    - alpha: class weight balancing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.

    Args:
        alpha: Weight for positive class. Set < 0.5 to penalize false negatives less.
               Typical range [0.25, 0.75]. alpha=0.25 from RetinaNet paper.
        gamma: Focusing parameter. gamma=0 → BCE. gamma=2 → standard focal.
               Higher gamma → more focus on hard examples.
        reduction: 'mean', 'sum', or 'none'

    Example:
        >>> criterion = FocalLoss(alpha=0.25, gamma=2.0)
        >>> loss = criterion(logits, labels)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Raw logits, shape (N,) or (N, 1)
            targets: Binary labels {0, 1}, shape (N,)
        """
        inputs = inputs.squeeze(1) if inputs.dim() == 2 else inputs
        targets = targets.float()

        # Standard BCE
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Probability of the true class
        pt = torch.exp(-bce_loss)

        # Focal weight: down-weights easy (high-pt) examples
        focal_weight = (1 - pt) ** self.gamma

        # Class balance weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy.
    Simpler alternative to Focal Loss — upweights positive examples by pos_weight.

    Use when: class imbalance is moderate (<10:1) and Focal Loss is over-engineering.

    Args:
        pos_weight: Weight for positive class. Set to neg_count / pos_count.
                    e.g., 96.5/3.5 ≈ 27.6 for ISIC 2024 class ratio.
    """

    def __init__(self, pos_weight: float = 27.6):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.squeeze(1) if inputs.dim() == 2 else inputs
        pos_weight = self.pos_weight.to(inputs.device)
        return F.binary_cross_entropy_with_logits(
            inputs, targets.float(), pos_weight=pos_weight
        )


class LabelSmoothingBCE(nn.Module):
    """
    BCE with label smoothing. Prevents overconfidence.

    In medical imaging, ground truth labels may have annotation noise.
    Label smoothing (epsilon ~0.1) improves calibration.
    """

    def __init__(self, epsilon: float = 0.1, pos_weight: float = 1.0):
        super().__init__()
        self.epsilon = epsilon
        self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.squeeze(1) if inputs.dim() == 2 else inputs
        # Smooth labels: 0 → epsilon, 1 → 1 - epsilon
        smoothed = targets.float() * (1 - self.epsilon) + self.epsilon * 0.5
        pos_weight = torch.tensor([self.pos_weight]).to(inputs.device)
        return F.binary_cross_entropy_with_logits(inputs, smoothed, pos_weight=pos_weight)


def get_loss_function(loss_name: str = 'focal', **kwargs) -> nn.Module:
    """
    Loss function factory.

    Args:
        loss_name: One of 'focal', 'weighted_bce', 'label_smoothing'
        **kwargs: Loss-specific parameters

    Returns:
        Instantiated loss function
    """
    loss_map = {
        'focal': FocalLoss,
        'weighted_bce': WeightedBCELoss,
        'label_smoothing': LabelSmoothingBCE,
        'bce': nn.BCEWithLogitsLoss,
    }
    if loss_name not in loss_map:
        raise ValueError(f'Unknown loss: {loss_name}. Choose from {list(loss_map.keys())}')
    return loss_map[loss_name](**kwargs)


if __name__ == '__main__':
    # Smoke test
    batch_size = 16
    logits = torch.randn(batch_size)
    labels = torch.randint(0, 2, (batch_size,)).float()

    focal = FocalLoss(alpha=0.25, gamma=2.0)
    wbce = WeightedBCELoss(pos_weight=27.6)

    print(f'Focal Loss: {focal(logits, labels).item():.4f}')
    print(f'Weighted BCE: {wbce(logits, labels).item():.4f}')
    print('Loss functions OK.')

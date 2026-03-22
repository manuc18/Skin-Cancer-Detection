"""
src/models/fusion.py
Hybrid fusion model: EfficientNet image encoder + Metadata MLP → Binary classifier.

Architecture motivation:
  - Prior work (Esteva 2017) used images only; real clinical workflow uses both.
  - This project bridges the gap: visual texture + patient context → better calibrated output.
  - Temperature scaling applied post-hoc for clinical probability calibration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional


# ─── Image Encoder ────────────────────────────────────────────────────────────

class ImageEncoder(nn.Module):
    """
    EfficientNet-B4 encoder with ImageNet pretraining.

    Why EfficientNet-B4?
    - Compound scaling balances width, depth, resolution efficiently
    - Strong ImageNet features transfer well to dermoscopy (Esteva 2017)
    - B4 gives better accuracy than B0-B3 with manageable compute
    - Alternative: ViT-Small/16 (specify backbone='vit_small_patch16_224')
    """

    def __init__(self,
                 backbone: str = 'efficientnet_b4',
                 pretrained: bool = True,
                 dropout: float = 0.2):
        super().__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,       # Remove classifier head → returns embeddings
            global_pool='avg'    # Global average pooling
        )

        self.embedding_dim = self.backbone.num_features
        self.dropout = nn.Dropout(dropout)

        print(f'Image encoder: {backbone} | embedding_dim={self.embedding_dim}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor (B, C, H, W), normalized to ImageNet stats
        Returns:
            Image embedding (B, embedding_dim)
        """
        features = self.backbone(x)  # (B, embedding_dim)
        return self.dropout(features)


# ─── Metadata MLP ─────────────────────────────────────────────────────────────

class MetadataMLP(nn.Module):
    """
    Metadata branch: encodes tabular features into a dense embedding.

    Design: BatchNorm → Linear → GELU → Dropout → Linear
    BatchNorm at input handles scale variation across metadata features.
    GELU activation (smoother than ReLU) for tabular data.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 output_dim: int = 64,
                 dropout: float = 0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

        print(f'Metadata MLP: {input_dim} → {hidden_dim} → {output_dim}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Metadata tensor (B, input_dim)
        Returns:
            Metadata embedding (B, output_dim)
        """
        return self.net(x)


# ─── Fusion Classifier ────────────────────────────────────────────────────────

class FusionClassifier(nn.Module):
    """
    Hybrid fusion model: concatenates image and metadata embeddings.

    Architecture:
        Image (224×224) → EfficientNet-B4 → [1792-d embedding]
                                                         ↘
                                                 Concat [1792+64 = 1856-d]
                                                         ↗
        Metadata (~40 features) → MLP → [64-d embedding]
                                                         ↓
                                             FC [1856→512] → ReLU → Dropout
                                                         ↓
                                             FC [512→1] → Sigmoid

    Post-training: Temperature scaling calibrates output probabilities.
    """

    def __init__(self,
                 metadata_dim: int,
                 backbone: str = 'efficientnet_b4',
                 pretrained: bool = True,
                 meta_hidden: int = 128,
                 meta_embedding: int = 64,
                 fusion_hidden: int = 512,
                 img_dropout: float = 0.2,
                 meta_dropout: float = 0.3,
                 fusion_dropout: float = 0.4):
        super().__init__()

        # Branches
        self.image_encoder = ImageEncoder(backbone, pretrained, img_dropout)
        self.metadata_mlp = MetadataMLP(metadata_dim, meta_hidden, meta_embedding, meta_dropout)

        # Fusion head
        fusion_input_dim = self.image_encoder.embedding_dim + meta_embedding
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden // 2, 1)
        )

        print(f'Fusion input dim: {fusion_input_dim}')

    def forward(self,
                images: torch.Tensor,
                metadata: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, C, H, W)
            metadata: (B, metadata_dim)
        Returns:
            Logits (B, 1) — pass through sigmoid for probabilities
        """
        img_feat = self.image_encoder(images)      # (B, img_embed_dim)
        meta_feat = self.metadata_mlp(metadata)     # (B, meta_embed_dim)
        fused = torch.cat([img_feat, meta_feat], dim=1)  # (B, img+meta)
        return self.fusion_head(fused)              # (B, 1)

    def get_probabilities(self,
                           images: torch.Tensor,
                           metadata: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid-activated probabilities (B,)."""
        return torch.sigmoid(self.forward(images, metadata)).squeeze(1)


# ─── Image-Only Model ─────────────────────────────────────────────────────────

class ImageOnlyClassifier(nn.Module):
    """
    DL baseline: EfficientNet with a classification head.
    Used for the 'Deep Learning' implementation pathway.
    """

    def __init__(self,
                 backbone: str = 'efficientnet_b4',
                 pretrained: bool = True,
                 dropout: float = 0.3):
        super().__init__()

        self.encoder = ImageEncoder(backbone, pretrained, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        return self.classifier(features)

    def get_probabilities(self, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(images)).squeeze(1)


# ─── Temperature Scaling (Calibration) ───────────────────────────────────────

class TemperatureScaling(nn.Module):
    """
    Post-hoc calibration via temperature scaling (Guo et al. 2017).

    A single scalar T is learned on a held-out validation set.
    New logits = original_logits / T

    When T > 1: model was overconfident → probabilities become less extreme.
    When T < 1: model was underconfident → probabilities sharpen.

    Clinical motivation: Raw model outputs are NOT calibrated probabilities.
    A model predicting p=0.9 may only be right 60% of the time.
    Temperature scaling fixes this without retraining.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        return logits / self.temperature

    def calibrate(self,
                  logits: torch.Tensor,
                  labels: torch.Tensor,
                  lr: float = 0.01,
                  max_iter: int = 100) -> float:
        """
        Fit temperature on validation logits/labels using NLL loss.

        Args:
            logits: Raw model logits from validation set (N,)
            labels: True binary labels (N,)
            lr: Learning rate for temperature optimization
            max_iter: Maximum optimization steps

        Returns:
            Optimal temperature value
        """
        logits = logits.detach()
        labels = labels.float().detach()

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = F.binary_cross_entropy_with_logits(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)

        optimal_T = self.temperature.item()
        print(f'Temperature calibrated: T={optimal_T:.4f}')
        return optimal_T

    def calibrated_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Returns calibrated probabilities."""
        return torch.sigmoid(self.forward(logits)).squeeze(1)


if __name__ == '__main__':
    # Smoke test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Testing on {device}')

    batch_size = 4
    metadata_dim = 40
    image_size = 224

    images = torch.randn(batch_size, 3, image_size, image_size).to(device)
    metadata = torch.randn(batch_size, metadata_dim).to(device)

    # Test fusion model
    model = FusionClassifier(metadata_dim=metadata_dim, pretrained=False).to(device)
    logits = model(images, metadata)
    probs = model.get_probabilities(images, metadata)
    print(f'Fusion model output: {logits.shape}, probs range: [{probs.min():.3f}, {probs.max():.3f}]')

    # Test calibration
    cal = TemperatureScaling()
    fake_logits = torch.randn(100)
    fake_labels = torch.randint(0, 2, (100,))
    T = cal.calibrate(fake_logits, fake_labels)
    print(f'Temperature: {T:.4f}')

    print('All models OK.')

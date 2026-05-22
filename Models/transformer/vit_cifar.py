import torch.nn as nn
from torchvision.models import VisionTransformer


class ViTTinyCIFAR(nn.Module):
    """
    Vision Transformer reference architecture adapted to CIFAR-10.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = VisionTransformer(
            image_size=32,
            patch_size=4,
            num_layers=6,
            num_heads=3,
            hidden_dim=192,
            mlp_dim=768,
            dropout=0.0,
            attention_dropout=0.0,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)


class ViTSmallCIFAR(nn.Module):
    """
    ViT-Small style model with CIFAR-10 image and patch sizes.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = VisionTransformer(
            image_size=32,
            patch_size=4,
            num_layers=8,
            num_heads=6,
            hidden_dim=384,
            mlp_dim=1536,
            dropout=0.0,
            attention_dropout=0.0,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)

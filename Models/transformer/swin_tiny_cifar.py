import torch.nn as nn
from torchvision.models import swin_t


class SwinTinyCIFAR(nn.Module):
    """
    Swin Transformer Tiny reference architecture with CIFAR-10 classifier head.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = swin_t(weights=None, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

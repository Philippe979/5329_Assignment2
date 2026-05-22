import torch.nn as nn
from torchvision.models import mobilenet_v2


class MobileNetV2CIFAR(nn.Module):
    """
    MobileNetV2 reference architecture with the classifier replaced for CIFAR-10.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = mobilenet_v2(weights=None, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

import torch.nn as nn
from torchvision.models import efficientnet_b0


class EfficientNetB0CIFAR(nn.Module):
    """
    EfficientNet-B0 reference architecture adapted by replacing the classifier head.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

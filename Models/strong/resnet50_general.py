import torch.nn as nn
from torchvision.models import resnet50


class ResNet50General(nn.Module):
    """
    ImageNet-style ResNet-50 adapted by replacing only the classifier head.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

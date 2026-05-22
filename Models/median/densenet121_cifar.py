import torch.nn as nn
from torchvision.models import densenet121


class DenseNet121CIFAR(nn.Module):
    """
    DenseNet-121 reference architecture adapted by replacing the classifier head.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = densenet121(weights=None)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

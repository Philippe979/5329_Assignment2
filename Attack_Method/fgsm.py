import torch
import torch.nn as nn


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def _channel_tensor(values, images: torch.Tensor) -> torch.Tensor:
    return torch.tensor(values, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)


def denormalize_cifar10(images: torch.Tensor) -> torch.Tensor:
    mean = _channel_tensor(CIFAR10_MEAN, images)
    std = _channel_tensor(CIFAR10_STD, images)
    return images * std + mean


def normalize_cifar10(images: torch.Tensor) -> torch.Tensor:
    mean = _channel_tensor(CIFAR10_MEAN, images)
    std = _channel_tensor(CIFAR10_STD, images)
    return (images - mean) / std


def fgsm_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 8 / 255,
    device: str = "cuda",
):
    """
    FGSM attack for CIFAR-10 models trained on normalized inputs.

    The dataloader returns normalized images, but epsilon is defined in raw
    image space. The returned adversarial images are normalized again so they
    can be passed directly into the model.

    Args:
        model: trained model
        images: normalized CIFAR-10 images
        labels: ground truth labels
        epsilon: raw-space perturbation magnitude, usually 8/255
        device: target device

    Returns:
        normalized adversarial images
    """
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    images = images.clone().detach().to(device)
    labels = labels.to(device)

    raw_images = denormalize_cifar10(images).detach()
    raw_images.requires_grad = True

    normalized_images = normalize_cifar10(raw_images)
    outputs = model(normalized_images)
    loss = nn.CrossEntropyLoss()(outputs, labels)

    model.zero_grad()
    loss.backward()

    grad = raw_images.grad.data
    raw_adv_images = raw_images + epsilon * grad.sign()
    raw_adv_images = torch.clamp(raw_adv_images, 0.0, 1.0)
    adv_images = normalize_cifar10(raw_adv_images)

    return adv_images.detach()

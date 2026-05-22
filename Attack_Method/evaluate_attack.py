import torch


def evaluate_fgsm(
    model,
    dataloader,
    epsilon: float,
    attack_fn,
    device: str = "cuda",
):
    """
    Evaluate model under FGSM attack
    """
    model.eval()

    correct = 0
    total = 0

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        adv_images = attack_fn(model, images, labels, epsilon, device)

        outputs = model(adv_images)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total
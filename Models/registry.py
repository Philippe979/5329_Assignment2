from dataclasses import dataclass
from typing import Callable, Dict, List

from .baseline import AlexNetCIFAR, MobileNetV2CIFAR, VGG16BNCIFAR, VGG19BNCIFAR
from .median import DenseNet121CIFAR, ResNet18General, ResNet20, ResNet32
from .mlp import MLPMixerBase, MLPMixerSmall, MLPMixerTiny
from .strong import EfficientNetB0CIFAR, ResNet50General, WRN16_8, WRN28_10
from .transformer import SwinTinyCIFAR, ViTSmallCIFAR, ViTTinyCIFAR


@dataclass(frozen=True)
class ModelConfig:
    name: str
    builder: Callable
    capacity_level: str
    design_type: str
    family: str
    reference: str
    default_batch_size: int
    default_epochs: int
    default_lr: float
    clean_threshold: float


MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "vgg16_bn_cifar": ModelConfig(
        name="vgg16_bn_cifar",
        builder=VGG16BNCIFAR,
        capacity_level="Level 1 - Baseline",
        design_type="CIFAR-friendly",
        family="VGG",
        reference="Simonyan and Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition",
        default_batch_size=128,
        default_epochs=20,
        default_lr=0.05,
        clean_threshold=0.75,
    ),
    "vgg19_bn_cifar": ModelConfig(
        name="vgg19_bn_cifar",
        builder=VGG19BNCIFAR,
        capacity_level="Level 1 - Baseline",
        design_type="CIFAR-friendly",
        family="VGG",
        reference="Simonyan and Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition",
        default_batch_size=128,
        default_epochs=20,
        default_lr=0.05,
        clean_threshold=0.75,
    ),
    "alexnet_cifar": ModelConfig(
        name="alexnet_cifar",
        builder=AlexNetCIFAR,
        capacity_level="Level 1 - Baseline",
        design_type="General-adapted",
        family="AlexNet",
        reference="Krizhevsky et al., ImageNet Classification with Deep Convolutional Neural Networks",
        default_batch_size=128,
        default_epochs=20,
        default_lr=0.05,
        clean_threshold=0.68,
    ),
    "mobilenetv2_cifar": ModelConfig(
        name="mobilenetv2_cifar",
        builder=MobileNetV2CIFAR,
        capacity_level="Level 1 - Baseline",
        design_type="General-adapted",
        family="MobileNetV2",
        reference="Sandler et al., MobileNetV2: Inverted Residuals and Linear Bottlenecks",
        default_batch_size=128,
        default_epochs=20,
        default_lr=0.05,
        clean_threshold=0.72,
    ),
    "mlp_mixer_tiny": ModelConfig(
        name="mlp_mixer_tiny",
        builder=MLPMixerTiny,
        capacity_level="Level 1 - Baseline",
        design_type="MLP-adapted",
        family="MLP-Mixer",
        reference="Tolstikhin et al., MLP-Mixer: An all-MLP Architecture for Vision",
        default_batch_size=128,
        default_epochs=20,
        default_lr=0.03,
        clean_threshold=0.62,
    ),
    "vit_tiny_cifar": ModelConfig(
        name="vit_tiny_cifar",
        builder=ViTTinyCIFAR,
        capacity_level="Level 1 - Baseline",
        design_type="Transformer-adapted",
        family="Vision Transformer",
        reference="Dosovitskiy et al., An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        default_batch_size=128,
        default_epochs=20,
        default_lr=0.03,
        clean_threshold=0.62,
    ),
    "resnet20": ModelConfig(
        name="resnet20",
        builder=ResNet20,
        capacity_level="Level 2 - Median",
        design_type="CIFAR-friendly",
        family="CIFAR ResNet",
        reference="He et al., Deep Residual Learning for Image Recognition",
        default_batch_size=128,
        default_epochs=20,
        default_lr=0.10,
        clean_threshold=0.80,
    ),
    "resnet32": ModelConfig(
        name="resnet32",
        builder=ResNet32,
        capacity_level="Level 2 - Median",
        design_type="CIFAR-friendly",
        family="CIFAR ResNet",
        reference="He et al., Deep Residual Learning for Image Recognition",
        default_batch_size=128,
        default_epochs=20,
        default_lr=0.10,
        clean_threshold=0.80,
    ),
    "resnet18_general": ModelConfig(
        name="resnet18_general",
        builder=ResNet18General,
        capacity_level="Level 2 - Median",
        design_type="General-adapted",
        family="ResNet",
        reference="He et al., Deep Residual Learning for Image Recognition",
        default_batch_size=128,
        default_epochs=20,
        default_lr=0.05,
        clean_threshold=0.76,
    ),
    "densenet121_cifar": ModelConfig(
        name="densenet121_cifar",
        builder=DenseNet121CIFAR,
        capacity_level="Level 2 - Median",
        design_type="General-adapted",
        family="DenseNet",
        reference="Huang et al., Densely Connected Convolutional Networks",
        default_batch_size=64,
        default_epochs=20,
        default_lr=0.05,
        clean_threshold=0.78,
    ),
    "mlp_mixer_small": ModelConfig(
        name="mlp_mixer_small",
        builder=MLPMixerSmall,
        capacity_level="Level 2 - Median",
        design_type="MLP-adapted",
        family="MLP-Mixer",
        reference="Tolstikhin et al., MLP-Mixer: An all-MLP Architecture for Vision",
        default_batch_size=128,
        default_epochs=20,
        default_lr=0.03,
        clean_threshold=0.68,
    ),
    "vit_small_cifar": ModelConfig(
        name="vit_small_cifar",
        builder=ViTSmallCIFAR,
        capacity_level="Level 2 - Median",
        design_type="Transformer-adapted",
        family="Vision Transformer",
        reference="Dosovitskiy et al., An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        default_batch_size=64,
        default_epochs=20,
        default_lr=0.03,
        clean_threshold=0.68,
    ),
    "wrn_16_8": ModelConfig(
        name="wrn_16_8",
        builder=WRN16_8,
        capacity_level="Level 3 - Strong",
        design_type="CIFAR-friendly",
        family="WideResNet",
        reference="Zagoruyko and Komodakis, Wide Residual Networks",
        default_batch_size=64,
        default_epochs=20,
        default_lr=0.10,
        clean_threshold=0.82,
    ),
    "wrn_28_10": ModelConfig(
        name="wrn_28_10",
        builder=WRN28_10,
        capacity_level="Level 3 - Strong",
        design_type="CIFAR-friendly",
        family="WideResNet",
        reference="Zagoruyko and Komodakis, Wide Residual Networks",
        default_batch_size=32,
        default_epochs=20,
        default_lr=0.10,
        clean_threshold=0.82,
    ),
    "resnet50_general": ModelConfig(
        name="resnet50_general",
        builder=ResNet50General,
        capacity_level="Level 3 - Strong",
        design_type="General-adapted",
        family="ResNet",
        reference="He et al., Deep Residual Learning for Image Recognition",
        default_batch_size=64,
        default_epochs=20,
        default_lr=0.05,
        clean_threshold=0.78,
    ),
    "efficientnet_b0_cifar": ModelConfig(
        name="efficientnet_b0_cifar",
        builder=EfficientNetB0CIFAR,
        capacity_level="Level 3 - Strong",
        design_type="General-adapted",
        family="EfficientNet",
        reference="Tan and Le, EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
        default_batch_size=64,
        default_epochs=20,
        default_lr=0.05,
        clean_threshold=0.78,
    ),
    "mlp_mixer_base": ModelConfig(
        name="mlp_mixer_base",
        builder=MLPMixerBase,
        capacity_level="Level 3 - Strong",
        design_type="MLP-adapted",
        family="MLP-Mixer",
        reference="Tolstikhin et al., MLP-Mixer: An all-MLP Architecture for Vision",
        default_batch_size=64,
        default_epochs=20,
        default_lr=0.03,
        clean_threshold=0.70,
    ),
    "swin_tiny_cifar": ModelConfig(
        name="swin_tiny_cifar",
        builder=SwinTinyCIFAR,
        capacity_level="Level 3 - Strong",
        design_type="Transformer-adapted",
        family="Swin Transformer",
        reference="Liu et al., Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
        default_batch_size=32,
        default_epochs=20,
        default_lr=0.03,
        clean_threshold=0.70,
    ),
}


def list_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def get_model_config(model_name: str) -> ModelConfig:
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError as error:
        available = ", ".join(list_models())
        raise KeyError(f"Unknown model '{model_name}'. Available models: {available}") from error


def build_model(model_name: str, num_classes: int = 10):
    config = get_model_config(model_name)
    return config.builder(num_classes=num_classes)


def models_by_group(group: str) -> List[str]:
    if group == "all":
        return list_models()
    if group == "cifar_friendly":
        return [
            name for name, config in MODEL_REGISTRY.items()
            if config.design_type == "CIFAR-friendly"
        ]
    if group == "general_adapted":
        return [
            name for name, config in MODEL_REGISTRY.items()
            if config.design_type == "General-adapted"
        ]
    if group == "mlp_adapted":
        return [
            name for name, config in MODEL_REGISTRY.items()
            if config.design_type == "MLP-adapted"
        ]
    if group == "transformer_adapted":
        return [
            name for name, config in MODEL_REGISTRY.items()
            if config.design_type == "Transformer-adapted"
        ]
    if group in {"baseline", "median", "strong"}:
        level_token = {
            "baseline": "Level 1",
            "median": "Level 2",
            "strong": "Level 3",
        }[group]
        return [
            name for name, config in MODEL_REGISTRY.items()
            if config.capacity_level.startswith(level_token)
        ]
    raise ValueError(
        "group must be one of: all, cifar_friendly, general_adapted, "
        "mlp_adapted, transformer_adapted, baseline, median, strong"
    )

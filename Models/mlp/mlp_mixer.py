import torch
import torch.nn as nn


class MixerBlock(nn.Module):
    def __init__(
        self,
        num_patches: int,
        hidden_dim: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mixing = nn.Sequential(
            nn.Linear(num_patches, tokens_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tokens_mlp_dim, num_patches),
            nn.Dropout(dropout),
        )
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mixing = nn.Sequential(
            nn.Linear(hidden_dim, channels_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels_mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.token_norm(x)
        y = y.transpose(1, 2)
        y = self.token_mixing(y)
        y = y.transpose(1, 2)
        x = x + y
        return x + self.channel_mixing(self.channel_norm(x))


class MLPMixer(nn.Module):
    """
    MLP-Mixer reference architecture adapted to CIFAR-10 resolution.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        hidden_dim: int = 256,
        num_blocks: int = 8,
        tokens_mlp_dim: int = 128,
        channels_mlp_dim: int = 512,
        num_classes: int = 10,
        dropout: float = 0.0,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            3,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.blocks = nn.Sequential(
            *[
                MixerBlock(
                    num_patches=self.num_patches,
                    hidden_dim=hidden_dim,
                    tokens_mlp_dim=tokens_mlp_dim,
                    channels_mlp_dim=channels_mlp_dim,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


def MLPMixerTiny(num_classes: int = 10) -> MLPMixer:
    return MLPMixer(
        hidden_dim=128,
        num_blocks=4,
        tokens_mlp_dim=64,
        channels_mlp_dim=256,
        num_classes=num_classes,
    )


def MLPMixerSmall(num_classes: int = 10) -> MLPMixer:
    return MLPMixer(
        hidden_dim=256,
        num_blocks=8,
        tokens_mlp_dim=128,
        channels_mlp_dim=512,
        num_classes=num_classes,
    )


def MLPMixerBase(num_classes: int = 10) -> MLPMixer:
    return MLPMixer(
        hidden_dim=384,
        num_blocks=12,
        tokens_mlp_dim=192,
        channels_mlp_dim=768,
        num_classes=num_classes,
    )

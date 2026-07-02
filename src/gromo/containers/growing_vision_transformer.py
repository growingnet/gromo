from typing import Any, Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor

from gromo.containers.growing_block import GrowingBlock
from gromo.containers.growing_transformer import GrowingTransformerBlock
from gromo.containers.sequential_growing_container import SequentialGrowingModel
from gromo.utils.utils import compute_tensor_stats


def check_patch_grid(
    image_height: int,
    image_width: int,
    patch_size: int | tuple[int, int],
) -> tuple[tuple[int, int], int]:
    """Validate the patch size and return the patch grid."""
    if isinstance(patch_size, int):
        patch_height = patch_width = patch_size
    else:
        patch_height, patch_width = patch_size

    num_patches_height, remainder_height = divmod(image_height, patch_height)
    num_patches_width, remainder_width = divmod(image_width, patch_width)

    assert remainder_height == 0, "`image_height` must be divisible by `patch_size`"
    assert remainder_width == 0, "`image_width` must be divisible by `patch_size`"

    return (patch_height, patch_width), num_patches_height * num_patches_width


def _block_growth_targets(
    blocks: nn.ModuleList,
) -> list[GrowingBlock]:
    """Return the growable MLP blocks used to update each transformer block."""
    return [block.mlp for block in blocks]


class Tokenizer(nn.Module):
    """CCT convolutional tokenizer."""

    def __init__(
        self,
        kernel_size: int,
        stride: int,
        padding: int,
        pooling_kernel_size: int = 3,
        pooling_stride: int = 2,
        pooling_padding: int = 1,
        n_conv_layers: int = 1,
        n_input_channels: int = 3,
        n_output_channels: int = 64,
        in_planes: int = 64,
        activation: type[nn.Module] | None = None,
        max_pool: bool = True,
        conv_bias: bool = False,
    ) -> None:
        super().__init__()
        n_filter_list = (
            [n_input_channels]
            + [in_planes for _ in range(n_conv_layers - 1)]
            + [n_output_channels]
        )
        self.conv_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        n_filter_list[i],
                        n_filter_list[i + 1],
                        kernel_size=(kernel_size, kernel_size),
                        stride=(stride, stride),
                        padding=(padding, padding),
                        bias=conv_bias,
                    ),
                    nn.Identity() if activation is None else activation(),
                    (
                        nn.MaxPool2d(
                            kernel_size=pooling_kernel_size,
                            stride=pooling_stride,
                            padding=pooling_padding,
                        )
                        if max_pool
                        else nn.Identity()
                    ),
                )
                for i in range(n_conv_layers)
            ]
        )
        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def sequence_length(
        self,
        n_channels: int = 3,
        height: int = 224,
        width: int = 224,
    ) -> int:
        """Return the token sequence length produced for an image size."""
        device = next(self.parameters()).device
        return self.forward(
            torch.zeros((1, n_channels, height, width), device=device)
        ).shape[1]

    def forward(self, x: Tensor) -> Tensor:
        """Convert images to token sequences."""
        return self.flattener(self.conv_layers(x)).transpose(-2, -1)

    @staticmethod
    def init_weight(m: nn.Module) -> None:
        """Initialize convolutional tokenizer weights."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class TextTokenizer(nn.Module):
    """Text tokenizer kept for parity with Compact-Transformers utilities."""

    def __init__(
        self,
        kernel_size: int,
        stride: int,
        padding: int,
        pooling_kernel_size: int = 3,
        pooling_stride: int = 2,
        pooling_padding: int = 1,
        embedding_dim: int = 300,
        n_output_channels: int = 128,
        activation: type[nn.Module] | None = None,
        max_pool: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _ = args, kwargs
        self.max_pool = max_pool
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                1,
                n_output_channels,
                kernel_size=(kernel_size, embedding_dim),
                stride=(stride, 1),
                padding=(padding, 0),
                bias=False,
            ),
            nn.Identity() if activation is None else activation(),
            (
                nn.MaxPool2d(
                    kernel_size=(pooling_kernel_size, 1),
                    stride=(pooling_stride, 1),
                    padding=(pooling_padding, 0),
                )
                if max_pool
                else nn.Identity()
            ),
        )
        self.apply(self.init_weight)

    def seq_len(self, seq_len: int = 32, embed_dim: int = 300) -> int:
        """Return output sequence length for text inputs."""
        return self.forward(torch.zeros((1, seq_len, embed_dim)))[0].shape[1]

    def forward_mask(self, mask: Tensor) -> Tensor:
        """Propagate a token-validity mask through the convolution/pooling."""
        new_mask = mask.unsqueeze(1).float()
        cnn_weight = torch.ones(
            (1, 1, self.conv_layers[0].kernel_size[0]),
            device=mask.device,
            dtype=torch.float,
        )
        new_mask = functional.conv1d(
            new_mask,
            cnn_weight,
            None,
            self.conv_layers[0].stride[0],
            self.conv_layers[0].padding[0],
            1,
            1,
        )
        if self.max_pool:
            new_mask = functional.max_pool1d(
                new_mask,
                self.conv_layers[2].kernel_size[0],
                self.conv_layers[2].stride[0],
                self.conv_layers[2].padding[0],
                1,
                False,
                False,
            )
        return new_mask.squeeze(1) > 0

    def forward(self, x: Tensor, mask: Tensor | None = None):
        """Tokenize text embeddings and propagate an optional mask."""
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.transpose(1, 3).squeeze(1)
        if mask is not None:
            mask = self.forward_mask(mask)
            x = x * mask.unsqueeze(-1).float()
        return x, mask

    @staticmethod
    def init_weight(m: nn.Module) -> None:
        """Initialize text tokenizer convolution weights."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class Embedder(nn.Module):
    """Embedding helper kept for parity with Compact-Transformers utilities."""

    def __init__(
        self,
        word_embedding_dim: int = 300,
        vocab_size: int = 100000,
        padding_idx: int = 1,
        pretrained_weight: Tensor | None = None,
        embed_freeze: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _ = args, kwargs
        self.embeddings = (
            nn.Embedding.from_pretrained(pretrained_weight, freeze=embed_freeze)
            if pretrained_weight is not None
            else nn.Embedding(vocab_size, word_embedding_dim, padding_idx=padding_idx)
        )
        self.embeddings.weight.requires_grad = not embed_freeze

    def forward_mask(self, mask: Tensor) -> Tensor:
        """Convert an embedding mask to token-validity mask."""
        bsz, seq_len = mask.shape
        return mask.view(bsz, seq_len, 1).sum(-1) > 0

    def forward(self, x: Tensor, mask: Tensor | None = None):
        """Embed tokens and apply an optional validity mask."""
        embed = self.embeddings(x)
        if mask is not None:
            embed = embed * self.forward_mask(mask).unsqueeze(-1).float()
        return embed, mask

    @staticmethod
    def init_weight(m: nn.Module) -> None:
        """Initialize embedding helper weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif hasattr(m, "weight"):
            nn.init.normal_(m.weight)


class GrowingTransformerClassifier(SequentialGrowingModel):
    """CCT TransformerClassifier with growable feed-forward transformer blocks."""

    def __init__(
        self,
        seq_pool: bool = True,
        embedding_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_classes: int = 1000,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        stochastic_depth: float = 0.1,
        positional_embedding: Literal["sine", "learnable", "none"] = "learnable",
        sequence_length: int | None = None,
        device: torch.device | str | None = None,
        name: str = "GrowingTransformerClassifier",
    ) -> None:
        positional_embedding = (
            positional_embedding
            if positional_embedding in {"sine", "learnable", "none"}
            else "sine"
        )
        if sequence_length is None and positional_embedding != "none":
            raise ValueError(
                "`sequence_length` is required when positional embeddings are enabled."
            )
        in_features = embedding_dim * (sequence_length or 1)
        super().__init__(
            in_features=in_features,
            out_features=num_classes,
            device=device,
        )
        self.name = name

        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.d_ff = dim_feedforward
        self.num_classes = num_classes
        self.positional_embedding = positional_embedding
        self.num_tokens = 0

        positional_length = sequence_length
        self.class_emb: nn.Parameter | None = None
        if not seq_pool:
            if positional_length is not None:
                positional_length += 1
            self.class_emb = nn.Parameter(
                torch.zeros(1, 1, self.embedding_dim, device=self.device),
                requires_grad=True,
            )
            self.num_tokens = 1
        else:
            self.attention_pool = nn.Linear(
                self.embedding_dim,
                1,
                device=self.device,
            )

        self.positional_emb: nn.Parameter | None
        if positional_embedding == "learnable":
            assert positional_length is not None
            self.positional_emb = nn.Parameter(
                torch.zeros(1, positional_length, embedding_dim, device=self.device),
                requires_grad=True,
            )
            nn.init.trunc_normal_(self.positional_emb, std=0.2)
        elif positional_embedding == "sine":
            assert positional_length is not None
            self.positional_emb = nn.Parameter(
                self.sinusoidal_embedding(positional_length, embedding_dim).to(
                    self.device
                ),
                requires_grad=False,
            )
        else:
            self.positional_emb = None

        self.dropout = nn.Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks: nn.ModuleList[GrowingTransformerBlock] = nn.ModuleList(
            [
                GrowingTransformerBlock(
                    d_model=embedding_dim,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path_rate=dpr[i],
                    device=self.device,
                )
                for i in range(num_layers)
            ]
        )
        self._growable_layers: list[GrowingBlock] = _block_growth_targets(self.blocks)
        self.norm = nn.LayerNorm(embedding_dim, device=self.device)
        self.fc = nn.Linear(embedding_dim, num_classes, device=self.device)
        self.set_growing_layers(scheduling_method="all")
        self.apply(self.init_weight)
        if self.positional_emb is not None and positional_embedding == "learnable":
            nn.init.trunc_normal_(self.positional_emb, std=0.2)

    def set_growing_layers(
        self,
        scheduling_method: str = "all",
        index: int | None = None,
    ) -> None:
        """Select all blocks, the next block, or one specific block for growth."""
        self._growable_layers = _block_growth_targets(self.blocks)
        super().set_growing_layers(
            scheduling_method=scheduling_method,
            index=index,
        )

    def _pad_to_sequence_length(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Pad variable token sequences when no positional embedding fixes length."""
        if self.sequence_length is None or x.size(1) >= self.sequence_length:
            return x, attention_mask
        padding_length = self.sequence_length - x.size(1)
        x = functional.pad(x, (0, 0, 0, padding_length), mode="constant", value=0)
        if attention_mask is not None:
            attention_mask = functional.pad(
                attention_mask,
                (0, padding_length),
                mode="constant",
                value=False,
            )
        return x, attention_mask

    def _prepare_tokens(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Add optional class/position tokens before transformer blocks."""
        if self.positional_emb is None:
            x, attention_mask = self._pad_to_sequence_length(x, attention_mask)

        if self.class_emb is not None:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            if attention_mask is not None:
                cls_mask = torch.ones(
                    (attention_mask.shape[0], 1),
                    device=attention_mask.device,
                    dtype=torch.bool,
                )
                attention_mask = torch.cat((cls_mask, attention_mask.bool()), dim=1)

        if self.positional_emb is not None:
            if x.size(1) != self.positional_emb.size(1):
                raise ValueError(
                    "Input sequence length does not match positional embeddings."
                )
            x = x + self.positional_emb

        return self.dropout(x), attention_mask

    def _pool_tokens(self, x: Tensor) -> Tensor:
        """Pool encoded tokens using CCT sequence pooling or class token."""
        if self.seq_pool:
            weights = functional.softmax(self.attention_pool(x), dim=1)
            return torch.matmul(weights.transpose(-1, -2), x).squeeze(-2)
        return x[:, 0]

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass for token sequences."""
        x, attention_mask = self._prepare_tokens(x, mask)
        for block in self.blocks:
            x = block(
                x,
                mask=attention_mask,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
        x = self.norm(x)
        return self.fc(self._pool_tokens(x))

    def extended_forward(
        self,
        x: Tensor,
        mask: dict | None = None,
        attention_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass including the currently proposed GroMo extensions."""
        x, attention_mask = self._prepare_tokens(x, attention_mask)
        for block in self.blocks:
            x = block.extended_forward(
                x,
                mask=mask,
                attention_mask=attention_mask,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
        x = self.norm(x)
        return self.fc(self._pool_tokens(x))

    def update_information(self) -> Dict[str, Any]:
        """Collect growth information for active transformer blocks."""
        return {
            "blocks": super().update_information(),
            "d_model": self.embedding_dim,
            "d_ff": self.d_ff,
            "num_blocks": self.num_layers,
        }

    @property
    def first_order_improvement(self) -> torch.Tensor:
        """Return the improvement of the currently selected block."""
        if self.currently_updated_layer_index is None:
            return torch.stack(
                [layer.first_order_improvement for layer in self._growing_layers]
            ).max()
        return self.currently_updated_layer.first_order_improvement

    def weights_statistics(self) -> Dict[str, Any]:
        """Collect statistics for embeddings, blocks, and classifier head."""
        statistics: Dict[str, Any] = {
            "blocks": {
                i: block.weights_statistics() for i, block in enumerate(self.blocks)
            },
            "norm": {
                "weight": compute_tensor_stats(self.norm.weight),
                "bias": compute_tensor_stats(self.norm.bias),
            },
            "fc": {
                "weight": compute_tensor_stats(self.fc.weight),
            },
        }
        if self.seq_pool:
            statistics["attention_pool"] = {
                "weight": compute_tensor_stats(self.attention_pool.weight),
            }
            if self.attention_pool.bias is not None:
                statistics["attention_pool"]["bias"] = compute_tensor_stats(
                    self.attention_pool.bias
                )
        if self.class_emb is not None:
            statistics["class_emb"] = compute_tensor_stats(self.class_emb)
        if self.positional_emb is not None:
            statistics["positional_emb"] = compute_tensor_stats(self.positional_emb)
        if self.fc.bias is not None:
            statistics["fc"]["bias"] = compute_tensor_stats(self.fc.bias)
        return statistics

    @staticmethod
    def init_weight(m: nn.Module) -> None:
        """Initialize classifier linear and normalization weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels: int, dim: int) -> Tensor:
        """Create CCT sinusoidal positional embeddings."""
        pe = torch.FloatTensor(
            [
                [p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                for p in range(n_channels)
            ]
        )
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


class GrowingTransformer(SequentialGrowingModel):
    """Image transformer base using a tokenizer and growable transformer blocks."""

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        embedding_dim: int = 768,
        n_input_channels: int = 3,
        n_conv_layers: int = 1,
        kernel_size: int = 7,
        stride: int = 2,
        padding: int = 3,
        pooling_kernel_size: int = 3,
        pooling_stride: int = 2,
        pooling_padding: int = 1,
        dropout: float = 0.0,
        attention_dropout: float = 0.1,
        stochastic_depth: float = 0.1,
        num_layers: int = 14,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_classes: int = 1000,
        positional_embedding: Literal["sine", "learnable", "none"] = "learnable",
        seq_pool: bool = True,
        activation: type[nn.Module] | None = nn.ReLU,
        max_pool: bool = True,
        conv_bias: bool = False,
        device: torch.device | str | None = None,
        name: str = "GrowingTransformer",
    ) -> None:
        if isinstance(img_size, int):
            image_height = image_width = img_size
        else:
            image_height, image_width = img_size
        super().__init__(
            in_features=n_input_channels * image_height * image_width,
            out_features=num_classes,
            device=device,
        )
        self.name = name
        self.input_shape = (n_input_channels, image_height, image_width)
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.seq_pool = seq_pool
        self.d_model = embedding_dim
        self.d_ff = int(embedding_dim * mlp_ratio)
        self.num_blocks = num_layers

        self.tokenizer = Tokenizer(
            n_input_channels=n_input_channels,
            n_output_channels=embedding_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride,
            pooling_padding=pooling_padding,
            max_pool=max_pool,
            activation=activation,
            n_conv_layers=n_conv_layers,
            conv_bias=conv_bias,
        ).to(self.device)
        sequence_length = self.tokenizer.sequence_length(
            n_channels=n_input_channels,
            height=image_height,
            width=image_width,
        )
        self.classifier = GrowingTransformerClassifier(
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
            seq_pool=seq_pool,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding,
            device=self.device,
        )
        self._growable_layers: list[GrowingBlock] = _block_growth_targets(
            self.classifier.blocks
        )
        self.set_growing_layers(scheduling_method="all")

    def set_growing_layers(
        self,
        scheduling_method: str = "all",
        index: int | None = None,
    ) -> None:
        """Select all blocks, the next block, or one specific block for growth."""
        self._growable_layers = _block_growth_targets(self.classifier.blocks)
        super().set_growing_layers(
            scheduling_method=scheduling_method,
            index=index,
        )

    @property
    def blocks(self) -> nn.ModuleList:
        """Access the growable transformer blocks."""
        return self.classifier.blocks

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass for image inputs."""
        if attention_mask is not None:
            if mask is not None:
                raise ValueError("Use either `mask` or `attention_mask`, not both.")
            mask = attention_mask
        x = self.tokenizer(x)
        return self.classifier(
            x,
            mask=mask,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )

    def extended_forward(
        self,
        x: Tensor,
        mask: dict | None = None,
        attention_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass including the currently proposed GroMo extensions."""
        x = self.tokenizer(x)
        return self.classifier.extended_forward(
            x,
            mask=mask,
            attention_mask=attention_mask,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )

    def update_information(self) -> Dict[str, Any]:
        """Collect growth information for active transformer blocks."""
        blocks_information = super().update_information()
        return {
            "blocks": blocks_information,
            "classifier": {
                "blocks": blocks_information,
                "d_model": self.classifier.embedding_dim,
                "d_ff": self.classifier.d_ff,
                "num_blocks": self.classifier.num_layers,
            },
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "num_blocks": self.num_blocks,
        }

    @property
    def first_order_improvement(self) -> torch.Tensor:
        """Return the improvement of the currently selected block."""
        if self.currently_updated_layer_index is None:
            return torch.stack(
                [layer.first_order_improvement for layer in self._growing_layers]
            ).max()
        return self.currently_updated_layer.first_order_improvement

    def weights_statistics(self) -> Dict[str, Any]:
        """Collect statistics for tokenizer, blocks, and classifier head."""
        classifier_stats = self.classifier.weights_statistics()
        statistics: Dict[str, Any] = {
            "tokenizer": {},
            "classifier": classifier_stats,
            "blocks": classifier_stats["blocks"],
        }
        for i, layer in enumerate(self.tokenizer.conv_layers):
            conv = layer[0]
            statistics["tokenizer"][i] = {
                "weight": compute_tensor_stats(conv.weight),
            }
            if conv.bias is not None:
                statistics["tokenizer"][i]["bias"] = compute_tensor_stats(conv.bias)
        return statistics


class GrowingCCT(GrowingTransformer):
    """Compact Convolutional Transformer using growable transformer blocks."""

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        embedding_dim: int = 768,
        n_input_channels: int = 3,
        n_conv_layers: int = 1,
        kernel_size: int = 7,
        stride: int = 2,
        padding: int = 3,
        pooling_kernel_size: int = 3,
        pooling_stride: int = 2,
        pooling_padding: int = 1,
        dropout: float = 0.0,
        attention_dropout: float = 0.1,
        stochastic_depth: float = 0.1,
        num_layers: int = 14,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_classes: int = 1000,
        positional_embedding: Literal["sine", "learnable", "none"] = "learnable",
        seq_pool: bool = True,
        device: torch.device | str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        _ = args
        super().__init__(
            img_size=img_size,
            embedding_dim=embedding_dim,
            n_input_channels=n_input_channels,
            n_conv_layers=n_conv_layers,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride,
            pooling_padding=pooling_padding,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding,
            seq_pool=seq_pool,
            activation=nn.ReLU,
            max_pool=True,
            conv_bias=False,
            device=device,
            name="GrowingCCT",
            **kwargs,
        )


class GrowingViTLite(GrowingTransformer):
    """ViTLite using growable transformer blocks.

    This mirrors Compact-Transformers' ViTLite: one convolutional patch tokenizer,
    no activation or pooling in the tokenizer, and class-token pooling in the
    transformer classifier.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        embedding_dim: int = 768,
        n_input_channels: int = 3,
        kernel_size: int = 16,
        dropout: float = 0.0,
        attention_dropout: float = 0.1,
        stochastic_depth: float = 0.1,
        num_layers: int = 14,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_classes: int = 1000,
        positional_embedding: Literal["sine", "learnable", "none"] = "learnable",
        device: torch.device | str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        _ = args
        if isinstance(img_size, int):
            image_height = image_width = img_size
        else:
            image_height, image_width = img_size
        check_patch_grid(image_height, image_width, kernel_size)

        super().__init__(
            img_size=img_size,
            embedding_dim=embedding_dim,
            n_input_channels=n_input_channels,
            n_conv_layers=1,
            kernel_size=kernel_size,
            stride=kernel_size,
            padding=0,
            pooling_kernel_size=3,
            pooling_stride=2,
            pooling_padding=1,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding,
            seq_pool=False,
            activation=None,
            max_pool=False,
            conv_bias=True,
            device=device,
            name="GrowingViTLite",
            **kwargs,
        )


class GrowingCVT(GrowingTransformer):
    """CVT using growable transformer blocks.

    This mirrors Compact-Transformers' CVT: ViTLite's patch tokenizer with
    sequence pooling in the transformer classifier.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        embedding_dim: int = 768,
        n_input_channels: int = 3,
        kernel_size: int = 16,
        dropout: float = 0.0,
        attention_dropout: float = 0.1,
        stochastic_depth: float = 0.1,
        num_layers: int = 14,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_classes: int = 1000,
        positional_embedding: Literal["sine", "learnable", "none"] = "learnable",
        device: torch.device | str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        _ = args
        if isinstance(img_size, int):
            image_height = image_width = img_size
        else:
            image_height, image_width = img_size
        check_patch_grid(image_height, image_width, kernel_size)

        super().__init__(
            img_size=img_size,
            embedding_dim=embedding_dim,
            n_input_channels=n_input_channels,
            n_conv_layers=1,
            kernel_size=kernel_size,
            stride=kernel_size,
            padding=0,
            pooling_kernel_size=3,
            pooling_stride=2,
            pooling_padding=1,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding,
            seq_pool=True,
            activation=None,
            max_pool=False,
            conv_bias=True,
            device=device,
            name="GrowingCVT",
            **kwargs,
        )


class GrowingTextViTLite(SequentialGrowingModel):
    """TextViTLite using growable transformer blocks and masked attention."""

    def __init__(
        self,
        seq_len: int = 64,
        word_embedding_dim: int = 300,
        embedding_dim: int = 300,
        patch_size: int = 2,
        dropout: float = 0.0,
        attention_dropout: float = 0.1,
        stochastic_depth: float = 0.1,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_classes: int = 1000,
        positional_embedding: Literal["sine", "learnable", "none"] = "sine",
        vocab_size: int = 100000,
        padding_idx: int = 1,
        pretrained_weight: Tensor | None = None,
        embed_freeze: bool = False,
        device: torch.device | str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        _ = args, kwargs
        if seq_len % patch_size != 0:
            raise ValueError(
                f"sequence length ({seq_len}) has to be divisible by patch size "
                f"({patch_size})"
            )
        super().__init__(
            in_features=seq_len,
            out_features=num_classes,
            device=device,
        )
        self.name = "GrowingTextViTLite"
        self.seq_len = seq_len
        self.word_embedding_dim = word_embedding_dim
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes

        self.embedder = Embedder(
            word_embedding_dim=word_embedding_dim,
            vocab_size=vocab_size,
            padding_idx=padding_idx,
            pretrained_weight=pretrained_weight,
            embed_freeze=embed_freeze,
        ).to(self.device)
        self.tokenizer = TextTokenizer(
            n_output_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            max_pool=False,
            activation=None,
            embedding_dim=word_embedding_dim,
        ).to(self.device)
        sequence_length = self.tokenizer.seq_len(
            seq_len=seq_len,
            embed_dim=word_embedding_dim,
        )
        self.classifier = GrowingTransformerClassifier(
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
            seq_pool=False,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding,
            device=self.device,
        )
        self._growable_layers: list[GrowingBlock] = _block_growth_targets(
            self.classifier.blocks
        )
        self.set_growing_layers(scheduling_method="all")

    def set_growing_layers(
        self,
        scheduling_method: str = "all",
        index: int | None = None,
    ) -> None:
        """Select all blocks, the next block, or one specific block for growth."""
        self._growable_layers = _block_growth_targets(self.classifier.blocks)
        super().set_growing_layers(
            scheduling_method=scheduling_method,
            index=index,
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass for token id sequences."""
        x, mask = self.embedder(x, mask=mask)
        x, mask = self.tokenizer(x, mask=mask)
        return self.classifier(
            x,
            mask=mask,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )

    def extended_forward(
        self,
        x: Tensor,
        mask: dict | None = None,
        attention_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass including the currently proposed GroMo extensions."""
        x, token_mask = self.embedder(x, mask=attention_mask)
        x, token_mask = self.tokenizer(x, mask=token_mask)
        return self.classifier.extended_forward(
            x,
            mask=mask,
            attention_mask=token_mask,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )

    def update_information(self) -> Dict[str, Any]:
        """Collect growth information for active text transformer blocks."""
        blocks_information = super().update_information()
        return {
            "blocks": blocks_information,
            "classifier": {
                "blocks": blocks_information,
                "d_model": self.classifier.embedding_dim,
                "d_ff": self.classifier.d_ff,
                "num_blocks": self.classifier.num_layers,
            },
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
        }

    @property
    def first_order_improvement(self) -> torch.Tensor:
        """Return the improvement of the currently selected text transformer block."""
        if self.currently_updated_layer_index is None:
            return torch.stack(
                [layer.first_order_improvement for layer in self._growing_layers]
            ).max()
        return self.currently_updated_layer.first_order_improvement

    def weights_statistics(self) -> Dict[str, Any]:
        """Collect statistics for embedder, tokenizer, and classifier head."""
        conv = self.tokenizer.conv_layers[0]
        statistics: Dict[str, Any] = {
            "embedder": {
                "weight": compute_tensor_stats(self.embedder.embeddings.weight),
            },
            "tokenizer": {
                "weight": compute_tensor_stats(conv.weight),
            },
            "classifier": self.classifier.weights_statistics(),
        }
        if conv.bias is not None:
            statistics["tokenizer"]["bias"] = compute_tensor_stats(conv.bias)
        return statistics


if __name__ == "__main__":
    input_shape = (3, 32, 32)
    model = GrowingTransformer(
        img_size=input_shape[1:],
        embedding_dim=32,
        n_input_channels=input_shape[0],
        kernel_size=4,
        stride=4,
        padding=0,
        activation=None,
        max_pool=False,
        conv_bias=True,
        num_heads=4,
        mlp_ratio=2.0,
        num_layers=2,
        num_classes=10,
        seq_pool=False,
    )

    x = torch.rand((2, *input_shape))
    y = model(x)
    print(model)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

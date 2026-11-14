"""
Basic TSPulse-inspired timeseries embedding model.
Simplified implementation based on IBM Granite TSPulse architecture.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TSPulseConfig:
    """Configuration for TSPulse timeseries embedding model."""
    
    # Input configuration
    context_length: int = 512  # Length of input timeseries
    num_input_channels: int = 1  # Number of input variates (1 for univariate)
    
    # Patching configuration
    patch_length: int = 8  # Length of each patch
    patch_stride: int = 8  # Stride for patching (=patch_length for non-overlapping)
    
    # Model architecture
    d_model: int = 128  # Hidden dimension
    num_layers: int = 3  # Number of encoder layers
    expansion_factor: int = 2  # MLP expansion factor
    dropout: float = 0.2
    
    # Normalization
    norm_eps: float = 1e-5
    scaling: str = "revin"  # "revin", "mean", "std", or None
    
    # Attention
    gated_attn: bool = True
    attention_activation: str = "softmax"  # "softmax" or "sigmoid"
    
    # Mode
    mode: str = "common_channel"  # "common_channel" or "mix_channel"
    
    @property
    def num_patches(self) -> int:
        """Calculate number of patches from context length and patch configuration."""
        return (self.context_length - self.patch_length) // self.patch_stride + 1


class RevIN(nn.Module):
    """Reversible Instance Normalization for time series."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x: torch.Tensor, mode: str = "norm") -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, num_features)
            mode: "norm" for normalization, "denorm" for denormalization
        
        Returns:
            Normalized tensor and statistics (mean, std) if mode="norm"
        """
        if mode == "norm":
            # Calculate statistics along sequence dimension
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True) + self.eps
            x_norm = (x - mean) / std
            
            if self.affine:
                x_norm = x_norm * self.affine_weight + self.affine_bias
            
            return x_norm, mean, std
        elif mode == "denorm":
            raise NotImplementedError("Denormalization requires mean and std from forward pass")
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def inverse(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Reverse the normalization."""
        if self.affine:
            x = (x - self.affine_bias) / self.affine_weight
        return x * std + mean


class Patching(nn.Module):
    """Patch time series into fixed-length segments."""
    
    def __init__(self, patch_length: int, patch_stride: int):
        super().__init__()
        self.patch_length = patch_length
        self.patch_stride = patch_stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, num_channels)
        
        Returns:
            Patched tensor of shape (batch_size, num_channels, num_patches, patch_length)
        """
        batch_size, seq_length, num_channels = x.shape
        
        # Transpose to (batch_size, num_channels, seq_length)
        x = x.transpose(1, 2)
        
        # Use unfold to create patches
        patches = x.unfold(dimension=2, size=self.patch_length, step=self.patch_stride)
        # Shape: (batch_size, num_channels, num_patches, patch_length)
        
        return patches


class GatedAttention(nn.Module):
    """Gated attention mechanism."""
    
    def __init__(self, in_size: int, out_size: int, activation: str = "softmax"):
        super().__init__()
        self.attn_layer = nn.Linear(in_size, out_size)
        
        if activation == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated attention."""
        attn_weight = self.activation(self.attn_layer(x))
        return x * attn_weight


class PatchMixerBlock(nn.Module):
    """MLP-based mixer for patch dimension."""
    
    def __init__(self, config: TSPulseConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.num_patches, config.num_patches * config.expansion_factor),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.num_patches * config.expansion_factor, config.num_patches),
            nn.Dropout(config.dropout),
        )
        
        if config.gated_attn:
            self.gated_attn = GatedAttention(
                config.num_patches, 
                config.num_patches,
                activation=config.attention_activation
            )
        else:
            self.gated_attn = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch_size, num_channels, num_patches, d_model)
        
        Returns:
            Mixed tensor of same shape
        """
        residual = x
        
        # Normalize
        x = self.norm(x)
        
        # Transpose to mix patches: (B, C, P, D) -> (B, C, D, P)
        x = x.transpose(-1, -2)
        
        # Apply gated attention if enabled
        if self.gated_attn is not None:
            x = self.gated_attn(x)
        
        # MLP
        x = self.mlp(x)
        
        # Transpose back: (B, C, D, P) -> (B, C, P, D)
        x = x.transpose(-1, -2)
        
        return x + residual


class FeatureMixerBlock(nn.Module):
    """MLP-based mixer for feature dimension."""
    
    def __init__(self, config: TSPulseConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * config.expansion_factor),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * config.expansion_factor, config.d_model),
            nn.Dropout(config.dropout),
        )
        
        if config.gated_attn:
            self.gated_attn = GatedAttention(
                config.d_model,
                config.d_model,
                activation=config.attention_activation
            )
        else:
            self.gated_attn = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch_size, num_channels, num_patches, d_model)
        
        Returns:
            Mixed tensor of same shape
        """
        residual = x
        
        # Normalize
        x = self.norm(x)
        
        # Apply gated attention if enabled
        if self.gated_attn is not None:
            x = self.gated_attn(x)
        
        # MLP
        x = self.mlp(x)
        
        return x + residual


class TSPulseLayer(nn.Module):
    """Single TSPulse encoder layer combining patch and feature mixing."""
    
    def __init__(self, config: TSPulseConfig):
        super().__init__()
        self.num_patches = config.num_patches
        
        if config.num_patches > 1:
            self.patch_mixer = PatchMixerBlock(config)
        else:
            self.patch_mixer = None
        
        self.feature_mixer = FeatureMixerBlock(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch_size, num_channels, num_patches, d_model)
        
        Returns:
            Transformed tensor of same shape
        """
        if self.patch_mixer is not None:
            x = self.patch_mixer(x)
        x = self.feature_mixer(x)
        return x


class TSPulseEncoder(nn.Module):
    """TSPulse encoder with multiple layers."""
    
    def __init__(self, config: TSPulseConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TSPulseLayer(config) for _ in range(config.num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch_size, num_channels, num_patches, d_model)
        
        Returns:
            Encoded tensor of same shape
        """
        for layer in self.layers:
            x = layer(x)
        return x


class TSPulseEmbedding(nn.Module):
    """
    TSPulse-inspired timeseries embedding model.
    
    Takes raw timeseries data and produces embeddings suitable for downstream tasks.
    """
    
    def __init__(self, config: TSPulseConfig):
        super().__init__()
        self.config = config
        
        # Scaling/normalization
        if config.scaling == "revin":
            self.scaler = RevIN(config.num_input_channels, eps=config.norm_eps)
        else:
            self.scaler = None
        
        # Patching
        self.patching = Patching(config.patch_length, config.patch_stride)
        
        # Patch encoding: project patch to d_model dimension
        self.patch_encoding = nn.Linear(config.patch_length, config.d_model)
        
        # Encoder
        self.encoder = TSPulseEncoder(config)
        
        # Output normalization
        self.output_norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input timeseries of shape (batch_size, seq_length, num_channels)
        
        Returns:
            embeddings: Patch-level embeddings of shape (batch_size, num_channels, num_patches, d_model)
            pooled_embeddings: Pooled embeddings of shape (batch_size, d_model)
        """
        batch_size, seq_length, num_channels = x.shape
        
        # Validate input shape
        assert seq_length == self.config.context_length, \
            f"Expected seq_length={self.config.context_length}, got {seq_length}"
        assert num_channels == self.config.num_input_channels, \
            f"Expected num_channels={self.config.num_input_channels}, got {num_channels}"
        
        # Scale/normalize
        if self.scaler is not None:
            x, mean, std = self.scaler(x, mode="norm")
        
        # Patch the input
        # x: (batch_size, seq_length, num_channels)
        # patches: (batch_size, num_channels, num_patches, patch_length)
        patches = self.patching(x)
        
        # Encode patches to d_model dimension
        # (batch_size, num_channels, num_patches, patch_length) -> (batch_size, num_channels, num_patches, d_model)
        patch_embeddings = self.patch_encoding(patches)
        
        # Pass through encoder
        encoded = self.encoder(patch_embeddings)
        
        # Normalize output
        encoded = self.output_norm(encoded)
        
        # Create pooled representation by averaging over patches and channels
        # (batch_size, num_channels, num_patches, d_model) -> (batch_size, d_model)
        pooled = encoded.mean(dim=(1, 2))
        
        return encoded, pooled
    
    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get patch-level embeddings.
        
        Args:
            x: Input timeseries of shape (batch_size, seq_length, num_channels)
        
        Returns:
            Patch embeddings of shape (batch_size, num_channels, num_patches, d_model)
        """
        embeddings, _ = self.forward(x)
        return embeddings
    
    def get_sequence_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get sequence-level embedding (pooled).
        
        Args:
            x: Input timeseries of shape (batch_size, seq_length, num_channels)
        
        Returns:
            Sequence embedding of shape (batch_size, d_model)
        """
        _, pooled = self.forward(x)
        return pooled


def create_tspulse_embedding(
    context_length: int = 512,
    num_input_channels: int = 1,
    d_model: int = 128,
    num_layers: int = 3,
    patch_length: int = 8,
    **kwargs
) -> TSPulseEmbedding:
    """
    Convenience function to create a TSPulse embedding model.
    
    Args:
        context_length: Length of input timeseries
        num_input_channels: Number of input channels (1 for univariate)
        d_model: Hidden dimension
        num_layers: Number of encoder layers
        patch_length: Length of each patch
        **kwargs: Additional config parameters
    
    Returns:
        TSPulseEmbedding model
    """
    config = TSPulseConfig(
        context_length=context_length,
        num_input_channels=num_input_channels,
        d_model=d_model,
        num_layers=num_layers,
        patch_length=patch_length,
        **kwargs
    )
    return TSPulseEmbedding(config)

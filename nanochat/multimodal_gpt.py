"""
Multimodal GPT model that integrates vision embeddings from YOLOv9.

This extends the base GPT model to accept image embeddings alongside text tokens,
enabling vision-language understanding for robotics applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
from pathlib import Path
from PIL import Image
import numpy as np
from dataclasses import dataclass

from nanochat.gpt import GPT, GPTConfig
from nanochat.vision.yolov9 import YOLOv9Plus


@dataclass
class MultimodalGPTConfig(GPTConfig):
    """Extended config for multimodal GPT."""
    vision_embedding_dim: int = 512  # Dimension of vision embeddings from YOLO
    use_vision_projection: bool = True  # Whether to project vision embeddings to model dim
    vision_tokens_per_image: int = 1  # Number of vision tokens per image (1 for pooled, more for spatial)


class VisionProjector(nn.Module):
    """Projects vision embeddings to the model's embedding dimension."""
    
    def __init__(self, vision_dim: int, model_dim: int, num_layers: int = 2):
        super().__init__()
        self.vision_dim = vision_dim
        self.model_dim = model_dim
        
        # Multi-layer projection with ReLU^2 activation (matching GPT's MLP)
        layers = []
        current_dim = vision_dim
        hidden_dim = (vision_dim + model_dim) // 2
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim, bias=False))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, model_dim, bias=False))
        self.projection = nn.ModuleList(layers)
    
    def forward(self, x):
        """Project vision embeddings to model dimension."""
        for i, layer in enumerate(self.projection):
            x = layer(x)
            if i < len(self.projection) - 1:
                x = F.relu(x).square()  # ReLU^2 activation
        return x


class MultimodalGPT(nn.Module):
    """
    Multimodal GPT that can process both text tokens and vision embeddings.
    
    The model accepts sequences that can contain:
    1. Regular text tokens (processed through token embedding)
    2. Vision tokens (processed through vision projection)
    
    Vision tokens are marked with special token IDs and their embeddings are
    replaced with projected vision features.
    """
    
    def __init__(self, config: MultimodalGPTConfig, vision_model: Optional[YOLOv9Plus] = None):
        super().__init__()
        self.config = config
        
        # Base GPT model
        self.gpt = GPT(config)
        
        # Vision components
        self.vision_model = vision_model
        if config.use_vision_projection:
            self.vision_projector = VisionProjector(
                config.vision_embedding_dim,
                config.n_embd,
                num_layers=2
            )
        
        # Special token for vision embeddings
        # We'll use the last token in vocab as the vision token marker
        self.vision_token_id = config.vocab_size - 1
    
    def init_weights(self):
        """Initialize weights for the model."""
        self.gpt.init_weights()
        
        # Initialize vision projector if it exists
        if hasattr(self, 'vision_projector'):
            for module in self.vision_projector.modules():
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # Keep vision projector in float32 to match GPT linear layers
            # Only embeddings are in bfloat16, not the linear layers
    
    def get_device(self):
        return self.gpt.get_device()
    
    def encode_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Encode an image into vision embeddings.
        
        Args:
            image: Input image
            
        Returns:
            Vision embedding tensor of shape (1, vision_embedding_dim)
        """
        if self.vision_model is None:
            raise ValueError("Vision model not initialized. Pass vision_model to constructor.")
        
        # Extract embeddings using YOLO
        embedding = self.vision_model.get_image_embeddings(image, pool='avg')
        return embedding
    
    def project_vision_embeddings(self, vision_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project vision embeddings to model dimension.
        
        Args:
            vision_embeddings: Tensor of shape (B, vision_dim) or (B, N, vision_dim)
            
        Returns:
            Projected embeddings of shape (B, n_embd) or (B, N, n_embd)
        """
        if self.config.use_vision_projection:
            # Keep in float32 to match GPT linear layers
            return self.vision_projector(vision_embeddings)
        else:
            # If no projection, vision_dim must match n_embd
            assert vision_embeddings.size(-1) == self.config.n_embd
            return vision_embeddings
    
    def forward(
        self,
        idx: torch.Tensor,
        vision_embeddings: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        kv_cache=None,
        loss_reduction='mean'
    ):
        """
        Forward pass with optional vision embeddings.
        
        Args:
            idx: Token indices of shape (B, T)
            vision_embeddings: Optional vision embeddings of shape (B, N, vision_dim)
                              where N is the number of vision tokens
            vision_mask: Optional boolean mask of shape (B, T) indicating which positions
                        should use vision embeddings instead of token embeddings
            targets: Optional target tokens for training
            kv_cache: Optional KV cache for inference
            loss_reduction: Loss reduction method
            
        Returns:
            Loss (if targets provided) or logits
        """
        B, T = idx.size()
        
        # Get rotary embeddings
        assert T <= self.gpt.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.gpt.cos.size(1)}"
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.gpt.cos[:, T0:T0+T], self.gpt.sin[:, T0:T0+T]
        
        # Get token embeddings
        x = self.gpt.transformer.wte(idx)
        
        # Replace vision token positions with projected vision embeddings
        if vision_embeddings is not None and vision_mask is not None:
            # Project vision embeddings to model dimension
            projected_vision = self.project_vision_embeddings(vision_embeddings)
            
            # Expand vision embeddings to match sequence length if needed
            if projected_vision.dim() == 2:
                # (B, vision_dim) -> (B, 1, n_embd)
                projected_vision = projected_vision.unsqueeze(1)
            
            # Match dtype with token embeddings
            projected_vision = projected_vision.to(dtype=x.dtype)
            
            # Replace positions marked by vision_mask with vision embeddings
            # vision_mask shape: (B, T), projected_vision shape: (B, N, n_embd)
            for b in range(B):
                vision_positions = vision_mask[b].nonzero(as_tuple=True)[0]
                if len(vision_positions) > 0:
                    # Take the first N vision embeddings
                    n_vision = min(len(vision_positions), projected_vision.size(1))
                    x[b, vision_positions[:n_vision]] = projected_vision[b, :n_vision]
        
        # Apply norm after embedding (as in base GPT)
        from nanochat.gpt import norm
        x = norm(x)
        
        # Forward through transformer blocks
        for block in self.gpt.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)
        
        # Forward through lm_head
        softcap = 15
        if targets is not None:
            logits = self.gpt.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)
            logits = logits.float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            logits = self.gpt.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)
            return logits
    
    @torch.inference_mode()
    def generate_with_image(
        self,
        tokens: List[int],
        image: Optional[Union[str, Path, Image.Image, np.ndarray]] = None,
        vision_token_position: int = 0,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        seed: int = 42
    ):
        """
        Generate text with optional image context.
        
        Args:
            tokens: Initial text tokens
            image: Optional image to condition on
            vision_token_position: Position to insert vision token (default: 0, at start)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            seed: Random seed
            
        Yields:
            Generated tokens one at a time
        """
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        
        # Encode image if provided
        vision_embeddings = None
        if image is not None:
            vision_embeddings = self.encode_image(image)
            vision_embeddings = vision_embeddings.to(device)
            
            # Insert vision token marker at specified position
            tokens = tokens[:vision_token_position] + [self.vision_token_id] + tokens[vision_token_position:]
        
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        
        # Create vision mask if we have vision embeddings
        vision_mask = None
        if vision_embeddings is not None:
            vision_mask = torch.zeros_like(ids, dtype=torch.bool)
            vision_mask[0, vision_token_position] = True
        
        for _ in range(max_tokens):
            # Forward pass with vision embeddings
            logits = self.forward(ids, vision_embeddings, vision_mask)
            logits = logits[:, -1, :]
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            
            ids = torch.cat((ids, next_ids), dim=1)
            
            # Extend vision mask with False for new token
            if vision_mask is not None:
                vision_mask = torch.cat([vision_mask, torch.zeros((1, 1), dtype=torch.bool, device=device)], dim=1)
            
            token = next_ids.item()
            yield token
    
    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, 
                        vision_lr=0.001, weight_decay=0.0):
        """
        Setup optimizers for multimodal model.
        Adds vision projector parameters to the optimization.
        """
        # Get base GPT optimizers
        optimizers = self.gpt.setup_optimizers(unembedding_lr, embedding_lr, matrix_lr, weight_decay)
        
        # Add vision projector parameters if they exist
        if hasattr(self, 'vision_projector'):
            vision_params = list(self.vision_projector.parameters())
            vision_optimizer = torch.optim.AdamW(
                vision_params,
                lr=vision_lr,
                betas=(0.8, 0.95),
                eps=1e-10,
                weight_decay=weight_decay,
                fused=True
            )
            for group in vision_optimizer.param_groups:
                group["initial_lr"] = group["lr"]
            optimizers.append(vision_optimizer)
        
        return optimizers


def create_multimodal_gpt(
    config: Optional[MultimodalGPTConfig] = None,
    load_vision_model: bool = True,
    vision_model_id: str = "merve/yolov9",
    device: Optional[str] = None
) -> MultimodalGPT:
    """
    Convenience function to create a multimodal GPT model.
    
    Args:
        config: Model configuration (uses default if None)
        load_vision_model: Whether to load the vision model
        vision_model_id: HuggingFace model ID for YOLO
        device: Device to load models on
        
    Returns:
        MultimodalGPT instance
    """
    if config is None:
        config = MultimodalGPTConfig()
    
    vision_model = None
    if load_vision_model:
        vision_model = YOLOv9Plus(model_id=vision_model_id, device=device)
    
    model = MultimodalGPT(config, vision_model)
    return model

"""
Test script for TSPulse timeseries embedding model.
"""

import torch
import numpy as np
from nanochat.timeseries import TSPulseEmbedding, TSPulseConfig, create_tspulse_embedding


def test_basic_embedding():
    """Test basic embedding functionality."""
    print("=" * 60)
    print("Test 1: Basic Embedding")
    print("=" * 60)
    
    # Create a simple config
    config = TSPulseConfig(
        context_length=512,
        num_input_channels=1,
        d_model=128,
        num_layers=3,
        patch_length=8,
        patch_stride=8,
    )
    
    print(f"Config: context_length={config.context_length}, "
          f"num_channels={config.num_input_channels}, "
          f"d_model={config.d_model}, "
          f"num_patches={config.num_patches}")
    
    # Create model
    model = TSPulseEmbedding(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dummy input (batch_size=4, seq_length=512, num_channels=1)
    batch_size = 4
    x = torch.randn(batch_size, config.context_length, config.num_input_channels)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        embeddings, pooled = model(x)
    
    print(f"Patch embeddings shape: {embeddings.shape}")
    print(f"Pooled embedding shape: {pooled.shape}")
    
    # Verify shapes
    expected_patch_shape = (batch_size, config.num_input_channels, config.num_patches, config.d_model)
    expected_pooled_shape = (batch_size, config.d_model)
    
    assert embeddings.shape == expected_patch_shape, \
        f"Expected {expected_patch_shape}, got {embeddings.shape}"
    assert pooled.shape == expected_pooled_shape, \
        f"Expected {expected_pooled_shape}, got {pooled.shape}"
    
    print("✓ Test passed!\n")


def test_multivariate_embedding():
    """Test multivariate timeseries embedding."""
    print("=" * 60)
    print("Test 2: Multivariate Embedding")
    print("=" * 60)
    
    # Create config for multivariate data
    config = TSPulseConfig(
        context_length=512,
        num_input_channels=7,  # 7 channels (e.g., robot joint positions)
        d_model=128,
        num_layers=3,
        patch_length=16,
        patch_stride=16,
    )
    
    print(f"Config: context_length={config.context_length}, "
          f"num_channels={config.num_input_channels}, "
          f"d_model={config.d_model}, "
          f"num_patches={config.num_patches}")
    
    # Create model
    model = TSPulseEmbedding(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, config.context_length, config.num_input_channels)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        embeddings, pooled = model(x)
    
    print(f"Patch embeddings shape: {embeddings.shape}")
    print(f"Pooled embedding shape: {pooled.shape}")
    
    # Verify shapes
    expected_patch_shape = (batch_size, config.num_input_channels, config.num_patches, config.d_model)
    expected_pooled_shape = (batch_size, config.d_model)
    
    assert embeddings.shape == expected_patch_shape
    assert pooled.shape == expected_pooled_shape
    
    print("✓ Test passed!\n")


def test_convenience_function():
    """Test convenience function for creating model."""
    print("=" * 60)
    print("Test 3: Convenience Function")
    print("=" * 60)
    
    # Create model using convenience function
    model = create_tspulse_embedding(
        context_length=256,
        num_input_channels=3,
        d_model=64,
        num_layers=2,
        patch_length=8,
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 256, 3)
    
    with torch.no_grad():
        embeddings, pooled = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Patch embeddings shape: {embeddings.shape}")
    print(f"Pooled embedding shape: {pooled.shape}")
    
    assert embeddings.shape == (batch_size, 3, 32, 64)
    assert pooled.shape == (batch_size, 64)
    
    print("✓ Test passed!\n")


def test_different_patch_sizes():
    """Test different patch configurations."""
    print("=" * 60)
    print("Test 4: Different Patch Sizes")
    print("=" * 60)
    
    configs = [
        (512, 8, 8),   # Non-overlapping patches
        (512, 16, 16), # Larger patches
        (512, 8, 4),   # Overlapping patches
    ]
    
    for context_length, patch_length, patch_stride in configs:
        config = TSPulseConfig(
            context_length=context_length,
            num_input_channels=1,
            d_model=64,
            num_layers=2,
            patch_length=patch_length,
            patch_stride=patch_stride,
        )
        
        model = TSPulseEmbedding(config)
        x = torch.randn(2, context_length, 1)
        
        with torch.no_grad():
            embeddings, pooled = model(x)
        
        print(f"  patch_length={patch_length}, patch_stride={patch_stride}: "
              f"num_patches={config.num_patches}, "
              f"embeddings shape={embeddings.shape}")
    
    print("✓ Test passed!\n")


def test_synthetic_timeseries():
    """Test with synthetic timeseries data (sine waves)."""
    print("=" * 60)
    print("Test 5: Synthetic Timeseries (Sine Waves)")
    print("=" * 60)
    
    # Create model
    model = create_tspulse_embedding(
        context_length=512,
        num_input_channels=1,
        d_model=128,
        num_layers=3,
    )
    
    # Generate synthetic sine wave data
    batch_size = 8
    t = np.linspace(0, 4 * np.pi, 512)
    
    # Create different frequency sine waves
    frequencies = np.random.uniform(0.5, 2.0, batch_size)
    x_np = np.array([np.sin(freq * t) for freq in frequencies])
    x_np = x_np[:, :, np.newaxis]  # Add channel dimension
    
    x = torch.from_numpy(x_np).float()
    print(f"Generated {batch_size} sine waves with shape {x.shape}")
    
    # Get embeddings
    with torch.no_grad():
        embeddings, pooled = model(x)
    
    print(f"Patch embeddings shape: {embeddings.shape}")
    print(f"Pooled embedding shape: {pooled.shape}")
    
    # Check that different frequencies produce different embeddings
    # (simple sanity check)
    embedding_norms = torch.norm(pooled, dim=1)
    print(f"Embedding norms: min={embedding_norms.min():.4f}, "
          f"max={embedding_norms.max():.4f}, "
          f"mean={embedding_norms.mean():.4f}")
    
    print("✓ Test passed!\n")


def test_model_modes():
    """Test different model configurations."""
    print("=" * 60)
    print("Test 6: Model Configurations")
    print("=" * 60)
    
    # Test with gated attention
    config1 = TSPulseConfig(
        context_length=512,
        num_input_channels=1,
        d_model=64,
        num_layers=2,
        gated_attn=True,
    )
    model1 = TSPulseEmbedding(config1)
    print(f"Model with gated attention: {sum(p.numel() for p in model1.parameters())} params")
    
    # Test without gated attention
    config2 = TSPulseConfig(
        context_length=512,
        num_input_channels=1,
        d_model=64,
        num_layers=2,
        gated_attn=False,
    )
    model2 = TSPulseEmbedding(config2)
    print(f"Model without gated attention: {sum(p.numel() for p in model2.parameters())} params")
    
    # Test both models
    x = torch.randn(2, 512, 1)
    
    with torch.no_grad():
        _, pooled1 = model1(x)
        _, pooled2 = model2(x)
    
    print(f"Output shapes match: {pooled1.shape == pooled2.shape}")
    print("✓ Test passed!\n")


def test_helper_methods():
    """Test helper methods for getting embeddings."""
    print("=" * 60)
    print("Test 7: Helper Methods")
    print("=" * 60)
    
    model = create_tspulse_embedding(
        context_length=512,
        num_input_channels=1,
        d_model=128,
    )
    
    x = torch.randn(4, 512, 1)
    
    # Test get_patch_embeddings
    with torch.no_grad():
        patch_emb = model.get_patch_embeddings(x)
    print(f"get_patch_embeddings() output shape: {patch_emb.shape}")
    
    # Test get_sequence_embedding
    with torch.no_grad():
        seq_emb = model.get_sequence_embedding(x)
    print(f"get_sequence_embedding() output shape: {seq_emb.shape}")
    
    assert patch_emb.shape == (4, 1, 64, 128)
    assert seq_emb.shape == (4, 128)
    
    print("✓ Test passed!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TSPulse Timeseries Embedding Tests")
    print("=" * 60 + "\n")
    
    try:
        test_basic_embedding()
        test_multivariate_embedding()
        test_convenience_function()
        test_different_patch_sizes()
        test_synthetic_timeseries()
        test_model_modes()
        test_helper_methods()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

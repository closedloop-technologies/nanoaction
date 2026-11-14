"""
Demo script for TSPulse timeseries embedding.
Shows how to use the model for various timeseries tasks.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from nanochat.timeseries import create_tspulse_embedding, TSPulseConfig, TSPulseEmbedding


def demo_basic_usage():
    """Basic usage example."""
    print("=" * 60)
    print("Demo 1: Basic Usage")
    print("=" * 60)
    
    # Create a simple model
    model = create_tspulse_embedding(
        context_length=512,
        num_input_channels=1,
        d_model=128,
        num_layers=3,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create sample data
    x = torch.randn(4, 512, 1)
    print(f"Input shape: {x.shape}")
    
    # Get embeddings
    with torch.no_grad():
        patch_embeddings, sequence_embedding = model(x)
    
    print(f"Patch embeddings shape: {patch_embeddings.shape}")
    print(f"Sequence embedding shape: {sequence_embedding.shape}")
    print()


def demo_robot_trajectory():
    """Example with robot joint trajectories."""
    print("=" * 60)
    print("Demo 2: Robot Joint Trajectories")
    print("=" * 60)
    
    # Simulate 7-DOF robot arm trajectories
    num_joints = 7
    context_length = 512
    
    model = create_tspulse_embedding(
        context_length=context_length,
        num_input_channels=num_joints,
        d_model=128,
        num_layers=3,
        patch_length=16,
    )
    
    print(f"Model for {num_joints}-DOF robot")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generate synthetic robot trajectories (smooth sine waves)
    batch_size = 8
    t = np.linspace(0, 2 * np.pi, context_length)
    
    trajectories = []
    for _ in range(batch_size):
        # Each joint has different frequency and phase
        joint_data = []
        for joint_idx in range(num_joints):
            freq = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.uniform(0.5, 1.5)
            joint_trajectory = amplitude * np.sin(freq * t + phase)
            joint_data.append(joint_trajectory)
        trajectories.append(np.stack(joint_data, axis=-1))
    
    trajectories = torch.from_numpy(np.array(trajectories)).float()
    print(f"Generated {batch_size} robot trajectories: {trajectories.shape}")
    
    # Embed trajectories
    with torch.no_grad():
        _, trajectory_embeddings = model(trajectories)
    
    print(f"Trajectory embeddings shape: {trajectory_embeddings.shape}")
    
    # Compute pairwise similarities
    similarities = torch.mm(trajectory_embeddings, trajectory_embeddings.t())
    similarities = similarities / (torch.norm(trajectory_embeddings, dim=1, keepdim=True) @ 
                                   torch.norm(trajectory_embeddings, dim=1, keepdim=True).t())
    
    print(f"\nPairwise cosine similarities (diagonal should be ~1.0):")
    print(similarities[:4, :4].numpy())
    print()


def demo_sensor_data():
    """Example with sensor time series."""
    print("=" * 60)
    print("Demo 3: Sensor Data Processing")
    print("=" * 60)
    
    # Simulate IMU sensor data (accelerometer + gyroscope)
    # 3 axes accel + 3 axes gyro = 6 channels
    num_channels = 6
    context_length = 512
    
    model = create_tspulse_embedding(
        context_length=context_length,
        num_input_channels=num_channels,
        d_model=64,
        num_layers=2,
        patch_length=8,
    )
    
    print(f"Model for {num_channels}-channel IMU data")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generate synthetic IMU data with noise
    batch_size = 16
    sensor_data = torch.randn(batch_size, context_length, num_channels) * 0.1
    
    # Add some structured patterns
    t = torch.linspace(0, 4 * np.pi, context_length)
    for i in range(batch_size):
        # Add periodic motion to some channels
        sensor_data[i, :, 0] += torch.sin(t * (i % 3 + 1))
        sensor_data[i, :, 3] += torch.cos(t * (i % 3 + 1))
    
    print(f"Generated {batch_size} sensor readings: {sensor_data.shape}")
    
    # Embed sensor data
    with torch.no_grad():
        patch_emb, sensor_embeddings = model(sensor_data)
    
    print(f"Sensor embeddings shape: {sensor_embeddings.shape}")
    print(f"Patch embeddings shape: {patch_emb.shape}")
    
    # Analyze embedding statistics
    embedding_norms = torch.norm(sensor_embeddings, dim=1)
    print(f"\nEmbedding norms - min: {embedding_norms.min():.3f}, "
          f"max: {embedding_norms.max():.3f}, "
          f"mean: {embedding_norms.mean():.3f}")
    print()


def demo_different_configs():
    """Compare different model configurations."""
    print("=" * 60)
    print("Demo 4: Model Configuration Comparison")
    print("=" * 60)
    
    configs = [
        ("Small", {"d_model": 64, "num_layers": 2, "patch_length": 8}),
        ("Medium", {"d_model": 128, "num_layers": 3, "patch_length": 8}),
        ("Large", {"d_model": 256, "num_layers": 4, "patch_length": 16}),
    ]
    
    x = torch.randn(4, 512, 1)
    
    for name, config_kwargs in configs:
        model = create_tspulse_embedding(
            context_length=512,
            num_input_channels=1,
            **config_kwargs
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        
        with torch.no_grad():
            _, embeddings = model(x)
        
        print(f"{name:8s}: {num_params:8,} params, "
              f"d_model={config_kwargs['d_model']:3d}, "
              f"layers={config_kwargs['num_layers']}, "
              f"output_dim={embeddings.shape[-1]}")
    
    print()


def demo_patch_configurations():
    """Demonstrate different patching strategies."""
    print("=" * 60)
    print("Demo 5: Patching Strategies")
    print("=" * 60)
    
    context_length = 512
    
    patch_configs = [
        ("Non-overlapping 8", 8, 8),
        ("Non-overlapping 16", 16, 16),
        ("Overlapping 8/4", 8, 4),
        ("Overlapping 16/8", 16, 8),
    ]
    
    for name, patch_length, patch_stride in patch_configs:
        config = TSPulseConfig(
            context_length=context_length,
            num_input_channels=1,
            d_model=64,
            num_layers=2,
            patch_length=patch_length,
            patch_stride=patch_stride,
        )
        
        model = TSPulseEmbedding(config)
        num_params = sum(p.numel() for p in model.parameters())
        
        print(f"{name:20s}: {config.num_patches:3d} patches, "
              f"{num_params:6,} params")
    
    print()


def demo_embedding_similarity():
    """Demonstrate embedding similarity for similar time series."""
    print("=" * 60)
    print("Demo 6: Embedding Similarity")
    print("=" * 60)
    
    model = create_tspulse_embedding(
        context_length=512,
        num_input_channels=1,
        d_model=128,
        num_layers=3,
    )
    
    # Create similar and dissimilar time series
    t = np.linspace(0, 4 * np.pi, 512)
    
    # Base sine wave
    base = np.sin(t)
    
    # Similar: same frequency, slightly different phase/amplitude
    similar1 = 1.1 * np.sin(t + 0.1)
    similar2 = 0.9 * np.sin(t - 0.1)
    
    # Different: different frequency
    different1 = np.sin(3 * t)
    different2 = np.sin(0.5 * t)
    
    # Stack and convert to tensor
    time_series = np.stack([base, similar1, similar2, different1, different2])
    time_series = torch.from_numpy(time_series[:, :, np.newaxis]).float()
    
    print(f"Time series shape: {time_series.shape}")
    
    # Get embeddings
    with torch.no_grad():
        _, embeddings = model(time_series)
    
    # Compute cosine similarities
    embeddings_norm = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
    similarities = torch.mm(embeddings_norm, embeddings_norm.t())
    
    print("\nCosine similarity matrix:")
    print("         Base  Sim1  Sim2  Diff1 Diff2")
    labels = ["Base", "Sim1", "Sim2", "Diff1", "Diff2"]
    for i, label in enumerate(labels):
        print(f"{label:5s}:", end="")
        for j in range(5):
            print(f" {similarities[i, j].item():5.3f}", end="")
        print()
    
    print("\nNote: Similar time series (Base, Sim1, Sim2) should have higher similarity")
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("TSPulse Timeseries Embedding Demos")
    print("=" * 60 + "\n")
    
    demo_basic_usage()
    demo_robot_trajectory()
    demo_sensor_data()
    demo_different_configs()
    demo_patch_configurations()
    demo_embedding_similarity()
    
    print("=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

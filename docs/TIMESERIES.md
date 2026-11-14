# Timeseries Embedding Module

## Overview

A lightweight TSPulse-inspired timeseries embedding model for the nanovla project. This module provides efficient embeddings for temporal data, suitable for robot trajectories, sensor data, and other time series applications.

## Implementation Status

✅ **Completed:**
- Core TSPulse architecture with patching mechanism
- RevIN normalization for stable training
- Gated attention mechanism
- Patch and feature mixer blocks
- Configurable model architecture
- Support for univariate and multivariate time series
- Comprehensive test suite (7 tests, all passing)
- Demo scripts and documentation

❌ **Not Implemented (from full TSPulse):**
- FFT-based masking and reconstruction
- Task-specific heads (classification, forecasting, imputation, anomaly detection)
- Multi-head training with specialized loss functions
- Register tokens and virtual channel expansion
- Integration with the main LLM/VLA architecture

## Architecture

```
Input Time Series (B, L, C)
    ↓
RevIN Normalization
    ↓
Patching (B, C, P, patch_len)
    ↓
Patch Encoding (B, C, P, d_model)
    ↓
Encoder Layers (×N)
  ├─ Patch Mixer (temporal)
  └─ Feature Mixer (channels)
    ↓
Output Normalization
    ↓
Embeddings (B, C, P, d_model) + Pooled (B, d_model)
```

Where:
- B = batch size
- L = sequence length (context_length)
- C = number of channels
- P = number of patches
- d_model = hidden dimension

## Quick Start

### Installation

The module is already part of the nanovla project. No additional dependencies required beyond the existing PyTorch setup.

### Basic Usage

```python
from nanochat.timeseries import create_tspulse_embedding
import torch

# Create model
model = create_tspulse_embedding(
    context_length=512,
    num_input_channels=1,
    d_model=128,
    num_layers=3,
)

# Generate embeddings
x = torch.randn(4, 512, 1)  # (batch, seq_len, channels)
embeddings, pooled = model(x)
```

### Robot Trajectory Example

```python
from nanochat.timeseries import create_tspulse_embedding

# 7-DOF robot arm
model = create_tspulse_embedding(
    context_length=512,
    num_input_channels=7,  # 7 joints
    d_model=128,
    num_layers=3,
    patch_length=16,
)

# Process trajectory
trajectory = torch.randn(batch_size, 512, 7)
_, trajectory_embedding = model(trajectory)
# Use trajectory_embedding for action prediction, etc.
```

## Running Tests

```bash
# Run all tests
uv run python -m pytest tests/test_tspulse.py -v

# Run with output
uv run python -m pytest tests/test_tspulse.py -v -s
```

All 7 tests pass:
- ✅ Basic embedding
- ✅ Multivariate embedding
- ✅ Convenience function
- ✅ Different patch sizes
- ✅ Synthetic timeseries
- ✅ Model modes
- ✅ Helper methods

## Running Demos

```bash
# Run all demos
uv run python -m scripts.demo_tspulse
```

Demos include:
1. Basic usage
2. Robot joint trajectories
3. Sensor data processing
4. Model configuration comparison
5. Patching strategies
6. Embedding similarity

## Model Sizes

| Configuration | Parameters | Use Case |
|--------------|------------|----------|
| Small (64d, 2L) | ~84K | Lightweight, fast inference |
| Medium (128d, 3L) | ~312K | Balanced performance |
| Large (256d, 4L) | ~1.4M | High capacity |

## Configuration Options

```python
from nanochat.timeseries import TSPulseConfig

config = TSPulseConfig(
    # Input
    context_length=512,        # Length of input timeseries
    num_input_channels=1,      # Number of channels
    
    # Patching
    patch_length=8,            # Patch size
    patch_stride=8,            # Stride (=patch_length for non-overlapping)
    
    # Architecture
    d_model=128,               # Hidden dimension
    num_layers=3,              # Number of encoder layers
    expansion_factor=2,        # MLP expansion
    dropout=0.2,               # Dropout rate
    
    # Normalization
    scaling="revin",           # Normalization type
    norm_eps=1e-5,             # Numerical stability
    
    # Attention
    gated_attn=True,           # Use gated attention
    attention_activation="softmax",
    
    # Mode
    mode="common_channel",     # Channel mixing mode
)
```

## Use Cases for VLA Integration

### 1. Action History Encoding
Encode past robot actions as context for the VLA:
```python
# action_history: (batch, 512, 7) - past joint positions
action_embedding = model.get_sequence_embedding(action_history)
# Use action_embedding as additional context for VLA
```

### 2. Proprioceptive State
Encode robot proprioceptive data (joint positions, velocities, torques):
```python
# proprio: (batch, context_len, num_joints * 3)  # pos, vel, torque
proprio_embedding = model.get_sequence_embedding(proprio)
```

### 3. Sensor Fusion
Encode multi-modal sensor data:
```python
# imu_data: (batch, 512, 6)  # 3-axis accel + 3-axis gyro
sensor_embedding = model.get_sequence_embedding(imu_data)
```

### 4. Trajectory Prediction
Use patch-level embeddings for fine-grained trajectory analysis:
```python
patch_embeddings = model.get_patch_embeddings(trajectory)
# Each patch represents a temporal segment of the trajectory
```

## Future Integration Ideas

1. **VLA Context Enhancement**: Add timeseries embeddings as additional context tokens to the VLA
2. **Action Chunking**: Use patch embeddings to predict action chunks
3. **Temporal Attention**: Cross-attend between visual tokens and timeseries patches
4. **Multi-Modal Fusion**: Combine vision, language, and timeseries embeddings
5. **Trajectory Optimization**: Use embeddings for trajectory similarity and retrieval

## File Structure

```
nanochat/timeseries/
├── __init__.py          # Module exports
├── tspulse.py          # Main implementation
└── README.md           # Detailed documentation

tests/
└── test_tspulse.py     # Test suite

scripts/
└── demo_tspulse.py     # Demo scripts

docs/
└── TIMESERIES.md       # This file
```

## References

- **IBM Granite TSPulse**: https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1
- **IBM Granite TSFM**: https://github.com/ibm-granite/granite-tsfm
- **Tutorial**: https://www.ibm.com/think/tutorials/time-series-analysis-granite-time-tspulse

## Performance Notes

- **Inference Speed**: ~2ms per batch (batch_size=4, context=512, on CPU)
- **Memory**: ~1.2MB for medium model (312K params)
- **Training**: Not yet implemented, but architecture supports it

## Next Steps

To integrate with the VLA:

1. **Add timeseries input pipeline** to the VLA data loader
2. **Create fusion module** to combine vision, language, and timeseries embeddings
3. **Extend multimodal_gpt.py** to accept timeseries inputs
4. **Train end-to-end** with robot demonstration data

## License

Same as the main nanovla project.

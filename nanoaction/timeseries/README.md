# TSPulse Timeseries Embedding

A basic implementation of a TSPulse-inspired timeseries embedding model based on IBM Granite's [granite-timeseries-tspulse-r1](https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1).

## Overview

This module provides a lightweight, efficient model for embedding time series data. It uses a patching mechanism combined with MLP-based mixers to create rich representations of temporal data.

## Key Features

- **Patching**: Divides time series into fixed-length patches for efficient processing
- **RevIN Normalization**: Reversible Instance Normalization for stable training
- **Gated Attention**: Optional gated attention mechanism for enhanced expressiveness
- **Flexible Architecture**: Configurable depth, width, and patch sizes
- **Multivariate Support**: Handles both univariate and multivariate time series

## Architecture

The model consists of:

1. **Scaling/Normalization**: RevIN for input normalization
2. **Patching**: Splits time series into overlapping or non-overlapping patches
3. **Patch Encoding**: Projects patches to hidden dimension
4. **Encoder Layers**: Stack of mixer blocks that process:
   - Patch dimension (temporal mixing)
   - Feature dimension (channel mixing)
5. **Output**: Produces both patch-level and sequence-level embeddings

## Usage

### Basic Example

```python
from nanochat.timeseries import create_tspulse_embedding
import torch

# Create model
model = create_tspulse_embedding(
    context_length=512,      # Length of input timeseries
    num_input_channels=1,    # Number of channels (1 for univariate)
    d_model=128,             # Hidden dimension
    num_layers=3,            # Number of encoder layers
    patch_length=8,          # Length of each patch
)

# Generate embeddings
x = torch.randn(4, 512, 1)  # (batch_size, seq_length, num_channels)
embeddings, pooled = model(x)

# embeddings: (batch_size, num_channels, num_patches, d_model)
# pooled: (batch_size, d_model)
```

### Multivariate Time Series

```python
from nanochat.timeseries import TSPulseEmbedding, TSPulseConfig

# Configure for multivariate data (e.g., robot joint positions)
config = TSPulseConfig(
    context_length=512,
    num_input_channels=7,    # 7 robot joints
    d_model=128,
    num_layers=3,
    patch_length=16,
    patch_stride=16,
)

model = TSPulseEmbedding(config)

# Process robot trajectory
trajectory = torch.randn(2, 512, 7)  # 2 trajectories, 512 timesteps, 7 joints
patch_embeddings, sequence_embedding = model(trajectory)
```

### Configuration Options

```python
config = TSPulseConfig(
    # Input configuration
    context_length=512,           # Length of input timeseries
    num_input_channels=1,         # Number of input channels
    
    # Patching
    patch_length=8,               # Length of each patch
    patch_stride=8,               # Stride (=patch_length for non-overlapping)
    
    # Architecture
    d_model=128,                  # Hidden dimension
    num_layers=3,                 # Number of encoder layers
    expansion_factor=2,           # MLP expansion factor
    dropout=0.2,                  # Dropout rate
    
    # Normalization
    scaling="revin",              # "revin", "mean", "std", or None
    norm_eps=1e-5,                # Epsilon for numerical stability
    
    # Attention
    gated_attn=True,              # Use gated attention
    attention_activation="softmax", # "softmax" or "sigmoid"
    
    # Mode
    mode="common_channel",        # "common_channel" or "mix_channel"
)
```

## Helper Methods

```python
# Get only patch-level embeddings
patch_emb = model.get_patch_embeddings(x)
# Shape: (batch_size, num_channels, num_patches, d_model)

# Get only sequence-level embedding (pooled)
seq_emb = model.get_sequence_embedding(x)
# Shape: (batch_size, d_model)
```

## Model Size

The model is designed to be lightweight:

- **Univariate (512 context, 128 d_model, 3 layers)**: ~312K parameters
- **Multivariate (512 context, 7 channels, 128 d_model, 3 layers)**: ~267K parameters
- **Small (256 context, 64 d_model, 2 layers)**: ~53K parameters

## Use Cases

This embedding model can be used for:

1. **Robot Action Embeddings**: Encode robot trajectories for VLA models
2. **Sensor Data Processing**: Process time series from sensors
3. **Anomaly Detection**: Detect unusual patterns in temporal data
4. **Classification**: Classify time series sequences
5. **Similarity Search**: Find similar time series patterns

## Testing

Run the test suite:

```bash
uv run python -m pytest tests/test_tspulse.py -v
```

## Differences from Full TSPulse

This is a simplified implementation focusing on the core embedding functionality. The full IBM Granite TSPulse includes:

- FFT-based masking and reconstruction heads
- Task-specific heads (classification, forecasting, imputation, anomaly detection)
- Multi-head training with specialized loss functions
- Advanced features like register tokens and virtual channel expansion

This implementation provides the core encoder that can be extended for specific downstream tasks.

## References

- [IBM Granite TSPulse Model Card](https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1)
- [IBM Granite TSFM Repository](https://github.com/ibm-granite/granite-tsfm)
- [Time Series Analysis with Granite Time](https://www.ibm.com/think/tutorials/time-series-analysis-granite-time-tspulse)

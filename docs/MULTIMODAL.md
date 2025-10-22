# Multimodal Vision-Language Integration

This document describes the multimodal vision-language capabilities added to nanochat, enabling the model to process both images and text for robotics applications.

## Overview

The multimodal extension integrates **YOLOv9** vision embeddings with the GPT language model, allowing the system to:

- Extract visual features from images
- Condition text generation on visual input
- Support vision-language understanding for robotics tasks

## Architecture

### Components

1. **YOLOv9Plus** (`nanochat/vision/yolov9.py`)
   - Wraps ultralytics YOLOv9 for feature extraction
   - Extracts intermediate layer embeddings using forward hooks
   - Supports multiple pooling strategies (avg, max, flatten)

2. **MultimodalGPT** (`nanochat/multimodal_gpt.py`)
   - Extends base GPT with vision embedding support
   - Projects vision features to model embedding dimension
   - Replaces special vision tokens with projected embeddings

3. **VisionProjector**
   - Multi-layer projection network
   - Maps vision embeddings (84-dim) to model dimension (e.g., 256-dim)
   - Uses ReLU² activation for consistency with GPT MLP

## Usage

### Basic Example

```python
import torch
from PIL import Image
from nanochat.vision import YOLOv9Plus
from nanochat.multimodal_gpt import create_multimodal_gpt, MultimodalGPTConfig

# 1. Extract vision embeddings
yolo = YOLOv9Plus(device='cuda')
image = Image.open('robot_scene.jpg')
vision_emb = yolo.get_image_embeddings(image, pool='avg')
print(f"Vision embedding shape: {vision_emb.shape}")  # (1, 84)

# 2. Create multimodal model
config = MultimodalGPTConfig(
    sequence_len=256,
    vocab_size=50257,
    n_layer=12,
    n_head=8,
    n_kv_head=8,
    n_embd=768,
    vision_embedding_dim=84,  # Match YOLO output
    use_vision_projection=True,
)

model = create_multimodal_gpt(config, device='cpu')
model.init_weights()
model.to(device='cuda', dtype=torch.float32)

# 3. Generate text conditioned on image
prompt_tokens = [1, 2, 3, 4, 5]  # Your tokenized prompt
vision_token_id = 50256  # Special token for vision

# Create input with vision token
tokens = torch.tensor([[vision_token_id] + prompt_tokens], device='cuda')
vision_mask = torch.zeros_like(tokens, dtype=torch.bool)
vision_mask[0, 0] = True  # Mark first position as vision

# Forward pass
logits = model(tokens, vision_emb.unsqueeze(0), vision_mask)

# Generate
generated = model.generate_with_image(
    prompt_tokens=prompt_tokens,
    image=image,
    max_new_tokens=50,
    temperature=0.8
)
```

### Vision Embedding Extraction

```python
from nanochat.vision import YOLOv9Plus

# Initialize YOLO
yolo = YOLOv9Plus(device='cuda')

# Extract embeddings with different pooling
avg_emb = yolo.get_image_embeddings(image, pool='avg')    # (1, 84)
max_emb = yolo.get_image_embeddings(image, pool='max')    # (1, 84)
flat_emb = yolo.get_image_embeddings(image, pool='flatten')  # (1, C*H*W)

# Extract spatial features (before pooling)
features = yolo.extract_features(image)  # (1, C, H, W)
```

### Training with Vision

```python
# Prepare batch with vision
batch_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
batch_images = [...]  # List of PIL Images

# Extract vision embeddings
vision_embeddings = []
for img in batch_images:
    emb = yolo.get_image_embeddings(img)
    vision_embeddings.append(emb)
vision_embeddings = torch.cat(vision_embeddings, dim=0)

# Create vision mask (mark positions with vision tokens)
vision_mask = torch.zeros_like(batch_tokens, dtype=torch.bool)
vision_mask[:, 0] = True  # First token is vision

# Forward pass with loss
loss = model(
    batch_tokens,
    vision_embeddings,
    vision_mask,
    targets=batch_tokens,
    loss_reduction='mean'
)

# Backward
loss.backward()
optimizer.step()
```

## Configuration

### MultimodalGPTConfig

Extends `GPTConfig` with vision-specific parameters:

```python
@dataclass
class MultimodalGPTConfig(GPTConfig):
    vision_embedding_dim: int = 84  # Dimension of YOLO embeddings
    use_vision_projection: bool = True  # Project vision to model dim
    vision_tokens_per_image: int = 1  # Number of vision tokens per image
```

### Key Parameters

- **vision_embedding_dim**: Must match YOLO output dimension (84 for default)
- **use_vision_projection**: Set to `True` to project vision features
- **vision_tokens_per_image**: Currently supports 1 (pooled embedding)

## Technical Details

### Dtype Handling

The model uses mixed precision:

- **Embeddings**: bfloat16 (for memory efficiency)
- **Linear layers**: float32 (for numerical stability)
- **Vision projector**: float32, converted to bfloat16 before injection

The code automatically handles dtype conversions:

```python
# Vision embeddings are projected in float32
projected_vision = self.vision_projector(vision_embeddings)

# Then converted to match token embeddings (bfloat16)
projected_vision = projected_vision.to(dtype=x.dtype)
```

### Memory Management

For GPU training:

1. Load YOLO on GPU for inference
2. Extract embeddings and move to CPU if needed
3. Clear CUDA cache between batches
4. Use gradient checkpointing for large models

```python
import torch

# Clear CUDA cache
torch.cuda.empty_cache()
torch.cuda.synchronize()

# Monitor memory
allocated = torch.cuda.memory_allocated(0) / 1e9
print(f"GPU memory: {allocated:.2f} GB")
```

## Testing

Run the comprehensive GPU test:

```bash
uv run python test_multimodal_gpu.py
```

This tests:

1. YOLOv9 embedding extraction on GPU
2. Multimodal GPT model creation
3. Forward passes with vision
4. Text generation conditioned on images

Expected output:

```text
✓ ALL TESTS PASSED!
Final GPU memory: 0.04 GB
```

## Performance

### Benchmarks (RTX 2080 SUPER, 8GB)

| Component | Memory | Time |
|-----------|--------|------|
| YOLOv9 load | 0.10 GB | ~2s |
| Embedding extraction | 0.11 GB | ~50ms |
| Small model (2.1M params) | 0.01 GB | - |
| Forward pass (batch=1, seq=16) | 0.04 GB | ~10ms |
| Generation (10 tokens) | 0.04 GB | ~100ms |

### Scaling

For larger models:

- Use smaller `device_batch_size` to fit in memory
- Consider using CPU for YOLO inference to save GPU memory
- Use gradient accumulation for effective larger batches

## Limitations

1. **Single vision token**: Currently supports 1 pooled embedding per image
2. **Fixed YOLO model**: Uses ultralytics default YOLOv9
3. **No spatial attention**: Vision features are globally pooled
4. **Training not optimized**: Vision encoder is frozen during training

## Future Improvements

1. **Multiple vision tokens**: Support spatial grid of vision embeddings
2. **Trainable vision encoder**: Fine-tune YOLO on robotics data
3. **Cross-attention**: Add explicit vision-language attention layers
4. **Efficient inference**: Quantization and pruning for edge deployment
5. **Multi-image support**: Process sequences of images for video understanding

## References

- [YOLOv9 Paper](https://arxiv.org/abs/2402.13616)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [nanochat](https://github.com/karpathy/nanochat)
- [OpenVLA](https://openvla.github.io/)
- [Pi0.5](https://github.com/Physical-Intelligence/openpi)

## Troubleshooting

### CUDA OOM Errors

If you encounter out-of-memory errors:

```python
# Use CPU for YOLO
yolo = YOLOv9Plus(device='cpu')

# Or reduce model size
config = MultimodalGPTConfig(
    n_layer=2,  # Fewer layers
    n_embd=256,  # Smaller embedding
    ...
)
```

### Dtype Mismatches

Ensure proper dtype handling:

```python
# Always convert model to float32 for stability
model.to(device='cuda', dtype=torch.float32)

# Vision embeddings will be automatically converted
```

### Import Errors

Make sure to set PYTHONPATH:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
uv run python your_script.py
```

Or use the wrapper script:

```bash
./run_multimodal.sh
```

## License

MIT - Same as nanochat

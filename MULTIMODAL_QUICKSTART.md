# Multimodal Vision-Language Quick Start

Get started with vision-language capabilities in under 5 minutes!

## Installation

Ensure dependencies are installed:

```bash
uv sync
```

## Quick Test

Run the comprehensive test to verify everything works:

```bash
uv run python test_multimodal_gpu.py
```

Expected output:

```text
✓ ALL TESTS PASSED!
Final GPU memory: 0.04 GB
```

## Basic Usage

### Extract Vision Embeddings

```python
from nanochat.vision import YOLOv9Plus
from PIL import Image

# Load YOLO
yolo = YOLOv9Plus(device='cuda')

# Extract embeddings from image
image = Image.open('robot_scene.jpg')
embedding = yolo.get_image_embeddings(image, pool='avg')
print(f"Shape: {embedding.shape}")  # (1, 84)
```

### Create Multimodal Model

```python
from nanochat.multimodal_gpt import create_multimodal_gpt, MultimodalGPTConfig
import torch

# Configure model
config = MultimodalGPTConfig(
    sequence_len=256,
    vocab_size=50257,
    n_layer=12,
    n_embd=768,
    vision_embedding_dim=84,
)

# Create and initialize
model = create_multimodal_gpt(config, device='cpu')
model.init_weights()
model.to(device='cuda', dtype=torch.float32)
```

### Generate with Vision

```python
# Prepare inputs
prompt_tokens = [1, 2, 3, 4, 5]
vision_token_id = 50256

# Create batch
tokens = torch.tensor([[vision_token_id] + prompt_tokens], device='cuda')
vision_mask = torch.zeros_like(tokens, dtype=torch.bool)
vision_mask[0, 0] = True

# Forward pass
logits = model(tokens, embedding.unsqueeze(0), vision_mask)
```

## Next Steps

- Read [docs/MULTIMODAL.md](docs/MULTIMODAL.md) for complete documentation
- See [README-NANOACTION.md](README-NANOACTION.md) for robotics integration
- Check [test_multimodal_gpu.py](test_multimodal_gpu.py) for more examples

## Troubleshooting

### CUDA OOM

Use CPU for YOLO:

```python
yolo = YOLOv9Plus(device='cpu')
```

### Import Errors

Set PYTHONPATH:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Performance

On RTX 2080 SUPER (8GB):

- YOLO load: ~2s, 0.10 GB
- Embedding extraction: ~50ms
- Small model (2.1M params): 0.01 GB
- Forward pass: ~10ms, 0.04 GB

## License

MIT

#!/usr/bin/env python3
"""
Test multimodal GPT with GPU, handling CUDA memory properly.
"""

import torch
from PIL import Image
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def clear_cuda_cache():
    """Clear CUDA cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def test_yolo_embeddings():
    """Test YOLO embedding extraction on GPU."""
    print("=" * 70)
    print("TEST 1: YOLOv9 Embeddings on GPU")
    print("=" * 70)
    
    from nanochat.vision import YOLOv9Plus
    
    # Clear cache first
    clear_cuda_cache()
    
    # Create YOLO model on GPU
    print("\n1. Loading YOLOv9 model on GPU...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
        print(f"   Cached: {torch.cuda.memory_reserved(0) / 1e9:.4f} GB")
    
    yolo = YOLOv9Plus(device=device)
    print("   ✓ Model loaded")
    
    if device == 'cuda':
        print(f"   Allocated after load: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
    
    # Create test image
    print("\n2. Creating test image...")
    image = Image.new('RGB', (640, 480), color=(100, 150, 200))
    print(f"   ✓ Image: {image.size}")
    
    # Extract embeddings
    print("\n3. Extracting embeddings...")
    try:
        embedding = yolo.get_image_embeddings(image, pool='avg')
        print(f"   ✓ Embedding shape: {embedding.shape}")
        print(f"   ✓ Embedding device: {embedding.device}")
        print(f"   ✓ Stats: mean={embedding.mean():.4f}, std={embedding.std():.4f}")
        
        if device == 'cuda':
            print(f"   Memory after inference: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
        
        return yolo, embedding
    except torch.cuda.OutOfMemoryError as e:
        print(f"   ✗ CUDA OOM Error: {e}")
        print("\n   Trying with CPU instead...")
        clear_cuda_cache()
        del yolo
        
        yolo = YOLOv9Plus(device='cpu')
        embedding = yolo.get_image_embeddings(image, pool='avg')
        print(f"   ✓ Embedding shape (CPU): {embedding.shape}")
        return yolo, embedding


def test_multimodal_gpt():
    """Test multimodal GPT on GPU."""
    print("\n" + "=" * 70)
    print("TEST 2: Multimodal GPT on GPU")
    print("=" * 70)
    
    from nanochat.multimodal_gpt import create_multimodal_gpt, MultimodalGPTConfig
    
    # Clear cache
    clear_cuda_cache()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n1. Creating model on {device}...")
    
    # Create small config to fit in memory
    # Note: vision_embedding_dim should match YOLO output (84 for the last layer)
    config = MultimodalGPTConfig(
        sequence_len=128,
        vocab_size=1000,
        n_layer=2,  # Small for testing
        n_head=4,
        n_kv_head=4,
        n_embd=256,
        vision_embedding_dim=84,  # Match YOLO output
    )
    
    try:
        model = create_multimodal_gpt(
            config=config,
            load_vision_model=False,  # Don't load YOLO again
            device='cpu'  # Create on CPU first
        )
        model.init_weights()  # Initialize weights on CPU
        # Convert everything to float32 to avoid dtype issues
        model.to(device=device, dtype=torch.float32)
        print("   ✓ Model created")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ Parameters: {total_params:,}")
        
        if device == 'cuda':
            print(f"   Memory: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
        
        return model
    except torch.cuda.OutOfMemoryError:
        print("   ✗ CUDA OOM, using CPU...")
        clear_cuda_cache()
        model = create_multimodal_gpt(
            config=config,
            load_vision_model=False,
            device='cpu'
        )
        model.init_weights()
        print("   ✓ Model created on CPU")
        return model


def test_forward_pass(model):
    """Test forward pass."""
    print("\n" + "=" * 70)
    print("TEST 3: Forward Pass")
    print("=" * 70)
    
    device = model.get_device()
    print(f"\n1. Running on {device}...")
    
    # Small batch to save memory
    batch_size = 1
    seq_len = 16
    
    tokens = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    # Use actual YOLO embedding dimension (84)
    vision_emb = torch.randn(batch_size, 1, 84, device=device)
    vision_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    vision_mask[:, 0] = True
    
    print(f"   Batch size: {batch_size}, Seq len: {seq_len}")
    
    # Forward
    model.eval()
    with torch.no_grad():
        logits = model(tokens, vision_emb, vision_mask)
    
    print(f"   ✓ Logits shape: {logits.shape}")
    
    if device.type == 'cuda':
        print(f"   Memory: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
    
    return logits


def test_generation(model):
    """Test generation with image."""
    print("\n" + "=" * 70)
    print("TEST 4: Text Generation")
    print("=" * 70)
    
    device = model.get_device()
    
    # Create image
    image = Image.new('RGB', (640, 480), color='red')
    print(f"\n1. Image: {image.size}")
    
    # Encode image
    print("2. Encoding image...")
    if model.vision_model is None:
        # Create a temporary YOLO model
        from nanochat.vision import YOLOv9Plus
        yolo = YOLOv9Plus(device='cpu')  # Use CPU for YOLO to save GPU memory
        vision_emb = yolo.get_image_embeddings(image)
        vision_emb = vision_emb.to(device)
        del yolo
        clear_cuda_cache()
    else:
        vision_emb = model.encode_image(image)
    
    print(f"   ✓ Embedding shape: {vision_emb.shape}")
    
    # Generate
    print("3. Generating tokens...")
    prompt_tokens = [1, 2, 3, 4, 5]
    
    model.eval()
    generated = []
    
    # Create vision mask
    tokens_with_vision = [model.vision_token_id] + prompt_tokens
    ids = torch.tensor([tokens_with_vision], dtype=torch.long, device=device)
    vision_mask = torch.zeros_like(ids, dtype=torch.bool)
    vision_mask[0, 0] = True
    
    # Generate a few tokens
    for i in range(10):
        with torch.no_grad():
            logits = model(ids, vision_emb.unsqueeze(0), vision_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=1)
            vision_mask = torch.cat([vision_mask, torch.zeros((1, 1), dtype=torch.bool, device=device)], dim=1)
            generated.append(next_token.item())
    
    print(f"   ✓ Generated {len(generated)} tokens: {generated}")
    
    if device.type == 'cuda':
        print(f"   Memory: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("MULTIMODAL GPT - GPU TEST")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA not available, tests will run on CPU")
    else:
        print(f"\n✓ CUDA available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
        # Test 1: YOLO embeddings
        yolo, embedding = test_yolo_embeddings()
        
        # Clean up YOLO to free memory
        del yolo
        clear_cuda_cache()
        
        # Test 2: Create model
        model = test_multimodal_gpt()
        
        # Test 3: Forward pass
        logits = test_forward_pass(model)
        
        # Test 4: Generation
        test_generation(model)
        
        # Success
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        
        if torch.cuda.is_available():
            print(f"\nFinal GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ ERROR")
        print("=" * 70)
        print(f"\n{e}")
        import traceback
        traceback.print_exc()
        
        if torch.cuda.is_available():
            print(f"\nGPU memory at error: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")


if __name__ == "__main__":
    main()

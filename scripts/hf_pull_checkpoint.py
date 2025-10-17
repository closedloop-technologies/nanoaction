"""Pulls weights and tokenizer from hugging face

Default checkpoint: https://huggingface.co/karpathy/nanochat-d32

Usage:
python hf_pull_weights.py [--repo REPO_ID]

This script downloads model weights and tokenizer files from HuggingFace and places them
in the correct directory structure:
- tokenizer files (token_bytes.pt, tokenizer.pkl) -> ~/.cache/nanochat/tokenizer/
- checkpoint files (meta_*.json, model_*.pt) -> ~/.cache/nanochat/chatsft_checkpoints/d32/
"""

import os
import argparse
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser(description="Download nanochat weights from HuggingFace")
    parser.add_argument(
        "--repo",
        type=str,
        default="karpathy/nanochat-d32",
        help="HuggingFace repo ID (default: karpathy/nanochat-d32)"
    )
    args = parser.parse_args()

    # Expand home directory
    base_dir = Path.home() / ".cache" / "nanochat"
    tokenizer_dir = base_dir / "tokenizer"
    checkpoint_dir = base_dir / "chatsft_checkpoints" / "d32"

    # Create directories
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from HuggingFace repo: {args.repo}")
    print(f"Base directory: {base_dir}")

    # Define files to download and their destinations
    files_to_download = [
        # (filename_in_repo, destination_dir)
        ("token_bytes.pt", tokenizer_dir),
        ("tokenizer.pkl", tokenizer_dir),
        ("meta_000650.json", checkpoint_dir),
        ("model_000650.pt", checkpoint_dir),
    ]

    # Download each file
    for filename, dest_dir in files_to_download:
        print(f"\nDownloading {filename}...")
        try:
            # Download to cache first
            downloaded_path = hf_hub_download(
                repo_id=args.repo,
                filename=filename,
                local_dir_use_symlinks=False,
            )
            
            # Copy to the correct location
            dest_path = dest_dir / filename
            shutil.copy2(downloaded_path, dest_path)
            print(f"  ✓ Saved to {dest_path}")
            
        except Exception as e:
            print(f"  ✗ Error downloading {filename}: {e}")
            print(f"    Continuing with remaining files...")

    print("\n" + "="*60)
    print("Download complete!")
    print(f"Tokenizer files: {tokenizer_dir}")
    print(f"Checkpoint files: {checkpoint_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

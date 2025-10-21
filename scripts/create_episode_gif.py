#!/usr/bin/env python3
"""Create a GIF from observation images in a pickle file."""

import pickle
import sys
from pathlib import Path
from PIL import Image
import io

def create_gif_from_pickle(pickle_path: str, output_path: str = None):
    """Extract images from pickle file and create a GIF."""
    
    # Load pickle file
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract images from steps
    images = []
    for step in data['steps']:
        # Get JPEG bytes from observation
        img_bytes = step['observation']['image']
        # Convert to PIL Image
        img = Image.open(io.BytesIO(img_bytes))
        images.append(img)
    
    print(f"Extracted {len(images)} images from {pickle_path}")
    
    # Set output path
    if output_path is None:
        pickle_file = Path(pickle_path)
        output_path = pickle_file.parent / f"{pickle_file.stem}.gif"
    
    # Create GIF
    if images:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=100,  # 100ms per frame = 10 fps
            loop=0
        )
        print(f"Created GIF: {output_path}")
        print(f"  - Frames: {len(images)}")
        print(f"  - Size: {images[0].size}")
    else:
        print("No images found!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_episode_gif.py <pickle_file> [output_gif]")
        sys.exit(1)
    
    pickle_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    create_gif_from_pickle(pickle_path, output_path)

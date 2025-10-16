#!/usr/bin/env python3
"""
Generate nanoAction logo using pyfiglet dos_rebel font
"""

import subprocess
import pyfiglet
from PIL import Image, ImageDraw, ImageFont


def generate_ascii_art(text, font='dos_rebel', width=120):
    """Generate ASCII art using pyfiglet"""
    ascii_art = pyfiglet.figlet_format(text, font=font, width=width)
    return ascii_art


def save_ascii_to_png(ascii_text, output_path, bg_color=(45, 45, 45), text_color=(255, 255, 255), padding=0):
    """Convert ASCII art to PNG image with tight cropping"""
    lines = ascii_text.split('\n')

    # Remove leading and trailing empty lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in lines]

    # Use monospace font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 16)
        except:
            font = ImageFont.load_default()

    # Calculate dimensions based on actual content
    max_width = max(len(line) for line in lines) if lines else 0
    num_lines = len(lines)

    char_width = 10
    char_height = 18
    img_width = max_width * char_width + (2 * padding) - padding*20
    img_height = num_lines * char_height + (2 * padding)

    # Create image
    img = Image.new('RGB', (img_width, img_height), bg_color)
    draw = ImageDraw.Draw(img)

    # Draw text
    y_offset = padding
    for line in lines:
        draw.text((padding+char_width*1.5, y_offset), line, fill=text_color, font=font)
        y_offset += char_height

    # Save
    img.save(output_path)
    print(f"✓ Logo saved to {output_path}")


if __name__ == "__main__":
    # Generate ASCII art
    print("Generating nanoAction logo with dos_rebel font...")
    ascii_art = generate_ascii_art("nanoAction", font='dos_rebel', width=120)

    # Print to console
    print("\n" + ascii_art)

    # Save to PNG
    output_file = "./dev/nanoaction.png"
    save_ascii_to_png(ascii_art, output_file)

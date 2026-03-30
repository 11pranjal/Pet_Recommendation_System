#!/usr/bin/env python3
"""
Image Preprocessing Script for Pet Adoption System
===================================================
Walks petimage/ and:
  1. Resizes oversized images to max 800×800 (keeping aspect ratio)
  2. Converts to optimised JPEG (quality 85) for faster web loading
  3. Strips EXIF metadata
  4. Removes Zone.Identifier files (Windows WSL artifact)
  5. Prints a summary of savings

Usage:  python3 preprocess_images.py          (dry-run by default)
        python3 preprocess_images.py --apply   (actually processes images)
"""

import os
import sys
from PIL import Image

PETIMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "petimage")
MAX_DIMENSION = 800   # max width or height in pixels
JPEG_QUALITY = 85     # quality for JPEG compression
SIZE_THRESHOLD = 200_000  # only process files larger than 200 KB


def human_size(nbytes):
    for unit in ['B', 'KB', 'MB']:
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} GB"


def process_image(filepath, apply=False):
    """Process a single image. Returns (original_size, new_size) or None if skipped."""
    original_size = os.path.getsize(filepath)

    # Skip small files
    if original_size < SIZE_THRESHOLD:
        return None

    try:
        with Image.open(filepath) as img:
            orig_w, orig_h = img.size

            # Determine if we need to resize
            need_resize = orig_w > MAX_DIMENSION or orig_h > MAX_DIMENSION

            if not need_resize and original_size < SIZE_THRESHOLD * 2:
                return None  # image is already small enough

            # Convert to RGB if necessary (e.g. RGBA PNGs)
            if img.mode in ('RGBA', 'P', 'LA'):
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if 'A' in img.mode else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize using high-quality Lanczos resampling
            if need_resize:
                img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)

            if not apply:
                # Estimate new size by saving to temporary bytes
                import io
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=JPEG_QUALITY, optimize=True)
                new_size = buf.tell()
                return (original_size, new_size)

            # Save — replace extension with .jpg if it was .png
            base, ext = os.path.splitext(filepath)
            new_path = base + '.jpg' if ext.lower() == '.png' else filepath

            img.save(new_path, format='JPEG', quality=JPEG_QUALITY, optimize=True)
            new_size = os.path.getsize(new_path)

            # Remove old file if we changed extension
            if new_path != filepath and os.path.exists(new_path):
                os.remove(filepath)

            return (original_size, new_size)

    except Exception as e:
        print(f"  ⚠ Error processing {filepath}: {e}")
        return None


def cleanup_zone_files():
    """Remove Zone.Identifier files (WSL artifact)."""
    removed = 0
    for root, dirs, files in os.walk(PETIMAGE_DIR):
        # Skip .git
        dirs[:] = [d for d in dirs if d != '.git']
        for f in files:
            if ':Zone' in f or f.endswith('.Identifier'):
                path = os.path.join(root, f)
                try:
                    os.remove(path)
                    removed += 1
                except Exception:
                    pass
    return removed


def main():
    apply = '--apply' in sys.argv

    print("=" * 60)
    print("  PET IMAGE PREPROCESSOR")
    print("=" * 60)
    print(f"  Mode:          {'APPLY (writing changes)' if apply else 'DRY-RUN (preview only)'}")
    print(f"  Max dimension: {MAX_DIMENSION}px")
    print(f"  JPEG quality:  {JPEG_QUALITY}")
    print(f"  Size threshold:{human_size(SIZE_THRESHOLD)}")
    print("=" * 60)

    # Cleanup zone files first
    zone_removed = cleanup_zone_files()
    if zone_removed:
        print(f"\n  Removed {zone_removed} Zone.Identifier files")

    # Collect all image files
    image_files = []
    for root, dirs, files in os.walk(PETIMAGE_DIR):
        dirs[:] = [d for d in dirs if d != '.git']
        for f in sorted(files):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_files.append(os.path.join(root, f))

    print(f"\n  Found {len(image_files)} images in petimage/\n")

    total_original = 0
    total_new = 0
    processed = 0
    skipped = 0

    for filepath in image_files:
        rel_path = os.path.relpath(filepath, PETIMAGE_DIR)
        result = process_image(filepath, apply=apply)

        if result is None:
            skipped += 1
            continue

        orig_size, new_size = result
        total_original += orig_size
        total_new += new_size
        processed += 1
        saving = orig_size - new_size
        pct = (saving / orig_size * 100) if orig_size > 0 else 0
        print(f"  {'✅' if apply else '📋'} {rel_path}")
        print(f"     {human_size(orig_size)} → {human_size(new_size)}  (saved {human_size(saving)}, {pct:.0f}%)")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Images processed: {processed}")
    print(f"  Images skipped:   {skipped} (already optimised)")

    if processed > 0:
        total_saving = total_original - total_new
        pct = (total_saving / total_original * 100) if total_original > 0 else 0
        print(f"  Original total:   {human_size(total_original)}")
        print(f"  New total:        {human_size(total_new)}")
        print(f"  Total saved:      {human_size(total_saving)} ({pct:.0f}%)")

    if not apply and processed > 0:
        print(f"\n  Run with --apply to actually process images:")
        print(f"  $ python3 preprocess_images.py --apply")

    print("=" * 60)


if __name__ == "__main__":
    main()

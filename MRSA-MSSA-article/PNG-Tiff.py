#!/usr/bin/env python3
"""
Batch-convert all .png images in the current directory
to .tiff at 400 DPI (or whatever you choose).
"""

from PIL import Image
import glob
import os

# ───── EDIT THIS LINE: set your desired DPI (e.g., 400×400 or 600×600) ─────
target_dpi = (400, 400)
# ─────────────────────────────────────────────────────────────────────────────

# 1. Find all PNG files
png_files = sorted(glob.glob("*.png"))
if not png_files:
    print("No .png files found in current directory.")
    exit(0)

# 2. Create output directory
output_dir = "tiff_output"
os.makedirs(output_dir, exist_ok=True)

# 3. Load each PNG, save as TIFF with DPI metadata
for png_file in png_files:
    img = Image.open(png_file)
    # If you want to **resample** the image to a higher pixel dimension (optional),
    # you could do something like:
    #   new_size = (img.width * scale, img.height * scale)
    #   img = img.resize(new_size, Image.LANCZOS)
    #
    # But typically setting the DPI metadata alone is enough to produce a high-resolution TIFF.
    tiff_filename = os.path.join(output_dir, png_file.replace(".png", ".tiff"))
    img.save(tiff_filename, dpi=target_dpi)
    print(f"Saved {tiff_filename} at {target_dpi[0]} DPI")

#!/usr/bin/env python3
"""
Combine individual *_seasonality.png files into a 3 × 5 composite.

Usage
-----
$ python make_seasonality_grid.py
Creates seasonality_grid.png (≈15" × 20" @ 300 dpi) in the same folder.
"""

import glob, os, math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ------------------------------------------------------------------
png_files = sorted(glob.glob("*_seasonality.png"))
if len(png_files) != 13:
    raise RuntimeError(f"Expected 13 PNGs, found {len(png_files)}")

ncols = 3
nrows = math.ceil(len(png_files) / ncols)

fig, axes = plt.subplots(nrows, ncols,
                         figsize=(ncols * 5, nrows * 4),
                         dpi=300)

axes = axes.flatten()
for ax, fname in zip(axes, png_files):
    img = mpimg.imread(fname)
    ax.imshow(img)
    ax.set_axis_off()
    # short label from file name
    title = os.path.basename(fname)\
              .replace("_seasonality.png", "")\
              .replace("__", "_")\
              .replace("_", " ")
    ax.set_title(title, fontsize=7)

for ax in axes[len(png_files):]:      # hide any unused axes
    ax.set_visible(False)

fig.tight_layout()
fig.savefig("seasonality_grid.png", bbox_inches="tight")
print("✓ Saved composite to seasonality_grid.png")

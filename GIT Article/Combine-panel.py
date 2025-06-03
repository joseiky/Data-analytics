#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# List your six panel filenames and (optionally) titles:
files = [
    'WP6_scree.png',
    'WP6_biplot.png',
    'Figure6A.png',
    'Figure6B.png',
    'Figure6C.png',
    'Figure6D.png',
]
titles = [
    'Scree Plot',
    'PCA Biplot',
    '6A: Richness',
    '6B: Clusters',
    '6C: BV & Gender',
    '6D: Cluster × BV',
]

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
axes = axes.flatten()

for ax, fname, ttl in zip(axes, files, titles):
    img = mpimg.imread(fname)
    ax.imshow(img, aspect='equal')
    ax.axis('off')
    ax.set_title(ttl, fontsize=12)

plt.tight_layout(pad=1.0)
fig.savefig('Figure6_combined_2x3.png', dpi=300)

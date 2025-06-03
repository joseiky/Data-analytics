#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

files = ['Figure7A.png','Figure7B.png','Figure7C.png']
titles = ['7A: Prevalence','7B: Heatmap','7C: Stratified by Gender']

fig = plt.figure(figsize=(12, 10), dpi=300)
gs  = GridSpec(2, 2, figure=fig, height_ratios=[1,1])

# Top row
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
# Bottom row spans both columns
ax3 = fig.add_subplot(gs[1,:])

for ax, fname, ttl in zip([ax1, ax2, ax3], files, titles):
    img = mpimg.imread(fname)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(ttl, fontsize=12)

plt.tight_layout(pad=1.0)
fig.savefig('Figure7_combined_2x2.png', dpi=300)

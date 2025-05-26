import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# === CONFIGURATION ===
image_paths = [
    "Confusion_original.png",
    "Confusion_filtered.png",
    "ROC_original.png",
    "ROC_filtered.png",
    "SHAP_filtered.png",
    "LR_original_forestplot.png"  # optional: add forest plot here
]

titles = [
    "Confusion Matrix – Original",
    "Confusion Matrix – Filtered",
    "ROC Curve – Original",
    "ROC Curve – Filtered",
    "SHAP Summary – Filtered",
    "Forest Plot – Original (Optional)"
]

# Grid size
n_cols = 2
n_rows = (len(image_paths) + n_cols - 1) // n_cols  # automatically calculate

# === PLOT ===
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
axes = axes.flatten()

for idx, (img_path, title) in enumerate(zip(image_paths, titles)):
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
    else:
        axes[idx].text(0.5, 0.5, f"Image Not Found:\n{img_path}", ha='center', va='center', wrap=True)
        axes[idx].axis('off')

# Turn off unused axes
for j in range(idx + 1, len(axes)):
    axes[j].axis('off')

# Tight layout and save
plt.tight_layout()
output_path = "Sensitivity_All_Combined_Plots.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight')
print(f"✅ Combined figure saved to: {output_path}")
plt.show()

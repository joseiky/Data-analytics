import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
from PIL import Image

# === CONFIG ===
input_file = "Merged_ML_Outputs.xlsx"
temp_folder = "shap_temp"
output_path = "SHAP_Summary_Grid.png"
os.makedirs(temp_folder, exist_ok=True)

# === Load SHAP sheets ===
xls = pd.ExcelFile(input_file)
shap_sheets = [s for s in xls.sheet_names if "shap" in s]

# === Generate individual SHAP bar plots and save as temp PNGs ===
plot_paths = []
for i, sheet in enumerate(shap_sheets[:4]):
    df = xls.parse(sheet)
    if "y_true" not in df.columns:
        continue

    X = df.drop(columns="y_true").values
    feature_names = df.drop(columns="y_true").columns.tolist()
    title = sheet.replace("_shap", "")

    shap.summary_plot(
        X,
        features=X,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        plot_size=(8, 6)
    )
    file_path = os.path.join(temp_folder, f"{title}_bar.png")
    plt.title(f"SHAP Summary – {title}", fontsize=13)
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()
    plot_paths.append(file_path)

# === Combine all 4 SHAP plots into a 2x2 grid ===
images = [Image.open(p) for p in plot_paths]
widths, heights = zip(*(i.size for i in images))

# Assuming all images same size
single_w, single_h = widths[0], heights[0]
grid_img = Image.new('RGB', (2 * single_w, 2 * single_h), color='white')

# Paste into grid
positions = [(0, 0), (single_w, 0), (0, single_h), (single_w, single_h)]
for img, pos in zip(images, positions):
    grid_img.paste(img, pos)

# Save final grid
grid_img.save(output_path)

print(f"✅ Final SHAP summary grid saved to: {output_path}")

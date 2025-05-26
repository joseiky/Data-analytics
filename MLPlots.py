import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

# === CONFIG ===
input_file = "Merged_ML_Outputs.xlsx"
output_dir = "shap_grids"
os.makedirs(output_dir, exist_ok=True)

# === Load Excel File ===
xls = pd.ExcelFile(input_file)
shap_sheets = [s for s in xls.sheet_names if "shap" in s]

# === SHAP Bar Plot Grid ===
fig_bar, axes_bar = plt.subplots(2, 2, figsize=(16, 12))
axes_bar = axes_bar.flatten()

# === SHAP Beeswarm Plot Grid ===
fig_bee, axes_bee = plt.subplots(2, 2, figsize=(16, 12))
axes_bee = axes_bee.flatten()

# === Loop Through SHAP Sheets ===
for i, sheet in enumerate(shap_sheets[:4]):
    df = xls.parse(sheet)
    if "y_true" not in df.columns:
        continue

    X = df.drop(columns="y_true").values
    feature_names = df.drop(columns="y_true").columns.tolist()
    title = sheet.replace("_shap", "")

    # Bar plot (summary plot type = bar)
    plt.sca(axes_bar[i])
    shap.summary_plot(
        X,
        features=X,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        color_bar=True,
        plot_size=(6, 4)
    )
    axes_bar[i].set_title(f"SHAP Feature Importance – {title}", fontsize=12)

    # Beeswarm plot (summary plot type = default)
    plt.sca(axes_bee[i])
    shap.summary_plot(
        X,
        features=X,
        feature_names=feature_names,
        plot_type="dot",  # equivalent to beeswarm
        show=False,
        color_bar=True,
        plot_size=(6, 4)
    )
    axes_bee[i].set_title(f"SHAP Beeswarm – {title}", fontsize=12)

# === Save Both Grids ===
plt.figure(fig_bar.number)
fig_bar.tight_layout()
bar_path = os.path.join(output_dir, "SHAP_Summary_Bar_Grid.png")
fig_bar.savefig(bar_path, dpi=300)

plt.figure(fig_bee.number)
fig_bee.tight_layout()
bee_path = os.path.join(output_dir, "SHAP_Summary_Beeswarm_Grid.png")
fig_bee.savefig(bee_path, dpi=300)

print(f"✅ SHAP bar plot grid saved to: {bar_path}")
print(f"✅ SHAP beeswarm plot grid saved to: {bee_path}")

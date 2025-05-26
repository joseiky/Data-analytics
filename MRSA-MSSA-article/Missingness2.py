import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

# Load the cleaned, merged panel
df = pd.read_pickle("git_panel_clean.pkl")

# Filter to GIT-only panel if desired
df = df[df['panel'] == 'GIT'].copy()

# Optional sample if file is large
sampled_df = df.sample(n=500, random_state=42) if len(df) > 500 else df

# Plot
plt.figure(figsize=(16, 9))
msno.matrix(sampled_df, fontsize=12, sparkline=False)
plt.title("Missingness Matrix: GIT Panel Only", fontsize=16)
plt.tight_layout()
plt.savefig("FigureS1_Missingness_Matrix_GITPanelOnly.png", dpi=400)

print("✅ Saved: FigureS1_Missingness_Matrix_GITPanelOnly.png")

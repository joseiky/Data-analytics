import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel("GIT-No-toxins.xlsx")

# Standardize column names
df.columns = df.columns.str.strip().str.replace(r"[\s\-]+", "_", regex=True).str.lower()

# Replace known blank placeholders
df.replace(r"^\s*\\?N\s*$", pd.NA, regex=True, inplace=True)

# Optional: limit to first N rows if full rendering is too slow
sampled_df = df.sample(n=500, random_state=42) if len(df) > 500 else df

# Generate missingness matrix
plt.figure(figsize=(16, 9))
msno.matrix(sampled_df, fontsize=12, sparkline=False)
plt.title("Missingness Matrix: GIT-No-toxins Dataset", fontsize=16)
plt.tight_layout()
plt.savefig("FigureS1_Missingness_Matrix.png", dpi=400)

print("✅ Saved: FigureS1_Missingness_Matrix.png")

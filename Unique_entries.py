import pandas as pd

# ------------------------------------------------------------------
# File paths
INPUT_FILE  = "Encoded_Dataset.xlsx"      # adjust if necessary
OUTPUT_FILE = "Replace.xlsx"
# ------------------------------------------------------------------

# Read columns C and U as strings (avoids mixed‑type sort errors)
df = pd.read_excel(INPUT_FILE, usecols="C,U", dtype=str)
df.columns = ["source", "Pt_Ethnicity"]

# Clean whitespace
df["source"] = df["source"].str.strip()
df["Pt_Ethnicity"] = df["Pt_Ethnicity"].str.strip()

# Unique, sorted lists
unique_source    = pd.DataFrame({"Unique_Source":    sorted(df["source"].dropna().unique())})
unique_ethnicity = pd.DataFrame({"Unique_Ethnicity": sorted(df["Pt_Ethnicity"].dropna().unique())})

# Write each list to its own sheet
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    unique_source.to_excel(writer,    sheet_name="Unique_Source",    index=False)
    unique_ethnicity.to_excel(writer, sheet_name="Unique_Ethnicity", index=False)

print(f"Unique entries saved to '{OUTPUT_FILE}'.")

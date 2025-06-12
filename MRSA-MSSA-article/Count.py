import pandas as pd

# === Load and deduplicate ===
FILE = "Table S1. Dataset-MRSA-MSSA_deidentified.xlsx"
SHEET = 0  # First sheet

df = pd.read_excel(FILE, sheet_name=SHEET, dtype=str)
n_original = len(df)
df_nodup = df.drop_duplicates()
n_unique = len(df_nodup)

print(f"Original rows: {n_original}")
print(f"Unique rows after removing duplicates: {n_unique}\n")

# (Optional: save deduplicated data)
df_nodup.to_excel("deduplicated_rows.xlsx", index=False)

# === Analysis ===
test_types = [
    "S. aureus", "Methicillin resistance", "CA-MRSA; PVL DNA", "AST"
]

summary_rows = []

for test in test_types:
    df_test = df_nodup[df_nodup['Test'] == test]
    n_total = len(df_test)
    row = {"Test": test, "Total": n_total}

    if test == "AST":
        for result in ["AR", "AI", "AS"]:
            n_result = (df_test['Result'] == result).sum()
            pct = (n_result / n_total * 100) if n_total else 0
            row[result + "_count"] = n_result
            row[result + "_percent"] = pct
    else:
        for code, label in [("P", "Positive"), ("N", "Negative")]:
            n_result = (df_test['Result'] == code).sum()
            pct = (n_result / n_total * 100) if n_total else 0
            row[label + "_count"] = n_result
            row[label + "_percent"] = pct

    summary_rows.append(row)

# Convert to DataFrame for pretty print and export
summary_df = pd.DataFrame(summary_rows)

print("=== Summary Table ===")
print(summary_df.to_string(index=False))

summary_df.to_csv("test_summary.csv", index=False)
print("\n✓ Results exported to 'test_summary.csv'")

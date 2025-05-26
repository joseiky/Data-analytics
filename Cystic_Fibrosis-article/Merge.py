import os
import pandas as pd

input_folder = "ml_outputs"
output_file = "Feature_Outputs.xlsx"

# Create a writer object
writer = pd.ExcelWriter(output_file, engine="xlsxwriter")

# Loop through files
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        filepath = os.path.join(input_folder, filename)
        sheet_name = filename.replace(".csv", "")[:31]  # Excel sheet names max 31 chars
        try:
            df = pd.read_csv(filepath)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")

writer.close()
print(f"✅ All files merged into: {output_file}")

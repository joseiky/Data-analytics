import os
import shutil

# Define your project root and target subfolder
repo_root = '.'  # Use '.' if running from the repo folder, else specify full path
subfolder = 'Candida-infections'

# List of files to move (excluding folders and LICENSE/README)
files_to_move = [
    'Check-rows.py',
    'Inspect.py',
    'Replacement.py',
    'Replacement2.py',
    'Replacement3.py',
    'Undo_Replacement.py',
    'Unique.py',
    'WP1.py',
    'export_sheets.py',
    'merge_results.py'
]

# Move files
for filename in files_to_move:
    src = os.path.join(repo_root, filename)
    dst = os.path.join(repo_root, subfolder, filename)
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Moved {filename} to {subfolder}/")
    else:
        print(f"File {filename} not found in root.")

# Delete the file "Candida infections" (if it exists)
candida_file = os.path.join(repo_root, 'Candida infections')
if os.path.exists(candida_file):
    os.remove(candida_file)
    print("Deleted file 'Candida infections'")
else:
    print("File 'Candida infections' not found.")

print("Done.")

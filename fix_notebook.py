import json
import glob

for file_path in glob.glob("*.ipynb"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "markdown":
                cell.pop("execution_count", None)
                cell.pop("outputs", None)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)

        print(f"Fixed: {file_path}")

    except Exception as e:
        print(f"Error in {file_path}: {e}")

print("Done!")
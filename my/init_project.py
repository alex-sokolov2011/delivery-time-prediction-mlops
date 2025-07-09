import os

# Folders to create
folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src/prepare"
]

# Initial files to create with minimal content
files = {
    "Makefile": "",
    "README.md": "# Delivery Time Prediction MLOps\n",
    "src/prepare/__init__.py": "",
    "src/prepare/make_directories.py": "# Placeholder for directory creation logic\n",
    "src/prepare/merge_data.py": "# Placeholder for dataset merge logic\n",
    "src/prepare/split_data.py": "# Placeholder for train/validation split logic\n",
    "pyproject.toml": "# Placeholder for dependency configuration (optional)\n"
}

def create_structure():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")

    for filepath, content in files.items():
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Created file: {filepath}")

if __name__ == "__main__":
    create_structure()

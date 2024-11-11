import shutil
from pathlib import Path

def migrate_codebase():
    # Define the migration mappings
    migrations = {
        "backend/data/data_preprocessing.py": "backend/domain/services/preprocessing.py",
        "backend/models/prophet_model.py": "backend/domain/models/statistical/prophet.py",
        "backend/utils/ui_components.py": "backend/infrastructure/ui/components.py",
        "backend/utils/app_components.py": "backend/domain/services/forecasting.py"
    }

    # Create necessary directories
    for dest in migrations.values():
        Path(dest).parent.mkdir(parents=True, exist_ok=True)

    # Move files
    for src, dest in migrations.items():
        if Path(src).exists():
            shutil.move(src, dest)
            print(f"Moved {src} to {dest}")
        else:
            print(f"Warning: Source file {src} not found")

if __name__ == "__main__":
    migrate_codebase() 
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent

DATA_PATH = str(_PROJECT_ROOT / "data" / "raw" / "dataset_practica_final.csv")
PROCESSED_DATA_PATH = str(_PROJECT_ROOT / "data" / "processed" / "processed_dataset.csv")
MODEL_PATH = str(_PROJECT_ROOT / "models")
OUTPUT_PATH = str(_PROJECT_ROOT / "outputs")
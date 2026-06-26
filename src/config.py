from pathlib import Path

# Raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parents[1]

# Directorios principales
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "histopathologic-cancer-detection"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# Datos del concurso
TRAIN_DIR = RAW_DIR / "train"
TEST_DIR = RAW_DIR / "test"
TRAIN_LABELS_CSV = RAW_DIR / "train_labels.csv"
SAMPLE_SUBMISSION_CSV = RAW_DIR / "sample_submission.csv"

# Parámetros base
IMAGE_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_WORKERS = 4
RANDOM_SEED = 42

# Modelo / entrenamiento
MODEL_NAME = "custom_cnn"
DEVICE = "cuda"

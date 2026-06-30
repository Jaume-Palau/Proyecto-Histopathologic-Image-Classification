from pathlib import Path

# Raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parents[1]

# Directorios principales
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "histopathologic-cancer-detection"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
CHECKPOINTS = OUTPUTS_DIR / "checkpoints"
MODELS_DIR = OUTPUTS_DIR / "models"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"

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


################################################################################
## Hyperparameter Tuning: Version Base
################################################################################

config = {
    "lr": 0.003,
    "beta1": 0.9,
    "beta2": 0.9,
    "weight_decay": 0,
    "batch_size": 50,
    "dataloader_worker_count": 10,
    "dataloader_prefetch_factor": 5000,
    "name": "1a",
}


sweep_config = {
    "method": "bayes",

    "metric": {
        "name": "AUC-ROC",
        "goal": "maximize"
    },

    "parameters": {
        "name" : {
            "value" : "Prueba1"
        },

        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 5e-3
        },

        "beta1": {
            "distribution": "uniform",
            "min": 0.85,
            "max": 0.95
        },

        "beta2": {
            "distribution": "uniform",
            "min": 0.90,
            "max": 0.999
        },

        "weight_decay": {
            "values": [0, 1e-6, 1e-5, 1e-4, 1e-3]
        },

        "batch_size": {
            "values": [32, 50, 64, 96, 128]
        },

        "dataloader_worker_count": {
            "value": 10
        },

        "dataloader_prefetch_factor": {
            "value": 2
        },

        "epochs": {
            "value": 3
        }
    }
}
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed" / "whisper_embeddings"
OUTPUTS_DIR = BASE_DIR / "outputs"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"

# Raw Data Paths
TRAIN_CSV = RAW_DATA_DIR / "train.csv"
TEST_CSV = RAW_DATA_DIR / "test.csv"
TRAIN_AUDIO_DIR = RAW_DATA_DIR / "train"
TEST_AUDIO_DIR = RAW_DATA_DIR / "test"

# Processed Data Paths
WHISPER_TRAIN_CSV = PROCESSED_DATA_DIR / "whisper_train.csv"
WHISPER_TEST_CSV = PROCESSED_DATA_DIR / "whisper_test.csv"

# Output Paths
SUBMISSION_CSV = SUBMISSIONS_DIR / "submission_whisper_ensemble.csv"

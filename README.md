# Emotion Detection from Hindi Speech

This repository contains a high-performance Machine Learning pipeline designed to classify human emotion (Angry, Happy, Neutral, Sad) from raw Hindi speech audio. It was developed for the DataVerse Kaggle Competition and achieved a peak accuracy of **76.68%** on the leaderboard.

## The Challenge
Training deep neural networks on audio from scratch requires massive datasets. Because the provided dataset contained roughly 1,000 audio files, training custom architectures (like CNNs or LSTMs) from scratch led to overfitting. 

To solve this low-resource problem, this project completely bypasses traditional spectrogram training and utilizes **Transfer Learning** via pre-trained foundation models.

## The Pipeline

Our unified framework operates in two distinct stages:

### 1. Data Pre-Processing & Feature Extraction (OpenAI Whisper)
Instead of training an AI to understand audio, we used OpenAI's `Whisper-Base` model as a frozen feature extractor. 
* **Pre-processing:** Raw `.wav` files are loaded using SciPy, converted to Mono, resampled strictly to **16kHz**, and padded or truncated to exactly **30 seconds** (the required input window for Whisper).
* **Extraction:** The audio is passed through the Whisper-Base encoder. We ignore the text-generation decoder and instead extract the final **512-Dimensional hidden states**. These embeddings serve as a rich, mathematical fingerprint of the acoustic and semantic "vibe" of the audio.

### 2. The Machine Learning Ensemble
Once the audio is converted into 512D embeddings, the data is scaled using `StandardScaler` and fed into a diverse "Board of Directors" consisting of three distinct Machine Learning models:
* **Logistic Regression:** Configured with `class_weight='balanced'` to heavily penalize mistakes on rare minority classes (like Neutral).
* **Support Vector Machine (SVM):** Configured with an RBF (Radial Basis Function) Kernel to draw complex, non-linear boundaries between the emotion embeddings.
* **XGBoost:** A gradient boosting classifier to capture complex decision-tree logic.

### 3. Soft-Voting
Instead of relying on a single model or using "Hard Voting" (majority rules), this pipeline utilizes **Soft-Voting**. Each of the three models outputs a percentage confidence for every emotion (e.g., 80% Sad, 20% Angry). The pipeline averages these probabilities together to make the final prediction. This ensures that a highly confident model can override two models that are only slightly confident, maximizing accuracy.

## Getting Started

### Installation
Install the required libraries (PyTorch, Transformers, XGBoost, Scikit-Learn, etc.):
```bash
pip install -r requirements.txt
```

### Running the Code
1. Place your raw audio files in `data/raw/train/` and `data/raw/test/`. Ensure `train.csv` and `test.csv` are placed in `data/raw/`.
2. Extract the embeddings using Whisper:
```bash
python src/data/extract_embeddings.py
```
3. Train the ensemble and generate predictions:
```bash
python src/models/train_ensemble.py
```

import os
import pandas as pd
import numpy as np
import torch
import torchaudio
from scipy.io import wavfile
from tqdm import tqdm
from transformers import WhisperModel, WhisperFeatureExtractor

# Import central configuration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def main():
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Using device: {device} for Whisper Extraction")

    print("Loading Whisper-Base...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
    model = WhisperModel.from_pretrained("openai/whisper-base").to(device)
    model.eval()

    def get_embedding(audio_path):
        sr, audio_array = wavfile.read(audio_path)
        
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
            audio_array = audio_array / np.max(np.abs(audio_array))
            
        if len(audio_array.shape) > 1:
            audio_array = audio_array[:, 0]
            
        speech = audio_array

        if sr != 16000:
            speech_tensor = torch.from_numpy(speech).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            speech = resampler(speech_tensor)[0].numpy()
        
        inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        
        with torch.no_grad():
            encoder_outputs = model.encoder(input_features)
            hidden_states = encoder_outputs.last_hidden_state
            
        embedding = hidden_states.mean(dim=1).cpu().numpy()[0]
        return embedding

    def process_dataset(csv_path, audio_dir, is_test=False):
        df = pd.read_csv(csv_path)
        records = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            fpath = os.path.join(audio_dir, row['filename'])
            if not os.path.exists(fpath):
                print(f"File missing: {fpath}")
                continue
            
            emb = get_embedding(fpath)
            feat_dict = {f"emb_{i}": val for i, val in enumerate(emb)}
            feat_dict['Id'] = row['Id']
            if not is_test:
                feat_dict['emotion'] = row['emotion']
            
            records.append(feat_dict)
            
        return pd.DataFrame(records)

    print("\n--- Extracting Train Embeddings ---")
    train_feats = process_dataset(config.TRAIN_CSV, config.TRAIN_AUDIO_DIR, is_test=False)
    train_feats.to_csv(config.WHISPER_TRAIN_CSV, index=False)
    print(f"Train Shape: {train_feats.shape}")

    print("\n--- Extracting Test Embeddings ---")
    test_feats = process_dataset(config.TEST_CSV, config.TEST_AUDIO_DIR, is_test=True)
    test_feats.to_csv(config.WHISPER_TEST_CSV, index=False)
    print(f"Test Shape: {test_feats.shape}")

    print("\n✅ Deep Semantic Embeddings Extracted Successfully!")

if __name__ == "__main__":
    main()

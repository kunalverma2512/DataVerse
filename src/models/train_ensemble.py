import os
import sys
import pandas as pd
import numpy as np
import warnings
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

warnings.filterwarnings('ignore')

# Import central configuration
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def main():
    os.makedirs(config.SUBMISSIONS_DIR, exist_ok=True)
    os.makedirs(config.OUTPUTS_DIR / "models", exist_ok=True)
    
    print("Loading Whisper Embeddings...")
    train_df = pd.read_csv(config.WHISPER_TRAIN_CSV)
    test_df = pd.read_csv(config.WHISPER_TEST_CSV)
    
    feature_cols = [c for c in train_df.columns if c.startswith('emb_')]
    print(f"Loaded {len(feature_cols)} Semantic Dimensions from Whisper.")
    
    X = train_df[feature_cols].values
    y_raw = train_df['emotion'].values
    
    X_test = test_df[feature_cols].values
    test_ids = test_df['Id'].values
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)
    
    models = {
        "SVM (RBF Kernel)": SVC(kernel='rbf', C=5.0, probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000, C=0.5, random_state=42, class_weight='balanced'),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=4, random_state=42)
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = {m: np.zeros((len(X), len(le.classes_))) for m in models}
    test_preds = {m: np.zeros((len(X_test), len(le.classes_))) for m in models}
    
    print("\n--- Starting 5-Fold Cross Validation ---")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        for m_name, model in models.items():
            model.fit(X_train, y_train)
            oof_preds[m_name][val_idx] = model.predict_proba(X_val)
            test_preds[m_name] += model.predict_proba(X_test) / skf.n_splits
            
    print("\n--- Cross-Validation Results ---")
    ensemble_oof = np.zeros_like(oof_preds["SVM (RBF Kernel)"])
    for m_name in models:
        preds_class = np.argmax(oof_preds[m_name], axis=1)
        score = f1_score(y, preds_class, average='macro')
        print(f"{m_name} F1-Macro: {score:.4f}")
        ensemble_oof += oof_preds[m_name] / len(models)
        
    ensemble_class = np.argmax(ensemble_oof, axis=1)
    final_score = f1_score(y, ensemble_class, average='macro')
    print(f"\n🌟 Mega Ensemble OOF F1-Macro: {final_score:.4f}")
    
    print("\nClassification Report (Ensemble):")
    print(classification_report(y, ensemble_class, target_names=le.classes_))
    
    print("Generating Final Predictions for Hidden Test Set...")
    ensemble_test = np.zeros_like(test_preds["SVM (RBF Kernel)"])
    for m_name in models:
        ensemble_test += test_preds[m_name] / len(models)
        
    final_test_classes = np.argmax(ensemble_test, axis=1)
    final_test_labels = le.inverse_transform(final_test_classes)
    
    submission = pd.DataFrame({'Id': test_ids, 'emotion': final_test_labels})
    submission.to_csv(config.SUBMISSION_CSV, index=False)
    print(f"\n✅ Submission saved to: {config.SUBMISSION_CSV}")

    print("\n--- Training Final Models on FULL Dataset for Production ---")
    for m_name, model in models.items():
        model.fit(X, y)
        safe_name = m_name.replace(' ', '_').replace('(', '').replace(')', '')
        model_path = config.OUTPUTS_DIR / f"models/{safe_name}_final.pkl"
        joblib.dump(model, model_path)
        
    joblib.dump(scaler, config.OUTPUTS_DIR / "models/scaler.pkl")
    joblib.dump(le, config.OUTPUTS_DIR / "models/label_encoder.pkl")
    print("✅ All final models and scalers successfully saved to outputs/models/!")

if __name__ == "__main__":
    main()

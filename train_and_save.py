"""
train_and_save.py
Trains a robust prosodic classifier and saves the weights to model.pkl.
"""
import argparse
import csv
import os
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from features import load_wav, speech_before, frame_energy_db, f0_contour

def extract_features(x, sr, pause_start):
    # Extract 1.5s of speech context preceding the pause
    seg = speech_before(x, sr, pause_start, window_s=1.5)
    if len(seg) < sr // 10:
        return np.zeros(6, dtype=np.float32)
    
    e = frame_energy_db(seg, sr)
    f0 = f0_contour(seg, sr)
    voiced = f0[f0 > 0]
    
    # 1. Energy Context & Decay
    final_e = e[-5:].mean() if len(e) >= 5 else -120.0
    prior_e = e[:-5].mean() if len(e) > 5 else -120.0
    energy_drop = final_e - prior_e  # Negative value means falling volume
    
    # 2. Pitch Contour Slopes
    if len(voiced) >= 4:
        pitch_final = voiced[-3:].mean()
        pitch_trend = voiced[-10:].mean() - voiced[:10].mean()
        # Linear slope of the last 5 voiced frames
        pitch_slope = np.polyfit(np.arange(5), voiced[-5:], 1)[0] if len(voiced) >= 5 else 0.0
    else:
        pitch_final = 0.0
        pitch_trend = 0.0
        pitch_slope = 0.0
        
    # 3. Time Duration Context
    context_duration = len(seg) / sr

    return np.array([
        final_e, 
        energy_drop, 
        pitch_final, 
        pitch_trend, 
        pitch_slope, 
        context_duration
    ], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))
    cache = {}
    X, y = [], []
    
    print("Extracting training features...")
    for r in rows:
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        X.append(extract_features(x, sr, float(r["pause_start"])))
        y.append(1 if r["label"] == "eot" else 0)
        
    X, y = np.array(X), np.array(y)
    
    # Using Gradient Boosting for non-linear boundary rules (CPU friendly)
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    clf.fit(X, y)
    
    # Save the trained model Artifact
    with open("model.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("Model successfully trained and serialized to 'model.pkl'.")

if __name__ == "__main__":
    main()

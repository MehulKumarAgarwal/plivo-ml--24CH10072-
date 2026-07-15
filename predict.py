"""Inference script for EOT detection.
Loads a pre-trained model and predicts on unseen data.

Usage:
    python predict.py --data_dir eot_data/english --out predictions.csv
"""
import argparse
import csv
import os
import numpy as np
import joblib

from features import load_wav, speech_before, frame_energy_db, f0_contour

def extract_features(x, sr, pause_start):
    """Features from audio STRICTLY BEFORE pause_start."""
    seg = speech_before(x, sr, pause_start, window_s=1.5)
    
    # If the segment is too short, return zeros for all 5 features
    if len(seg) < sr // 10:
        return np.zeros(5, dtype=np.float32)
        
    e = frame_energy_db(seg, sr)
    f0 = f0_contour(seg, sr)
    voiced = f0[f0 > 0]
    
    # Feature 1: Energy Decay (Difference between very end and slightly before)
    e_end = e[-10:].mean() if len(e) >= 10 else 0
    e_mid = e[-30:-10].mean() if len(e) >= 30 else e_end
    energy_diff = e_end - e_mid
    
    # Feature 2: Pitch (F0) Slope over the last 10 voiced frames
    if len(voiced) > 5:
        x_vals = np.arange(len(voiced[-10:]))
        slope, _ = np.polyfit(x_vals, voiced[-10:], 1) 
    else:
        slope = 0.0
        
    # Feature 3: Final Pitch mean
    final_pitch = voiced[-3:].mean() if len(voiced) >= 3 else 0.0
    
    # Feature 4: Voicing Fraction (How much of the window is actual speech vs silence)
    voicing_fraction = len(voiced) / max(1, len(f0))
    
    return np.array([
        e[-5:].mean(),        
        energy_diff,          
        slope,                
        final_pitch,          
        voicing_fraction      
    ], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    # 1. Load the trained pipeline (Scaler + RandomForest)
    try:
        clf = joblib.load("eot_model.joblib")
    except FileNotFoundError:
        print("Error: eot_model.joblib not found. Run train.py first to save the model.")
        return

    # 2. Read the new data's labels.csv
    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))
    cache = {}
    
    # 3. Open output file and write headers
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        
        # 4. Iterate over pauses, extract features, and predict
        for r in rows:
            path = os.path.join(args.data_dir, r["audio_file"])
            if path not in cache:
                cache[path] = load_wav(path)
            x, sr = cache[path]
            
            # Extract features for this single pause
            feats = extract_features(x, sr, float(r["pause_start"]))
            
            # .predict_proba expects a 2D array, so we reshape the 1D feature array
            feats = feats.reshape(1, -1) 
            
            # Get the probability of class 1 (end of turn)
            p_eot = clf.predict_proba(feats)[0, 1]
            
            w.writerow([r["turn_id"], r["pause_index"], f"{p_eot:.4f}"])
            
    print(f"wrote {len(rows)} predictions -> {args.out}")

if __name__ == "__main__":
    main()

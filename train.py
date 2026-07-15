"""Skeleton: prosodic features + classifier. Runs as-is, scores poorly ON
PURPOSE. Your hour goes into extract_features() and what you learn from
your errors.

    python train.py --data_dir eot_data/english --out predictions.csv
"""
import argparse
import csv
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib  # Needed to save the model for predict.py
from sklearn.model_selection import GroupShuffleSplit

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
    energy_diff = e_end - e_mid # Negative means trailing off
    
    # Feature 2: Pitch (F0) Slope over the last 10 voiced frames
    if len(voiced) > 5:
        # np.polyfit returns [slope, intercept]
        x_vals = np.arange(len(voiced[-10:]))
        slope, _ = np.polyfit(x_vals, voiced[-10:], 1) 
    else:
        slope = 0.0
        
    # Feature 3: Final Pitch mean
    final_pitch = voiced[-3:].mean() if len(voiced) >= 3 else 0.0
    
    # Feature 4: Voicing Fraction (How much of the window is actual speech vs silence)
    voicing_fraction = len(voiced) / max(1, len(f0))
    
    return np.array([
        e[-5:].mean(),        # Starter: Absolute final energy
        energy_diff,          # Hypothesis: Trailing energy
        slope,                # Hypothesis: Pitch slope (flat/rising vs falling)
        final_pitch,          # Starter: Final pitch value
        voicing_fraction      # Hypothesis: density of speech
    ], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    # Load the labels and extract features
    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))
    cache = {}
    X, y, groups, keys = [], [], [], []
    for r in rows:
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        X.append(extract_features(x, sr, float(r["pause_start"])))
        y.append(1 if r["label"] == "eot" else 0)
        groups.append(r["turn_id"])
        keys.append((r["turn_id"], r["pause_index"]))
    X, y = np.array(X), np.array(y)

    # Sanity check on held-out TURNS
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
                  .split(X, y, groups))
                  
    # Upgrade: Using a Pipeline with a Scaler and a Random Forest
    clf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42)
        # You can also comment out the line above and try this SVM instead:
        # SVC(probability=True, class_weight="balanced", random_state=42)
    )
    
    clf.fit(X[tr], y[tr])
    print(f"held-out turn accuracy: {clf.score(X[te], y[te]):.3f} "
          f"(chance ~ {max(np.mean(y), 1-np.mean(y)):.3f})")

    # Refit on everything, write predictions for the scorer
    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pi_p in zip(keys, p):
            w.writerow([tid, pi, f"{pi_p:.4f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")
    
    # Save the trained model so predict.py can load it
    joblib.dump(clf, "eot_model.joblib")
    print("Saved trained model to eot_model.joblib")


if __name__ == "__main__":
    main()

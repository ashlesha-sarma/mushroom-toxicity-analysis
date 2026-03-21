# src/evaluate.py
# Full evaluation report on the test set.
# Run:  python src/evaluate.py

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_data, preprocess, get_splits

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH    = os.path.join(PROJECT_ROOT, "models", "model.pkl")
RESULTS_DIR   = os.path.join(PROJECT_ROOT, "results")
CM_PATH       = os.path.join(RESULTS_DIR,  "confusion_matrix.png")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("  🍄  Mushroom Classifier — Evaluation Report (2023 Dataset)")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print("\n❌ No model found. Run:  python src/train_model.py\n")
        sys.exit(1)

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    preprocessor = bundle["preprocessor"]
    rf           = bundle["rf"]

    print("\n📦 Loading test data...")
    df = load_data()
    X_raw, _, y, _, _ = preprocess(df)
    _, X_raw_test, _, y_test = get_splits(X_raw, y)

    X_test_encoded = preprocessor.transform(X_raw_test).astype("float32")
    y_pred  = rf.predict(X_test_encoded)
    y_proba = rf.predict_proba(X_test_encoded)[:, 1]

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Edible", "Poisonous"]))

    auc = roc_auc_score(y_test, y_proba)
    print(f"  ROC-AUC Score: {auc:.4f}  (1.0 = perfect)")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix Breakdown:")
    print(f"  True  Negatives : {tn:>6,}  ✅ edible correctly identified")
    print(f"  False Positives : {fp:>6,}  🟡 edible called poisonous (overcautious)")
    print(f"  False Negatives : {fn:>6,}  ☠️  poisonous called edible  ← want 0!")
    print(f"  True  Positives : {tp:>6,}  ✅ poisonous correctly identified")

    if fn == 0:
        print("\n  🎉 Zero False Negatives!")
    else:
        print(f"\n  ⚠️  {fn} poisonous mushrooms were misclassified as edible.")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Edible","Poisonous"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title("Confusion Matrix — 2023 Secondary Mushroom Dataset",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    plt.close()
    print(f"\n  📊 Confusion matrix → results/confusion_matrix.png")
    print("=" * 60)


if __name__ == "__main__":
    main()

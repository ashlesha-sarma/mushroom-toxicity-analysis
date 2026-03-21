# src/train_model.py
# ─────────────────────────────────────────────────────────────────────────────
# Trains the Random Forest on the 2023 UCI Secondary Mushroom Dataset.
#
#   - Saves model and preprocessor separately but in one dictionary
#     Prediction only needs raw feature values by running them through the preprocessor first.
#   - Handles both numeric AND categorical features correctly
#   - No odor feature = no shortcut learning, more realistic patterns
#   - Reports Recall as the primary metric (safety-critical classification)
#   - Saves feature importance plot with the top 20 most impactful features
#
# Run:
#   python src/train_model.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, confusion_matrix, classification_report
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import (
    load_data, preprocess, get_splits,
    NUMERIC_COLS, CATEGORICAL_COLS, ALL_FEATURE_COLS, FEATURE_OPTIONS
)


# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR      = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR     = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH      = os.path.join(MODELS_DIR,  "model.pkl")
METADATA_PATH   = os.path.join(MODELS_DIR,  "metadata.pkl")
IMPORTANCE_PATH = os.path.join(RESULTS_DIR, "feature_importance.png")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
RF_N_ESTIMATORS = 200   # More trees = more stable predictions on larger dataset
RF_MAX_DEPTH    = None  # Let trees grow fully (dataset is complex enough)
RF_MIN_SAMPLES  = 2     # Minimum samples per leaf
RF_RANDOM_STATE = 42


# ── Feature importance plot ────────────────────────────────────────────────────

def plot_feature_importance(rf_model, feature_names_after_ohe, top_n=20):
    """
    Extracts and plots the top N most important features from the Random Forest.

    Because OneHotEncoding splits one column into many binary columns,
    we show the OHE column names (e.g. 'cap-color_n' means cap-color=brown).
    This gives a more granular view of which specific values matter most.
    """
    importances = rf_model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]

    top_names   = [feature_names_after_ohe[i] for i in indices]
    top_values  = [importances[i]             for i in indices]

    # Color by feature group for readability
    def color_for(name):
        if any(name.startswith(n) for n in ["cap-diameter","stem-height","stem-width"]):
            return "#2d6a4f"   # dark green = numeric
        elif "gill" in name:
            return "#1d6fa4"   # blue = gill features
        elif "cap" in name:
            return "#74c69d"   # light green = cap features
        elif "stem" in name:
            return "#b7e4c7"   # pale green = stem features
        else:
            return "#95a5a6"   # gray = other

    colors = [color_for(n) for n in top_names]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(
        top_names[::-1], top_values[::-1],
        color=colors[::-1], edgecolor="white", linewidth=0.4
    )

    ax.set_xlabel("Feature Importance (mean decrease in impurity)", fontsize=11)
    ax.set_title(
        f"Top {top_n} Most Important Features — 2023 Secondary Mushroom Dataset\n"
        f"(No odor feature — model learns real biological patterns)",
        fontsize=12, fontweight="bold"
    )
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("#f9f9f9")

    for bar, val in zip(bars, top_values[::-1]):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#2d6a4f", label="Numeric measurements"),
        Patch(color="#1d6fa4", label="Gill features"),
        Patch(color="#74c69d", label="Cap features"),
        Patch(color="#b7e4c7", label="Stem features"),
        Patch(color="#95a5a6", label="Other features"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(IMPORTANCE_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Feature importance plot → results/feature_importance.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  🍄  Mushroom Toxicity Classifier — Training (2023 Dataset)")
    print("=" * 65)

    # ── 1. Load and preprocess ────────────────────────────────
    print("\n📦 Loading data...")
    df = load_data()

    print("\n🔧 Preprocessing...")
    X_raw, X, y, preprocessor, feature_names = preprocess(df)
    X_train, X_test, y_train, y_test = get_splits(X, y)

    print(f"\n  Train : {X_train.shape[0]:,} samples")
    print(f"  Test  : {X_test.shape[0]:,} samples")
    print(f"  Features after encoding: {X_train.shape[1]}")

    # ── 2. Train Random Forest ────────────────────────────────
    # We fit the RF model on the preprocessed training data.
    # We will later save both the preprocessor and the rf model to model.pkl.

    print(f"\n🌲 Training Random Forest ({RF_N_ESTIMATORS} trees)...")
    rf = RandomForestClassifier(
        n_estimators = RF_N_ESTIMATORS,
        max_depth    = RF_MAX_DEPTH,
        min_samples_leaf = RF_MIN_SAMPLES,
        random_state = RF_RANDOM_STATE,
        n_jobs       = -1,   # use all CPU cores
        class_weight = "balanced",  # compensates for any class imbalance
    )
    rf.fit(X_train, y_train)

    # ── 3. Evaluate ───────────────────────────────────────────
    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)

    acc       = accuracy_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    conf_scores    = y_proba.max(axis=1) * 100
    avg_confidence = conf_scores.mean()

    print(f"\n  ── Results on {X_test.shape[0]:,} test samples ──")
    print(f"  {'Accuracy'      :<22} {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  {'Recall'        :<22} {recall:.4f}  ← % poisonous correctly caught")
    print(f"  {'Precision'     :<22} {precision:.4f}")
    print(f"  {'F1 Score'      :<22} {f1:.4f}")
    print(f"  {'Avg Confidence':<22} {avg_confidence:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix:")
    print(f"  True  Negatives (edible   → edible)    : {tn:>6,}")
    print(f"  False Positives (edible   → poisonous) : {fp:>6,}  (overcautious)")
    print(f"  False Negatives (poisonous→ edible)    : {fn:>6,}  ← want this = 0!")
    print(f"  True  Positives (poisonous→ poisonous) : {tp:>6,}")

    if fn == 0:
        print("\n  🎉 Zero False Negatives — no poisonous mushroom was missed!")
    else:
        print(f"\n  ⚠️  {fn} poisonous mushrooms were misclassified as edible.")

    print("\n  Full classification report:")
    print(classification_report(y_test, y_pred, target_names=["Edible","Poisonous"]))

    # ── 4. Feature importance plot ────────────────────────────
    plot_feature_importance(rf, feature_names)

    # ── 5. Save model bundle ────────────────────────────────
    # We save (preprocessor + rf) together as a dict.
    # At prediction time, app.py calls:
    #   X_enc = preprocessor.transform(raw_df)
    #   proba = rf.predict_proba(X_enc)
    # No manual encoding needed anywhere else.
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"preprocessor": preprocessor, "rf": rf}, f)

    with open(METADATA_PATH, "wb") as f:
        pickle.dump({
            "feature_names":   feature_names,
            "numeric_cols":    NUMERIC_COLS,
            "categorical_cols": CATEGORICAL_COLS,
            "all_feature_cols": ALL_FEATURE_COLS,
            "feature_options": FEATURE_OPTIONS,
        }, f)

    print(f"\n  💾 Model saved    → models/model.pkl")
    print(f"  💾 Metadata saved → models/metadata.pkl")
    print("\n✅ Training complete! Run:  python app.py")
    print("=" * 65)


if __name__ == "__main__":
    main()

# src/preprocess.py
# ─────────────────────────────────────────────────────────────────────────────
# Downloads and preprocesses the 2023 UCI Secondary Mushroom Dataset.
#
# Why this dataset is better than the 1987 one:
#   - 61,069 samples across 173 species (vs 8,124 across 23)
#   - Covers Death Cap, Destroying Angel and other deadly species
#   - No odor feature — model can't cheat via shortcut learning
#   - Has real numeric measurements: cap-diameter, stem-height, stem-width
#   - More realistic missing value patterns
#
# What this file does:
#   1. Downloads secondary_data_shuffled.csv from the UCI/GitHub source
#   2. Saves it to data/mushrooms.csv
#   3. Handles missing values separately per column type
#   4. Builds a sklearn ColumnTransformer:
#        - StandardScaler   for the 3 numeric columns
#        - OneHotEncoder     for the 17 categorical columns
#      This is the correct approach when feature types differ.
#   5. Returns the fitted preprocessor + train/test splits
#
# Run directly to verify:
#   python src/preprocess.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "mushrooms.csv")

# Primary download source (UCI via GitHub mirror — most reliable)
DATA_URL = (
    "https://raw.githubusercontent.com/ghattab/secondarydata/"
    "main/data/secondary_data_shuffled.csv"
)
# Backup URL from UCI directly
DATA_URL_BACKUP = (
    "https://archive.ics.uci.edu/static/public/848/"
    "secondary+mushroom+dataset.zip"
)


# ── Column definitions ────────────────────────────────────────────────────────

# The 3 numeric (metrical) features — need StandardScaler
NUMERIC_COLS = [
    "cap-diameter",   # float, cm
    "stem-height",    # float, cm
    "stem-width",     # float, mm
]

# The 17 categorical (nominal) features — need OneHotEncoding
CATEGORICAL_COLS = [
    "cap-shape",
    "cap-surface",
    "cap-color",
    "does-bruise-or-bleed",
    "gill-attachment",
    "gill-spacing",
    "gill-color",
    "stem-root",
    "stem-surface",
    "stem-color",
    "veil-type",
    "veil-color",
    "has-ring",
    "ring-type",
    "spore-print-color",
    "habitat",
    "season",
]

TARGET_COL = "class"

# All 20 feature columns in dataset order
ALL_FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# ── Valid category codes for each nominal feature ─────────────────────────────
# Used by app.py for dropdowns and by Gemini prompt generation.
# Source: UCI Secondary Mushroom Dataset documentation
FEATURE_OPTIONS = {
    "cap-shape": {
        "b": "Bell", "c": "Conical", "x": "Convex", "f": "Flat",
        "s": "Sunken", "p": "Spherical", "o": "Others"
    },
    "cap-surface": {
        "i": "Fibrous", "g": "Grooves", "y": "Scaly", "s": "Smooth",
        "h": "Shiny", "l": "Leathery", "k": "Silky",
        "t": "Sticky", "w": "Wrinkled", "e": "Fleshy"
    },
    "cap-color": {
        "n": "Brown", "b": "Buff",   "g": "Gray",   "r": "Green",
        "p": "Pink",  "u": "Purple", "e": "Red",    "w": "White",
        "y": "Yellow","l": "Blue",   "o": "Orange", "k": "Black"
    },
    "does-bruise-or-bleed": {
        "t": "Yes (bruises/bleeds)", "f": "No"
    },
    "gill-attachment": {
        "a": "Adnate",   "x": "Adnexed",  "d": "Decurrent",
        "e": "Free",     "s": "Sinuate",  "p": "Pores", "f": "None"
    },
    "gill-spacing": {
        "c": "Close", "d": "Distant", "f": "None"
    },
    "gill-color": {
        "n": "Brown", "b": "Buff",   "g": "Gray",   "r": "Green",
        "p": "Pink",  "u": "Purple", "e": "Red",    "w": "White",
        "y": "Yellow","l": "Blue",   "o": "Orange", "k": "Black", "f": "None"
    },
    "stem-root": {
        "b": "Bulbous", "s": "Swollen", "c": "Club",
        "u": "Cup",     "e": "Equal",   "z": "Rhizomorphs", "r": "Rooted"
    },
    "stem-surface": {
        "i": "Fibrous", "g": "Grooves", "y": "Scaly", "s": "Smooth",
        "h": "Shiny",   "l": "Leathery","k": "Silky",
        "t": "Sticky",  "w": "Wrinkled","e": "Fleshy", "f": "None"
    },
    "stem-color": {
        "n": "Brown", "b": "Buff",   "g": "Gray",   "r": "Green",
        "p": "Pink",  "u": "Purple", "e": "Red",    "w": "White",
        "y": "Yellow","l": "Blue",   "o": "Orange", "k": "Black", "f": "None"
    },
    "veil-type": {
        "p": "Partial", "u": "Universal"
    },
    "veil-color": {
        "n": "Brown", "b": "Buff",   "g": "Gray",   "r": "Green",
        "p": "Pink",  "u": "Purple", "e": "Red",    "w": "White",
        "y": "Yellow","l": "Blue",   "o": "Orange", "k": "Black", "f": "None"
    },
    "has-ring": {
        "t": "Yes", "f": "No"
    },
    "ring-type": {
        "c": "Cobwebby",  "e": "Evanescent", "r": "Flaring",
        "g": "Grooved",   "l": "Large",      "p": "Pendant",
        "s": "Sheathing", "z": "Zone",       "y": "Scaly",
        "m": "Movable",   "f": "None"
    },
    "spore-print-color": {
        "n": "Brown", "b": "Buff",   "g": "Gray",   "r": "Green",
        "p": "Pink",  "u": "Purple", "e": "Red",    "w": "White",
        "y": "Yellow","l": "Blue",   "o": "Orange", "k": "Black"
    },
    "habitat": {
        "g": "Grasses", "l": "Leaves",  "m": "Meadows",
        "p": "Paths",   "h": "Heaths",  "u": "Urban",
        "w": "Waste",   "d": "Woods"
    },
    "season": {
        "s": "Spring", "u": "Summer", "a": "Autumn", "w": "Winter"
    },
}


# ── Download ──────────────────────────────────────────────────────────────────

def download_and_save():
    """
    Downloads the 2023 UCI Secondary Mushroom Dataset and saves it.
    The file uses ';' as delimiter (not comma) — pandas handles this.
    """
    print("📥 Downloading UCI Secondary Mushroom Dataset (2023)...")
    print(f"   Source: {DATA_URL}")

    try:
        df = pd.read_csv(DATA_URL, sep=";")
        print(f"   ✅ Downloaded {len(df):,} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"   ⚠️  Primary URL failed: {e}")
        print("   Trying backup source via ucimlrepo...")
        try:
            from ucimlrepo import fetch_ucirepo
            repo  = fetch_ucirepo(id=848)
            X_raw = repo.data.features
            y_raw = repo.data.targets
            df    = pd.concat([y_raw, X_raw], axis=1)
            df.rename(columns={df.columns[0]: "class"}, inplace=True)
            print(f"   ✅ Downloaded via ucimlrepo: {len(df):,} rows")
        except Exception as e2:
            print(f"   ❌ Both sources failed: {e2}")
            sys.exit(1)

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"   💾 Saved → data/mushrooms.csv")
    return df


def load_data():
    """Loads from saved CSV if it exists, otherwise downloads first."""
    if os.path.exists(DATA_PATH):
        print(f"📂 Loading data/mushrooms.csv...")
        df = pd.read_csv(DATA_PATH)
        print(f"   {len(df):,} rows loaded")
        return df
    return download_and_save()


# ── Preprocessing ──────────────────────────────────────────────────────────────

def preprocess(df):
    """
    Prepares the raw DataFrame for model training.

    Returns:
        X_raw  pd.DataFrame  — raw features (before encoding), kept for reference
        y      np.ndarray    — labels: 0=edible, 1=poisonous
        preprocessor         — fitted sklearn ColumnTransformer
        feature_names list   — output column names after transformation
    """
    df = df.copy()

    # ── Step 1: Encode the target ─────────────────────────────
    # 'e' (edible) → 0,  'p' (poisonous) → 1
    le = LabelEncoder()
    y  = le.fit_transform(df[TARGET_COL]).astype("int64")
    print(f"   Target: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    X_raw = df[ALL_FEATURE_COLS].copy()

    # ── Step 2: Handle '?' missing value markers ──────────────
    # The 2023 dataset uses '?' for unknown values in some columns.
    # Replace with NaN so sklearn imputers can handle them.
    X_raw.replace("?", np.nan, inplace=True)

    # ── Step 3: Report missing values ────────────────────────
    missing = X_raw.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"\n   Missing value summary:")
        for col, cnt in missing.items():
            print(f"     {col:<25} {cnt:>6,} missing ({cnt/len(X_raw)*100:.1f}%)")

    # ── Step 4: Build ColumnTransformer ──────────────────────
    # This is the key upgrade over the 1987 project.
    # Different columns need different transformations:
    #
    #   Numeric:      fill missing with median → StandardScaler
    #   Categorical:  fill missing with 'unknown' → OneHotEncoder
    #
    # Pipeline chains these steps so they run in order.

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),   # fill NaN with median
        ("scaler",  StandardScaler()),                   # mean=0, std=1
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline,         NUMERIC_COLS),
        ("cat", categorical_pipeline,     CATEGORICAL_COLS),
    ])

    # ── Step 5: Fit and transform ─────────────────────────────
    X_transformed = preprocessor.fit_transform(X_raw)

    # ── Step 6: Recover output feature names ─────────────────
    cat_feature_names = (
        preprocessor
        .named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(CATEGORICAL_COLS)
        .tolist()
    )
    feature_names = NUMERIC_COLS + cat_feature_names

    print(f"\n   Input features  : {len(ALL_FEATURE_COLS)}")
    print(f"   After encoding  : {len(feature_names)} columns")
    print(f"     - Numeric (scaled)  : {len(NUMERIC_COLS)}")
    print(f"     - Categorical (OHE) : {len(cat_feature_names)}")
    print(f"   Edible: {(y==0).sum():,}  |  Poisonous: {(y==1).sum():,}")

    return X_raw, X_transformed.astype("float32"), y, preprocessor, feature_names


def get_splits(X, y, test_size=0.2, random_state=42):
    """
    80/20 train-test split.
    stratify=y ensures both sets maintain the same class balance.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Preprocessing Pipeline — Self Test")
    print("=" * 60)

    df = download_and_save()
    print(f"\nColumns: {list(df.columns)}")
    print(f"Class distribution:\n{df['class'].value_counts()}")

    X_raw, X, y, preprocessor, feature_names = preprocess(df)
    X_train, X_test, y_train, y_test = get_splits(X, y)

    print(f"\n  Train : {X_train.shape}")
    print(f"  Test  : {X_test.shape}")
    print(f"\n  First 5 feature names: {feature_names[:5]}")
    print("\n✅ Preprocessing pipeline OK!")

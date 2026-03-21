# 🍄 Mushroom Toxicity Classifier — v2 (2023 Dataset)

Predicts whether a mushroom is **edible or poisonous** using the
2023 UCI Secondary Mushroom Dataset — 173 species, including Death Cap.

> ⚠️ **Disclaimer:** For educational purposes only. Never forage based on any ML model.

---

## Why the 2023 Dataset?

| | 1987 Dataset | 2023 Dataset |
|---|---|---|
| Samples | 8,124 | 61,069 |
| Species covered | 23 | **173** |
| Families | Agaricus + Lepiota only | Much broader |
| Death Cap (*Amanita phalloides*) | ❌ Not covered | ✅ Covered |
| Odor feature | ✅ Present (causes shortcut learning) | ❌ Removed (more realistic) |
| Numeric measurements | ❌ None | ✅ cap-diameter, stem-height, stem-width |
| Missing values | stalk-root (~30%) | Several columns (handled via imputation) |

The 1987 dataset only covers 23 species from two closely related families.
Showing it a Death Cap photo resulted in "Edible" because the model had never
seen any Amanita species. The 2023 dataset covers 173 species and was
simulated from a real mycology textbook.

---

## 📁 Project Structure

```
mushroom-ml/
│
├── data/
│   └── mushrooms.csv          ← downloaded automatically on first run
│
├── src/
│   ├── preprocess.py          ← download, clean, ColumnTransformer pipeline
│   ├── train_model.py         ← train Random Forest, save model + plots
│   └── evaluate.py            ← confusion matrix, recall, F1, ROC-AUC
│
├── models/
│   ├── model.pkl              ← saved {preprocessor + rf} bundle
│   └── metadata.pkl           ← feature names + column definitions
│
├── results/
│   └── feature_importance.png ← top 20 features bar chart
│
├── templates/
│   └── index.html             ← web UI
│
├── app.py                     ← Flask server
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### Step 1 — Install
```bash
pip install -r requirements.txt
```

### Step 2 — Train
```bash
python src/train_model.py
```
Downloads dataset (~61k rows), trains Random Forest with ColumnTransformer,
saves `models/model.pkl` and `results/feature_importance.png`.
Training takes ~30–60 seconds on a modern laptop.

### Step 3 — Set Gemini key (for photo analysis)
```bash
# Recommended for this project
copy .env.example .env
# then edit .env and paste your key

# Mac/Linux shell
export GEMINI_API_KEY=your_key_here

# Windows PowerShell
$env:GEMINI_API_KEY="your_key_here"

# Windows Command Prompt
set GEMINI_API_KEY=your_key_here
```
Free key at https://aistudio.google.com/app/apikey

### Step 4 — Run
```bash
python app.py
# Open: http://localhost:5000
```

---

## 🧠 Key ML Concepts in This Project

| Concept | Where |
|---|---|
| ColumnTransformer | `preprocess.py` — different transforms for numeric vs categorical |
| StandardScaler | `preprocess.py` — normalizes cap-diameter, stem-height, stem-width |
| OneHotEncoder | `preprocess.py` — encodes 17 categorical features |
| SimpleImputer | `preprocess.py` — handles missing values (median for numeric, 'unknown' for categorical) |
| sklearn Pipeline | `preprocess.py` — chains imputer + scaler/encoder |
| Random Forest | `train_model.py` — 200 trees, class_weight='balanced' |
| Recall vs Accuracy | `train_model.py` — primary metric is poisonous recall |
| predict_proba() | `app.py` — confidence scores, not just yes/no |
| Gemini Vision API | `app.py` — fills form from photo |

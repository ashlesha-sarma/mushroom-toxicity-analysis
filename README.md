# Mushroom Toxicity Classifier  
ML-based edible vs poisonous mushroom prediction using the 2023 UCI dataset

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Available-green?style=for-the-badge&logo=render)](https://mushrom-toxicity-analysis.onrender.com/)
---

### 🧱 Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Backend-Flask-000000?style=for-the-badge&logo=flask)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458?style=for-the-badge&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy)
![Render](https://img.shields.io/badge/Deployment-Render-46E3B7?style=for-the-badge&logo=render)

---

### 🧠 Problem

Accurately identifying whether a mushroom is edible or poisonous is a safety-critical task.  
Many existing datasets are limited in scope and allow models to rely on unrealistic shortcuts.

---

### 💡 Solution

A machine learning pipeline trained on the **2023 UCI Mushroom Dataset (61k+ samples, 173 species)** that removes shortcut features and focuses on realistic predictive signals.

---

### ✨ Features

- Binary classification: edible vs poisonous  
- Handles mixed feature types (categorical + numeric)  
- End-to-end pipeline (preprocessing + model bundled)  
- Web interface for real-time predictions  
- Model evaluation with recall-focused metrics  

---

### 🏗️ Architecture

User Input → Flask API → Preprocessing Pipeline → Random Forest Model → Prediction

---

### ⚙️ Setup

```bash
git clone https://github.com/your-username/mushroom-toxicity-analysis.git
cd mushroom-toxicity-analysis

pip install -r requirements.txt

# Train model
python src/train_model.py

# Run app
python app.py

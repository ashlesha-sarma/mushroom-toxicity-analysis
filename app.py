# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Flask web server for the Mushroom Toxicity Classifier (2023 Dataset).
#
# Routes:
#   GET  /               → web UI
#   POST /predict        → Random Forest prediction with confidence
#   POST /analyze-photo  → Gemini Flash reads photo → fills all fields
#
# Setup:
#   1. Copy .env.example to .env
#   2. Paste your Gemini key into .env  (get one free at aistudio.google.com)
#   3. Run python app.py — the key loads automatically
#
# Run:
#   python app.py  →  open http://localhost:5000
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import sys
import json
import time
import pickle
import threading
from typing import Optional
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

from google import genai

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocess import (
    NUMERIC_COLS, CATEGORICAL_COLS, ALL_FEATURE_COLS, FEATURE_OPTIONS
)

app = Flask(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join("models", "model.pkl")
METADATA_PATH = os.path.join("models", "metadata.pkl")

# ── Load model ────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print("\n❌ No trained model found.")
    print("   Run this first:  python src/train_model.py\n")
    sys.exit(1)

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

preprocessor = bundle["preprocessor"]
rf_model     = bundle["rf"]
print("✅ Model loaded (2023 Secondary Mushroom Dataset)")

# ── Gemini setup ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print("✅ Gemini API ready")
else:
    gemini_client = None
    print("⚠️  GEMINI_API_KEY not set — photo analysis disabled")
    print("   Get a free key: https://aistudio.google.com/app/apikey")

# Fallback chain — tried in order when quota is hit.
# Each model has an independent free-tier quota bucket.
#   gemini-2.5-flash        → stable GA, best multimodal quality
#   gemini-3-flash-preview  → newer generation, separate quota bucket
#   gemini-2.5-flash-lite   → lightest, highest free-tier RPD, last resort
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
]

FINISH_REASON_BY_CODE = {
    2: "MAX_TOKENS",
}

# Numeric feature ranges for the UI (min, max, default, step, unit)
NUMERIC_RANGES = {
    "cap-diameter": {"min": 0.2,  "max": 67.0, "default": 6.0,  "step": 0.1, "unit": "cm"},
    "stem-height":  {"min": 0.5,  "max": 35.0, "default": 7.0,  "step": 0.1, "unit": "cm"},
    "stem-width":   {"min": 0.5,  "max": 65.0, "default": 10.0, "step": 0.5, "unit": "mm"},
}

# ── Session state ─────────────────────────────────────────────────────────────
session_stats = {
    "total_tokens": 0,
    "lock": threading.Lock()
}


# ── Gemini helpers ────────────────────────────────────────────────────────────

def build_gemini_prompt() -> str:
    """
    Builds the prompt sent to Gemini alongside the mushroom photo.

    Design principles:
    - Output format is declared FIRST (JSON schema with concrete placeholder
      values), before any instructions. Models follow format constraints more
      reliably when shown the schema upfront.
    - Valid codes are listed inline per categorical field — no ambiguity.
    - Hard closing rule ("Output ONLY the JSON object") placed last, where
      instruction-following models weight it most strongly.
    - No polite preamble — every token is load-bearing.
    """
    cat_lines = []
    for feature, options in FEATURE_OPTIONS.items():
        codes = ", ".join(f'"{k}"' for k in options.keys())
        cat_lines.append(f'    "{feature}": "x",   // one of: {codes}')

    cat_block = "\n".join(cat_lines)

    return f"""Output ONLY a single JSON object — no markdown, no explanation, no text outside the braces.

JSON schema (fill every field):
{{
    "cap-diameter": 6.0,        // float, cm, range 0.2-67.0
    "stem-height":  7.0,        // float, cm, range 0.5-35.0
    "stem-width":   10.0,       // float, mm, range 0.5-65.0
{cat_block}
    "species_guess":      "Common Mushroom",
    "species_confidence": "high",
    "analysis_note":      "One sentence describing which visual features guided your estimate."
}}

Rules:
- Estimate numeric values from the image. Clamp to the stated range.
- For each categorical field use ONLY one of the exact quoted codes listed.
- If a feature is not visible, pick the most likely value for the species.
- "species_confidence" must be exactly "high", "medium", or "low".
- Output ONLY the JSON object. Any text outside {{ }} will break the parser."""


def extract_json(raw: str) -> dict:
    """
    Extracts and parses the JSON object from Gemini's raw response.

    Two-pass approach for maximum robustness:

    Pass 1 — strip markdown fences explicitly.
      Gemini often wraps output in ```json ... ``` even when told not to.
      Stripping the fence first isolates the JSON before the regex runs.

    Pass 2 — greedy regex { ... } extraction.
      re.DOTALL lets . match newlines, so multi-line objects are captured.
      Greedy .* finds the FIRST { and the LAST } — correct for nested objects.

    Pass 3 — remove JS-style comments / trailing commas.
      The prompt schema uses inline // comments for readability. If Gemini
      mirrors them back, we strip them before calling json.loads.

    Raises json.JSONDecodeError with a clear message if nothing is found.
    """
    # Pass 1: strip ```json ... ``` or ``` ... ``` fences
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1)

    # Pass 2: extract the outermost { ... } object
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise json.JSONDecodeError(
            f"No JSON object in Gemini response: {raw[:300]!r}", raw, 0
        )
    json_text = match.group(0)
    json_text = re.sub(r"//.*?(?=\r?\n|$)", "", json_text)
    json_text = re.sub(r",\s*([}\]])", r"\1", json_text)
    return json.loads(json_text)


def normalize_finish_reason(finish_reason) -> Optional[str]:
    """
    Normalizes Gemini finish_reason values across SDK versions.

    Older responses may expose an integer code, while newer SDKs return an
    enum-like object such as FinishReason.MAX_TOKENS.
    """
    if finish_reason is None:
        return None

    if isinstance(finish_reason, (int, np.integer)):
        return FINISH_REASON_BY_CODE.get(int(finish_reason), str(int(finish_reason)))

    name = getattr(finish_reason, "name", None)
    if isinstance(name, str) and name:
        return name.upper()

    value = getattr(finish_reason, "value", None)
    if isinstance(value, str) and value:
        return value.upper()
    if isinstance(value, (int, np.integer)):
        return FINISH_REASON_BY_CODE.get(int(value), str(int(value)))

    finish_str = str(finish_reason).strip()
    if not finish_str:
        return None
    if "." in finish_str:
        finish_str = finish_str.split(".")[-1]
    return finish_str.upper()


def usage_to_dict(response) -> dict:
    """Reads token usage safely when usage_metadata is absent or partial."""
    usage_meta = getattr(response, "usage_metadata", None)

    def as_int(value) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    return {
        "prompt": as_int(getattr(usage_meta, "prompt_token_count", 0)),
        "candidates": as_int(getattr(usage_meta, "candidates_token_count", 0)),
        "total": as_int(getattr(usage_meta, "total_token_count", 0)),
    }

def call_gemini_with_fallback(prompt: str, image_part) -> tuple[str, str, dict]:
    """
    Sends the prompt + image to Gemini with automatic model fallback
    and exponential backoff on quota / rate-limit errors (HTTP 429).

    Returns:
       (full_text, model_name, usage_metadata)
    """
    last_error = None

    for model in GEMINI_MODELS:
        for attempt in range(3):
            try:
                print(f"  🔍 Processing with {model} (attempt {attempt+1})...")
                response = gemini_client.models.generate_content(
                    model=model,
                    contents=[prompt, image_part],
                    config=genai.types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=2048,
                    ),
                )

                candidates = getattr(response, "candidates", None) or []
                if not candidates:
                    raise Exception(f"{model} returned no candidates.")

                candidate = candidates[0]
                content   = getattr(candidate, "content", None)
                parts     = getattr(content, "parts", None) or []
                full_text = "".join(
                    p.text for p in parts if hasattr(p, "text") and p.text
                ).strip()

                finish = normalize_finish_reason(getattr(candidate, "finish_reason", None))
                if finish == "MAX_TOKENS":
                    raise Exception(f"MAX_TOKENS: response truncated by {model}")
                if not full_text:
                    finish_msg = f" (finish_reason={finish})" if finish else ""
                    raise Exception(f"{model} returned an empty response{finish_msg}")

                usage = usage_to_dict(response)

                with session_stats["lock"]:
                    session_stats["total_tokens"] += usage["total"]

                return full_text.strip(), model, usage

            except Exception as e:
                err = str(e)
                is_quota = (
                    "429"                in err or
                    "quota"              in err.lower() or
                    "resource_exhausted" in err.lower() or
                    "rate_limit"         in err.lower()
                )
                if is_quota:
                    if attempt < 2:
                        wait = 2 ** attempt
                        print(f"  ⚠️  {model} quota hit, waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"  ❌ {model} quota exhausted.")
                        last_error = e
                        break
                else:
                    raise

    raise last_error or Exception(
        "All Gemini models exhausted their quota. Try again later."
    )

# ── ML helpers ────────────────────────────────────────────────────────────────

def validate_gemini_response(raw: dict) -> dict:
    """
    Cleans and validates the parsed JSON dict from Gemini.
    - Numeric: cast to float, clamp to range, default if missing/invalid.
    - Categorical: accept only known codes, default to first option.
    """
    cleaned = {}

    for col, cfg in NUMERIC_RANGES.items():
        val = raw.get(col, cfg["default"])
        try:
            val = float(val)
            val = max(cfg["min"], min(cfg["max"], val))
        except (TypeError, ValueError):
            val = cfg["default"]
        cleaned[col] = round(val, 1)

    for col, options in FEATURE_OPTIONS.items():
        val = raw.get(col, "")
        cleaned[col] = val if val in options else list(options.keys())[0]

    return cleaned


def raw_input_to_df(data: dict) -> pd.DataFrame:
    """
    Converts the frontend's raw input dict into a single-row DataFrame
    with the exact column order and dtypes the preprocessor expects.
    """
    row = {}
    for col in NUMERIC_COLS:
        try:
            row[col] = float(data.get(col, 0))
        except (TypeError, ValueError):
            row[col] = 0.0
    for col in CATEGORICAL_COLS:
        row[col] = str(data.get(col, ""))
    return pd.DataFrame([row])[ALL_FEATURE_COLS]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        features=FEATURE_OPTIONS,
        numeric_ranges=NUMERIC_RANGES,
        numeric_cols=NUMERIC_COLS,
        gemini_enabled=bool(gemini_client),
    )


@app.route("/analyze-photo", methods=["POST"])
def analyze_photo():
    """
    Receives a mushroom photo, sends it to Gemini,
    and returns all 20 feature values + species info as JSON.
    """
    if not gemini_client:
        return jsonify({
            "error": "Gemini API key not configured. "
                     "Set GEMINI_API_KEY and restart app.py."
        }), 503

    if "photo" not in request.files or request.files["photo"].filename == "":
        return jsonify({"error": "No photo file received."}), 400

    try:
        photo       = request.files["photo"]
        image_bytes = photo.read()
        mime_type   = photo.content_type or "image/jpeg"

        image_part = genai.types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type,
        )

        raw_text, model_used, usage = call_gemini_with_fallback(build_gemini_prompt(), image_part)
        gemini_data = extract_json(raw_text)
        features    = validate_gemini_response(gemini_data)

        # Print detailed usage info to terminal
        print(f"\n📊 Photo Analysis Complete:")
        print(f"   Model:     {model_used}")
        print(f"   Input:     {usage['prompt']} tokens")
        print(f"   Output:    {usage['candidates']} tokens")
        print(f"   Total:     {usage['total']} tokens")
        print(f"   Session cumulative: {session_stats['total_tokens']} tokens\n")

        return jsonify({
            "features":           features,
            "species":            gemini_data.get("species_guess",      "Unknown species"),
            "species_confidence": gemini_data.get("species_confidence", "low"),
            "analysis_note":      gemini_data.get("analysis_note",      ""),
        })

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Could not parse Gemini response: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Photo analysis failed: {str(e)}"}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives all 20 feature values (3 numeric + 17 categorical) as JSON.
    Runs the ColumnTransformer preprocessor + Random Forest.
    Returns prediction, confidence, and probability breakdown.
    """
    try:
        data = request.get_json()

        missing = [f for f in ALL_FEATURE_COLS if f not in data]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        input_df  = raw_input_to_df(data)
        X_encoded = preprocessor.transform(input_df).astype("float32")

        proba       = rf_model.predict_proba(X_encoded)[0]
        p_edible    = round(float(proba[0]) * 100, 2)
        p_poisonous = round(float(proba[1]) * 100, 2)
        pred_class  = int(np.argmax(proba))
        confidence  = round(float(proba[pred_class]) * 100, 2)
        label       = "Poisonous" if pred_class == 1 else "Edible"

        return jsonify({
            "prediction":  label,
            "confidence":  confidence,
            "p_edible":    p_edible,
            "p_poisonous": p_poisonous,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n🍄  Mushroom Toxicity Classifier — 2023 Dataset")
    print("   Open your browser at:  http://localhost:5000\n")
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=port)

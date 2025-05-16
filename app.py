# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import traceback
import numpy as np

# Load model and metadata
bundle = joblib.load('model.pkl')
model = bundle['model']
encoders = bundle['encoders']
features = bundle['features']
target_col = bundle['target']

DATA_FILE = 'data.csv'

app = Flask(__name__)
CORS(app)

# Mapping lowercase form keys to actual model columns
FORM_FIELD_MAP = {
    "a1": "A1", "a2": "A2", "a3": "A3", "a4": "A4", "a5": "A5",
    "a6": "A6", "a7": "A7", "a8": "A8", "a9": "A9", "a10": "A10",
    "age": "Age",
    "sex": "Sex",
    "jaundice": "Jauundice",
    "autism": "Family_ASD"
}

def ensure_numeric(df):
    """Ensure all columns are numeric."""
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df.isnull().any().any():
        raise ValueError("Data contains non-numeric or missing values after conversion.")

@app.route('/')
def home():
    return jsonify({"message": "Hi Shashwat... Autism Diagnosis API is running."})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Step 1: Receive and map user input
        raw_data = request.get_json()
        user_data = {FORM_FIELD_MAP.get(k.lower(), k): v for k, v in raw_data.items()}
        df = pd.DataFrame([user_data])

        # Step 2: Add missing features
        for col in features:
            if col not in df.columns:
                df[col] = '' if col in encoders else 0

        # Step 3: Reorder to match training
        df = df[features]

        # Step 4: Encode categorical features
        for col in df.columns:
            if col in encoders:
                le = encoders[col]
                df[col] = df[col].astype(str)
                fallback = str(le.classes_[0])
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else fallback)
                df[col] = le.transform(df[col])

        # Step 5: Ensure all numeric
        ensure_numeric(df)

        # Debug: print final input
        print("[DEBUG] Final input to model:")
        print(df.head())

        # Step 6: Predict
        pred = model.predict(df)[0]
        result_text = "Yes, you have autism" if pred == 1 else "No, you do not have autism"

        # Step 7: Save the user's response + prediction
        final_data = {col: user_data.get(col, "") for col in features}
        final_data[target_col] = int(pred)
        df_to_save = pd.DataFrame([final_data])

        if os.path.exists(DATA_FILE):
            df_to_save.to_csv(DATA_FILE, mode='a', index=False, header=False)
        else:
            df_to_save.to_csv(DATA_FILE, index=False)

        # Step 8: Return response
        return jsonify({
            "prediction": result_text,
            "prediction_raw": int(pred)
        })

    except Exception as e:
        print("[ERROR]", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    try:
        base_total = 6076
        base_autistic = 1804
        base_non_autistic = 4271

        if not os.path.exists(DATA_FILE):
            return jsonify({
                "total_users": base_total,
                "autistic": base_autistic,
                "non_autistic": base_non_autistic
            })

        df = pd.read_csv(DATA_FILE)
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df = df.dropna(subset=[target_col])

        additional_total = len(df)
        additional_autistic = int(df[target_col].sum())
        additional_non_autistic = additional_total - additional_autistic

        return jsonify({
            "total_users": base_total + additional_total,
            "autistic": base_autistic + additional_autistic,
            "non_autistic": base_non_autistic + additional_non_autistic
        })

    except Exception as e:
        print("[ERROR]", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
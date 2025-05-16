# backend/retrain.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    # Clean column names
    df.columns = df.columns.str.strip()

    # Drop unnecessary columns if they exist
    drop_cols = ['Case_No', 'Score', 'Who completed the test']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Identify the target column (usually ends with "ASD Traits" or contains "Class")
    target_col = None
    for col in df.columns:
        if 'class' in col.lower() or 'asd traits' in col.lower():
            target_col = col
            break

    if not target_col:
        raise Exception("Target column (class label) not found in dataset.")

    # Normalize and map class labels
    df[target_col] = df[target_col].astype(str).str.strip().str.upper().map({'YES': 1, 'NO': 0})

    # Diagnostic print
    print("[INFO] Target class distribution after mapping:")
    print(df[target_col].value_counts(dropna=False))

    # Remove any rows with NaN in the target column
    df = df[df[target_col].notna()]

    # Encode categorical values
    encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le  # Save encoder for use during prediction

    return df, target_col, encoders

def train_model(data_path='data.csv', model_path='model.pkl'):
    print("[INFO] Loading dataset...")
    df = pd.read_csv(data_path)

    print("[INFO] Preprocessing data...")
    df, target_column, encoders = preprocess_data(df)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    print(f"[INFO] Target column: {target_column}")
    print("[INFO] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[INFO] Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    print(f"[INFO] Saving model to {model_path}...")
    joblib.dump({
        'model': model,
        'encoders': encoders,
        'features': X.columns.tolist(),
        'target': target_column
    }, model_path)

    print("Training completed and model saved.")

if __name__ == '__main__':
    train_model()
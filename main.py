import io
import os
import base64
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

load_dotenv()

app = FastAPI(title="Cardiovascular Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUIRED = [
    "id", "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active", "cardio"
]
FEATURES = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active"
]
MODEL_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")


class Patient(BaseModel):
    age: float = Field(..., description="Age in years; dataset may encode age in days")
    gender: int
    height: float
    weight: float
    ap_hi: float
    ap_lo: float
    cholesterol: int
    gluc: int
    smoke: int
    alco: int
    active: int


def clean_dataframe(df: pd.DataFrame):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing columns: {', '.join(missing)}. Expected semicolon-separated cardio dataset."
        )

    before = len(df)
    df = df.drop_duplicates()
    duplicates_removed = before - len(df)

    for c in REQUIRED:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep rows with a valid target; numeric predictors are imputed later.
    df = df.dropna(subset=["cardio"])

    # Dataset commonly stores age in days. Convert to years for human-readable analysis.
    df["age_years"] = np.where(df["age"] > 120, df["age"] / 365.25, df["age"])
    df["bmi"] = df["weight"] / ((df["height"] / 100.0) ** 2)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df, duplicates_removed


def build_pipeline(model):
    numeric_features = FEATURES
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_features)
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def make_models():
    return {
        "Logistic Regression": build_pipeline(LogisticRegression(max_iter=2000)),
        "SVM": build_pipeline(SVC(probability=True, random_state=42)),
        "KNN": build_pipeline(KNeighborsClassifier(n_neighbors=7)),
        "Decision Tree": build_pipeline(
            DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=42)
        ),
        "Random Forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=3,
                random_state=42, n_jobs=-1
            )
        ),
    }


def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def analyze(df):
    X = df[FEATURES]
    y = df["cardio"].astype(int)

    if y.nunique() < 2:
        raise HTTPException(status_code=400, detail="Target column cardio must contain both 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = make_models()
    results = []
    fitted = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        results.append({
            "model": name,
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, prob)), 4) if prob is not None else None,
        })
        fitted[name] = model

    results.sort(key=lambda x: x["accuracy"], reverse=True)
    best_name = results[0]["model"]
    joblib.dump(fitted[best_name], MODEL_PATH)

    # Correlation matrix
    corr_cols = FEATURES + ["cardio"]
    corr = df[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    correlation_plot = plot_to_base64(fig)

    # Target distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=df["cardio"], ax=ax)
    ax.set_title("Cardiovascular Disease Distribution")
    ax.set_xlabel("Cardio (0 = No, 1 = Yes)")
    target_plot = plot_to_base64(fig)

    # Age distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="age_years", hue="cardio", bins=30, kde=True, ax=ax)
    ax.set_title("Age Distribution by Cardiovascular Disease")
    age_plot = plot_to_base64(fig)

    # BMI vs blood pressure
    fig, ax = plt.subplots(figsize=(7, 5))
    sample = df.sample(min(len(df), 5000), random_state=42)
    sns.scatterplot(
        data=sample, x="bmi", y="ap_hi", hue="cardio",
        alpha=0.45, ax=ax
    )
    ax.set_title("BMI vs Systolic Blood Pressure")
    bmi_bp_plot = plot_to_base64(fig)

    # Cholesterol vs target
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x="cholesterol", hue="cardio", ax=ax)
    ax.set_title("Cholesterol Category vs Cardiovascular Disease")
    cholesterol_plot = plot_to_base64(fig)

    # Confusion matrix for best model
    best_model = fitted[best_name]
    best_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, best_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix — {best_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    confusion_plot = plot_to_base64(fig)

    summary = {
        "rows_after_cleaning": int(len(df)),
        "duplicates_removed": int(df.attrs.get("duplicates_removed", 0)),
        "class_distribution": {
            str(k): int(v) for k, v in y.value_counts().sort_index().items()
        },
        "age_mean": round(float(df["age_years"].mean()), 2),
        "bmi_mean": round(float(df["bmi"].mean()), 2),
        "systolic_mean": round(float(df["ap_hi"].mean()), 2),
        "diastolic_mean": round(float(df["ap_lo"].mean()), 2),
    }

    return {
        "summary": summary,
        "models": results,
        "best_model": best_name,
        "correlation": corr.round(4).to_dict(),
        "plots": {
            "correlation": correlation_plot,
            "target_distribution": target_plot,
            "age_distribution": age_plot,
            "bmi_vs_bp": bmi_bp_plot,
            "cholesterol": cholesterol_plot,
            "confusion_matrix": confusion_plot,
        },
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze_dataset(file: Optional[UploadFile] = File(None)):
    if file:
        content = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(content), sep=";")
        except Exception:
            try:
                df = pd.read_csv(io.BytesIO(content))
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Unable to read CSV: {exc}")
    else:
        default_path = os.path.join(os.path.dirname(__file__), "..", "data", "cardio_train.csv")
        if not os.path.exists(default_path):
            raise HTTPException(status_code=400, detail="Upload a dataset or place cardio_train.csv in the data folder.")
        df = pd.read_csv(default_path, sep=";")

    cleaned, duplicates = clean_dataframe(df)
    cleaned.attrs["duplicates_removed"] = duplicates
    return analyze(cleaned)


@app.post("/api/predict")
def predict(patient: Patient):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=400, detail="Train/analyze the dataset first.")

    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([patient.model_dump()])[FEATURES]
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])

    return {
        "prediction": prediction,
        "label": "Higher predicted cardiovascular disease risk" if prediction == 1
                 else "Lower predicted cardiovascular disease risk",
        "probability": round(probability, 4),
    }


@app.post("/api/ai-explain")
def ai_explain(payload: dict):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not configured on the backend.")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

    prompt = f"""
You are assisting with an academic cardiovascular disease machine-learning project.
Explain the following model-analysis result in clear, non-diagnostic language.

Result:
{payload}

Requirements:
- Explain what the accuracy/precision/recall/F1/ROC-AUC mean.
- Identify the best model from the supplied metrics.
- Mention important correlations as associations, not proof of causation.
- Explain that this is a machine-learning prediction and not a medical diagnosis.
- Keep it concise and suitable for a project report/dashboard.
"""

    response = client.responses.create(
        model=model,
        input=prompt,
    )
    return {"explanation": response.output_text}

from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"

DATASETS = {
    "Salary Prediction": {
        "file": "salary_data.csv",
        "feature_candidates": ["YearsExperience", "Years of Experience", "Experience", "Years_Experience"],
        "target_candidates": ["Salary", "Salary (USD)", "salary"],
        "feature_label": "Years of Experience",
        "target_label": "Predicted Salary",
    },
    "Student Exam Score": {
        "file": "student_performance.csv",
        "feature_candidates": ["Hours_Studied", "Hours Studied", "Study Hours", "Hours"],
        "target_candidates": ["Exam_Score", "Exam Score", "Score", "Final_Score"],
        "feature_label": "Hours Studied",
        "target_label": "Predicted Exam Score",
    },
}

def find_column(df, candidates):
    # Match column names while ignoring spaces, underscores, and capitalization.
    normalized = {str(c).lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for candidate in candidates:
        key = candidate.lower().replace(" ", "").replace("_", "")
        if key in normalized:
            return normalized[key]
    return None

def load_and_train(dataset_name):
    config = DATASETS[dataset_name]
    path = DATA_DIR / config["file"]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Download the dataset from Kaggle and place it in the datasets folder."
        )

    df = pd.read_csv(path)
    feature_col = find_column(df, config["feature_candidates"])
    target_col = find_column(df, config["target_candidates"])
    if feature_col is None or target_col is None:
        raise ValueError(
            f"Could not find the required columns in {path.name}.\n"
            f"Columns found: {', '.join(map(str, df.columns))}\n"
            f"Expected an input column similar to {config['feature_candidates'][0]} and "
            f"a target column similar to {config['target_candidates'][0]}."
        )

    clean = df[[feature_col, target_col]].copy()
    clean[feature_col] = pd.to_numeric(clean[feature_col], errors="coerce")
    clean[target_col] = pd.to_numeric(clean[target_col], errors="coerce")
    clean = clean.dropna()
    if len(clean) < 5:
        raise ValueError("The dataset needs at least 5 usable rows after cleaning.")

    X = clean[[feature_col]]
    y = clean[target_col]
    # Fixed random_state makes the train/test split reproducible.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R²": float(r2_score(y_test, predictions)),
        "rows_used": int(len(clean)),
        "feature_column": str(feature_col),
        "target_column": str(target_col),
        "feature_label": config["feature_label"],
        "target_label": config["target_label"],
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
    }
    return model, metrics

if __name__ == "__main__":
    for name in DATASETS:
        try:
            _, result = load_and_train(name)
            print(f"\n{name}")
            for key, value in result.items():
                print(f"{key}: {value}")
        except Exception as exc:
            print(f"\n{name}: {exc}")

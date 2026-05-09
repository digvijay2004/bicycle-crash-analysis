"""
Crash Analysis Project — Python Conversion
Converted from Final_Project.Rmd

Goal: Predict severe non-motorized crash occurrence using Logistic Regression,
Random Forest, and XGBoost.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    RocCurveDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier

RANDOM_STATE = 25
DATA_PATH = Path("bicycle_new_use_final.csv")
TARGET = "severity"

# Same selected predictors used in the R models after feature selection
SELECTED_FEATURES = [
    "dayofwk",
    "lightcond",
    "alcohol",
    "drug",
    "traffic_signal",
    "relationtord",
    "nolanes",
    "truckinv",
    "intersection",
    "direction",
    "gender",
    "age",
    "speedlmt",
]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load data and remove missing rows, matching the R na.omit step."""
    df = pd.read_csv(path)
    df = df.dropna().copy()
    df[TARGET] = df[TARGET].astype(int)
    return df


def print_basic_summary(df: pd.DataFrame) -> None:
    """Print dataset summary similar to R summary/glimpse."""
    print("Dataset shape:", df.shape)
    print("\nColumns:")
    print(df.dtypes)
    print("\nTarget distribution:")
    print(df[TARGET].value_counts().sort_index())


def split_and_balance(df: pd.DataFrame):
    """Create 80/20 stratified split and oversample only the training data."""
    X = df[SELECTED_FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Python equivalent of balancing only training data.
    # ROSE is R-specific; RandomOverSampler keeps categorical values clean.
    oversampler = RandomOverSampler(random_state=123)
    X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, X_train_balanced, y_train_balanced


def save_balance_plot(y_train: pd.Series, y_balanced: pd.Series) -> None:
    """Save before/after class balance plots."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    y_train.value_counts().sort_index().plot(kind="bar", ax=axes[0], title="Before Oversampling")
    y_balanced.value_counts().sort_index().plot(kind="bar", ax=axes[1], title="After Oversampling")
    for ax in axes:
        ax.set_xlabel("Severity")
        ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig("class_balance.png", dpi=300)
    plt.close()


def get_onehot_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), SELECTED_FEATURES),
        ]
    )


def train_logistic_regression(X_train_balanced, y_train_balanced):
    model = Pipeline(
        steps=[
            ("preprocess", get_onehot_preprocessor()),
            ("model", LogisticRegression(max_iter=3000, solver="lbfgs")),
        ]
    )
    model.fit(X_train_balanced, y_train_balanced)
    return model


def train_lasso_logistic_regression(X_train_balanced, y_train_balanced):
    """Python equivalent of glmnet LASSO logistic regression."""
    model = Pipeline(
        steps=[
            ("preprocess", get_onehot_preprocessor()),
            (
                "model",
                LogisticRegressionCV(
                    Cs=10,
                    cv=5,
                    penalty="l1",
                    solver="saga",
                    scoring="roc_auc",
                    max_iter=5000,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X_train_balanced, y_train_balanced)
    return model


def train_random_forest(X_train_balanced, y_train_balanced):
    model = Pipeline(
        steps=[
            ("preprocess", get_onehot_preprocessor()),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )

    param_dist = {
        "model__n_estimators": [200, 500],
        "model__max_features": ["sqrt", "log2", None],
        "model__max_depth": [None, 8, 12, 16],
        "model__min_samples_split": [2, 5, 10],
    }

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=10,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train_balanced, y_train_balanced)
    return search.best_estimator_


def train_xgboost(X_train_balanced, y_train_balanced):
    # XGBoost works well with ordinal-encoded categories here.
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), SELECTED_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=4,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train_balanced, y_train_balanced)
    return model


def evaluate_model(name: str, model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    auc_score = roc_auc_score(y_test, y_proba)

    print(f"\n{name}")
    print("AUC:", round(auc_score, 3))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return {"name": name, "auc": auc_score, "fpr": fpr, "tpr": tpr, "model": model}


def save_roc_plot(results) -> None:
    plt.figure(figsize=(7, 6))
    for result in results:
        plt.plot(result["fpr"], result["tpr"], label=f'{result["name"]} (AUC={result["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=300)
    plt.close()


def save_xgboost_feature_importance(xgb_pipeline) -> None:
    xgb_model = xgb_pipeline.named_steps["model"]
    importance = pd.DataFrame(
        {
            "feature": SELECTED_FEATURES,
            "importance": xgb_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance.to_csv("xgboost_feature_importance.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.barh(importance["feature"], importance["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig("xgboost_feature_importance.png", dpi=300)
    plt.close()


def main() -> None:
    df = load_data()
    print_basic_summary(df)

    X_train, X_test, y_train, y_test, X_train_balanced, y_train_balanced = split_and_balance(df)
    save_balance_plot(y_train, y_train_balanced)

    print("\nTraining Logistic Regression...")
    logit_model = train_logistic_regression(X_train_balanced, y_train_balanced)

    print("\nTraining LASSO Logistic Regression...")
    lasso_model = train_lasso_logistic_regression(X_train_balanced, y_train_balanced)

    print("\nTraining Random Forest...")
    rf_model = train_random_forest(X_train_balanced, y_train_balanced)

    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(X_train_balanced, y_train_balanced)

    results = [
        evaluate_model("Logistic Regression", logit_model, X_test, y_test),
        evaluate_model("LASSO Logistic Regression", lasso_model, X_test, y_test),
        evaluate_model("Random Forest", rf_model, X_test, y_test),
        evaluate_model("XGBoost", xgb_model, X_test, y_test),
    ]

    save_roc_plot(results)
    save_xgboost_feature_importance(xgb_model)

    print("\nSaved outputs:")
    print("- class_balance.png")
    print("- roc_curves.png")
    print("- xgboost_feature_importance.csv")
    print("- xgboost_feature_importance.png")


if __name__ == "__main__":
    main()

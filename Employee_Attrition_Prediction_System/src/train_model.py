from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import (
    SENSITIVE_COLUMNS,
    prepare_features,
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR /
    "data" /
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

MODEL_DIR = BASE_DIR / "models"

REPORT_DIR = BASE_DIR / "reports"

FIG_DIR = REPORT_DIR / "figures"


MODEL_DIR.mkdir(
    exist_ok=True
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PREPROCESSOR
# ============================================================

def make_preprocessor(
    X: pd.DataFrame,
    scale_numeric: bool
) -> ColumnTransformer:

    numeric_features = (
        X
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    categorical_features = (
        X
        .select_dtypes(exclude=np.number)
        .columns
        .tolist()
    )

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        )
    ]

    if scale_numeric:

        numeric_steps.append(
            (
                "scaler",
                StandardScaler()
            )
        )

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "num",

                Pipeline(
                    numeric_steps
                ),

                numeric_features
            ),

            (
                "cat",

                Pipeline(
                    [
                        (
                            "imputer",

                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),

                        (
                            "onehot",

                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False
                            ),
                        ),
                    ]
                ),

                categorical_features
            ),
        ],

        remainder="drop",

        verbose_feature_names_out=True,
    )

    return preprocessor


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    y_true,
    probability,
    threshold
):

    prediction = (
        probability >= threshold
    ).astype(int)

    return {

        "accuracy":
            accuracy_score(
                y_true,
                prediction
            ),

        "precision":
            precision_score(
                y_true,
                prediction,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                prediction,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                prediction,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                probability
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                probability
            ),

        "threshold":
            threshold,
    }


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def tune_threshold(
    y_true,
    probability
):

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probability
        )
    )

    f1 = (
        2 * precision * recall /
        np.clip(
            precision + recall,
            1e-12,
            None
        )
    )

    best_index = int(
        np.nanargmax(
            f1[:-1]
        )
    )

    threshold = float(
        thresholds[best_index]
    )

    return float(
        np.clip(
            threshold,
            0.05,
            0.95
        )
    )


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print(
        f"Loading dataset:\n{DATA_PATH}"
    )

    df = pd.read_csv(
        DATA_PATH
    )

    print("\nDataset loaded successfully.")

    print(
        f"\nRows: {df.shape[0]}"
    )

    print(
        f"Columns: {df.shape[1]}"
    )

    # --------------------------------------------------------
    # TARGET ENCODING
    # --------------------------------------------------------

    df["Attrition"] = (
        df["Attrition"]
        .map(
            {
                "No": 0,
                "Yes": 1
            }
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # AUDIT INFORMATION
    # --------------------------------------------------------

    audit = df[
        SENSITIVE_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # CREATE X AND y
    # --------------------------------------------------------

    X_raw = df.drop(
        columns=["Attrition"]
    )

    y = df["Attrition"]

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    X = prepare_features(
        X_raw
    )

    print(
        f"\nFinal ML features: {len(X.columns)}"
    )

    print(
        f"Attrition rate: {y.mean():.2%}"
    )

    # ========================================================
    # TRAIN / VALIDATION / TEST
    # ========================================================

    X_temp, X_test, \
    y_temp, y_test, \
    audit_temp, audit_test = train_test_split(

        X,
        y,
        audit,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

    X_train, X_val, \
    y_train, y_val, \
    audit_train, audit_val = train_test_split(

        X_temp,
        y_temp,
        audit_temp,

        test_size=0.25,

        random_state=42,

        stratify=y_temp
    )

    print("\nDataset split:")

    print(
        f"Training:   {len(X_train)}"
    )

    print(
        f"Validation: {len(X_val)}"
    )

    print(
        f"Test:       {len(X_test)}"
    )

    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

    log_pipeline = Pipeline(

        [
            (
                "preprocessor",

                make_preprocessor(
                    X_train,
                    scale_numeric=True
                )
            ),

            (
                "model",

                LogisticRegression(

                    class_weight="balanced",

                    max_iter=2500,

                    solver="liblinear",

                    random_state=42
                )
            ),
        ]
    )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    rf_pipeline = Pipeline(

        [

            (
                "preprocessor",

                make_preprocessor(
                    X_train,
                    scale_numeric=False
                )
            ),

            (
                "model",

                RandomForestClassifier(

                    class_weight=
                    "balanced_subsample",

                    random_state=42,

                    n_jobs=-1
                )
            ),
        ]
    )

    # ========================================================
    # LOGISTIC SEARCH
    # ========================================================

    log_search = RandomizedSearchCV(

        log_pipeline,

        {
            "model__C":
                np.logspace(
                    -2,
                    1,
                    6
                )
        },

        n_iter=6,

        scoring="roc_auc",

        cv=5,

        random_state=42,

        n_jobs=-1,

        refit=True
    )

    # ========================================================
    # RANDOM FOREST SEARCH
    # ========================================================

    rf_search = RandomizedSearchCV(

        rf_pipeline,

        {

            "model__n_estimators":
                [300, 500, 700],

            "model__max_depth":
                [None, 8, 12, 16],

            "model__min_samples_leaf":
                [1, 2, 4, 6],

            "model__max_features":
                [
                    "sqrt",
                    "log2",
                    0.7
                ],
        },

        n_iter=10,

        scoring="roc_auc",

        cv=5,

        random_state=42,

        n_jobs=-1,

        refit=True
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\nTraining Logistic Regression..."
    )

    log_search.fit(
        X_train,
        y_train
    )

    print(
        "Logistic Regression complete."
    )

    print(
        "\nTraining Random Forest..."
    )

    rf_search.fit(
        X_train,
        y_train
    )

    print(
        "Random Forest complete."
    )

    # ========================================================
    # VALIDATION COMPARISON
    # ========================================================

    candidates = {

        "LogisticRegression":
            log_search.best_estimator_,

        "RandomForest":
            rf_search.best_estimator_,
    }

    validation_results = []

    for name, model in candidates.items():

        probability = (
            model
            .predict_proba(X_val)[:, 1]
        )

        threshold = tune_threshold(
            y_val,
            probability
        )

        metrics = evaluate(
            y_val,
            probability,
            threshold
        )

        metrics["model"] = name

        validation_results.append(
            metrics
        )

    validation_df = (
        pd.DataFrame(
            validation_results
        )
        .sort_values(
            "pr_auc",
            ascending=False
        )
    )

    validation_df.to_csv(
        REPORT_DIR /
        "validation_model_comparison.csv",

        index=False
    )

    print(
        "\nVALIDATION RESULTS"
    )

    print(
        validation_df.to_string(
            index=False
        )
    )

    # ========================================================
    # FINAL MODEL
    # ========================================================

    # Random Forest is used as the production candidate
    # because it can model nonlinear interactions and can
    # be explained using Tree SHAP.

    final_model = candidates[
        "RandomForest"
    ]

    validation_probability = (
        final_model
        .predict_proba(X_val)[:, 1]
    )

    threshold = tune_threshold(
        y_val,
        validation_probability
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    test_probability = (
        final_model
        .predict_proba(X_test)[:, 1]
    )

    test_metrics = evaluate(
        y_test,
        test_probability,
        threshold
    )

    test_metrics["model"] = (
        "RandomForest"
    )

    pd.DataFrame(
        [test_metrics]
    ).to_csv(

        REPORT_DIR /
        "test_metrics.csv",

        index=False
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    test_prediction = (
        test_probability >= threshold
    ).astype(int)

    report = classification_report(

        y_test,

        test_prediction,

        target_names=[
            "Stay",
            "Leave"
        ],

        output_dict=True
    )

    pd.DataFrame(
        report
    ).T.to_csv(

        REPORT_DIR /
        "classification_report.csv"
    )

    # ========================================================
    # SAVE HYPERPARAMETERS
    # ========================================================

    with open(
        REPORT_DIR /
        "best_params.json",

        "w",

        encoding="utf-8"
    ) as file:

        json.dump(
            rf_search.best_params_,
            file,
            indent=2,
            default=str
        )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_test,
        test_prediction
    )

    plt.figure(
        figsize=(6, 5)
    )

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=[
            "Stay",
            "Leave"
        ],

        yticklabels=[
            "Stay",
            "Leave"
        ]
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.title(
        "Employee Attrition Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(

        FIG_DIR /
        "confusion_matrix.png",

        dpi=200
    )

    plt.close()

    # ========================================================
    # ROC CURVE
    # ========================================================

    fpr, tpr, _ = roc_curve(
        y_test,
        test_probability
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(

        fpr,

        tpr,

        label=(
            "Random Forest "
            f"(AUC={test_metrics['roc_auc']:.3f})"
        )
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        FIG_DIR /
        "roc_curve.png",

        dpi=200
    )

    plt.close()

    # ========================================================
    # PRECISION-RECALL CURVE
    # ========================================================

    precision, recall, _ = (
        precision_recall_curve(
            y_test,
            test_probability
        )
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        recall,
        precision
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        f"Precision-Recall Curve "
        f"(AP={test_metrics['pr_auc']:.3f})"
    )

    plt.tight_layout()

    plt.savefig(

        FIG_DIR /
        "precision_recall_curve.png",

        dpi=200
    )

    plt.close()

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    preprocessor = (
        final_model
        .named_steps[
            "preprocessor"
        ]
    )

    estimator = (
        final_model
        .named_steps[
            "model"
        ]
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = (
        estimator
        .feature_importances_
    )

    feature_importance = pd.DataFrame(

        {
            "feature":
                feature_names,

            "importance":
                importances,
        }
    ).sort_values(

        "importance",

        ascending=False
    )

    feature_importance.to_csv(

        REPORT_DIR /
        "feature_importance.csv",

        index=False
    )

    plt.figure(
        figsize=(9, 7)
    )

    top = (
        feature_importance
        .head(20)
        .sort_values(
            "importance"
        )
    )

    plt.barh(

        top["feature"],

        top["importance"]
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top 20 Model Features"
    )

    plt.tight_layout()

    plt.savefig(

        FIG_DIR /
        "feature_importance.png",

        dpi=200
    )

    plt.close()

    # ========================================================
    # FAIRNESS / AUDIT ANALYSIS
    # ========================================================

    audit_results = []

    test_audit = (
        audit_test
        .reset_index(
            drop=True
        )
    )

    y_test_reset = (
        y_test
        .reset_index(
            drop=True
        )
    )

    prediction_reset = pd.Series(
        test_prediction
    )

    probability_reset = pd.Series(
        test_probability
    )

    for column in [
        "Gender",
        "MaritalStatus"
    ]:

        for group, indexes in (
            test_audit
            .groupby(column)
            .groups
            .items()
        ):

            indexes = list(
                indexes
            )

            actual_group = (
                y_test_reset
                .iloc[indexes]
            )

            prediction_group = (
                prediction_reset
                .iloc[indexes]
            )

            probability_group = (
                probability_reset
                .iloc[indexes]
            )

            audit_results.append(

                {
                    "attribute":
                        column,

                    "group":
                        str(group),

                    "n":
                        len(indexes),

                    "actual_attrition_rate":
                        actual_group.mean(),

                    "predicted_positive_rate":
                        prediction_group.mean(),

                    "recall":
                        recall_score(
                            actual_group,
                            prediction_group,
                            zero_division=0
                        ),

                    "average_predicted_probability":
                        probability_group.mean(),
                }
            )

    pd.DataFrame(
        audit_results
    ).to_csv(

        REPORT_DIR /
        "fairness_audit.csv",

        index=False
    )

    # ========================================================
    # SAVE MODEL BUNDLE
    # ========================================================

    bundle = {

        "model":
            final_model,

        "threshold":
            threshold,

        "raw_input_columns":
            X_raw.columns.tolist(),

        "model_input_columns":
            X.columns.tolist(),

        "sensitive_columns_excluded_from_model":
            SENSITIVE_COLUMNS,

        "test_metrics":
            test_metrics,

        "best_params":
            rf_search.best_params_,
    }

    joblib.dump(

        bundle,

        MODEL_DIR /
        "attrition_model.joblib"
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n===================================="
    )

    print(
        "FINAL TEST METRICS"
    )

    print(
        "===================================="
    )

    for key, value in test_metrics.items():

        if isinstance(
            value,
            float
        ):

            print(
                f"{key}: {value:.4f}"
            )

        else:

            print(
                f"{key}: {value}"
            )

    print(
        "\nModel saved successfully:"
    )

    print(
        MODEL_DIR /
        "attrition_model.joblib"
    )


if __name__ == "__main__":
    main()
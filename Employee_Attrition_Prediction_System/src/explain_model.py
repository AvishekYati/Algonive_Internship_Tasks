from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from preprocessing import prepare_features


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR /
    "data" /
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

MODEL_PATH = (
    BASE_DIR /
    "models" /
    "attrition_model.joblib"
)

OUTPUT_PATH = (
    BASE_DIR /
    "reports" /
    "figures" /
    "shap_summary.png"
)


def main():

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    bundle = joblib.load(
        MODEL_PATH
    )

    model = bundle[
        "model"
    ]

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    df = pd.read_csv(
        DATA_PATH
    )

    X_raw = df.drop(
        columns=["Attrition"]
    )

    X = prepare_features(
        X_raw
    )

    # -----------------------------------------------------
    # Transform data through trained preprocessing
    # -----------------------------------------------------

    preprocessor = (
        model
        .named_steps[
            "preprocessor"
        ]
    )

    estimator = (
        model
        .named_steps[
            "model"
        ]
    )

    # Limit explanation sample to make SHAP faster.
    sample = X.sample(
        n=min(
            200,
            len(X)
        ),
        random_state=42
    )

    transformed = (
        preprocessor
        .transform(sample)
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    # -----------------------------------------------------
    # Tree SHAP
    # -----------------------------------------------------

    explainer = shap.TreeExplainer(
        estimator
    )

    shap_values = (
        explainer.shap_values(
            transformed
        )
    )

    # Handle SHAP output differences between versions.
    if isinstance(
        shap_values,
        list
    ):

        values = shap_values[1]

    else:

        values = np.asarray(
            shap_values
        )

        if values.ndim == 3:

            values = values[:, :, 1]

    # -----------------------------------------------------
    # Create summary plot
    # -----------------------------------------------------

    plt.figure(
        figsize=(10, 8)
    )

    shap.summary_plot(

        values,

        transformed,

        feature_names=feature_names,

        max_display=20,

        show=False
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "SHAP explanation saved:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
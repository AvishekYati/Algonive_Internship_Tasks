from __future__ import annotations

import pandas as pd


TARGET = "Attrition"

# These fields are retained for auditing/analytics but intentionally
# excluded from production model scoring.
SENSITIVE_COLUMNS = [
    "Age",
    "Gender",
    "MaritalStatus",
]

# Constants, identifiers, and sensitive demographic attributes
# excluded from the prediction model.
DROP_FROM_MODEL = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours",
    "Age",
    "Gender",
    "MaritalStatus",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create business-oriented features from the raw HR data.
    """

    out = df.copy()

    # ---------------------------------------------------------
    # 1. Overall satisfaction index
    # ---------------------------------------------------------
    satisfaction_columns = [
        "EnvironmentSatisfaction",
        "JobSatisfaction",
        "RelationshipSatisfaction",
        "WorkLifeBalance",
    ]

    available = [
        column
        for column in satisfaction_columns
        if column in out.columns
    ]

    if available:
        out["SatisfactionIndex"] = out[available].mean(axis=1)

    # ---------------------------------------------------------
    # 2. Compensation feature
    # ---------------------------------------------------------
    out["IncomePerJobLevel"] = (
        out["MonthlyIncome"] /
        (out["JobLevel"] + 1)
    )

    # ---------------------------------------------------------
    # 3. Company tenure compared with total experience
    # ---------------------------------------------------------
    out["TenureToExperienceRatio"] = (
        out["YearsAtCompany"] /
        (out["TotalWorkingYears"] + 1)
    )

    # ---------------------------------------------------------
    # 4. Manager stability
    # ---------------------------------------------------------
    out["ManagerTenureRatio"] = (
        out["YearsWithCurrManager"] /
        (out["YearsAtCompany"] + 1)
    )

    # ---------------------------------------------------------
    # 5. Promotion waiting ratio
    # ---------------------------------------------------------
    out["PromotionWaitRatio"] = (
        out["YearsSinceLastPromotion"] /
        (out["YearsAtCompany"] + 1)
    )

    # ---------------------------------------------------------
    # 6. Early-career indicator
    # ---------------------------------------------------------
    out["EarlyCareerFlag"] = (
        out["TotalWorkingYears"] <= 3
    ).astype(int)

    # ---------------------------------------------------------
    # 7. Frequent job-change indicator
    # ---------------------------------------------------------
    out["FrequentJobChangeFlag"] = (
        out["NumCompaniesWorked"] >= 4
    ).astype(int)

    # ---------------------------------------------------------
    # 8. Overtime numerical flag
    # ---------------------------------------------------------
    out["OverTimeFlag"] = (
        out["OverTime"] == "Yes"
    ).astype(int)

    return out


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare raw HR records for ML inference.
    """

    out = add_engineered_features(df)

    # Remove target if present
    out = out.drop(
        columns=[TARGET],
        errors="ignore"
    )

    # Remove IDs, constants and audit-only demographics
    out = out.drop(
        columns=DROP_FROM_MODEL,
        errors="ignore"
    )

    return out


def validate_input_columns(df: pd.DataFrame) -> list[str]:
    """
    Check whether an uploaded CSV contains
    the raw columns required for scoring.
    """

    required = [
        "BusinessTravel",
        "DailyRate",
        "Department",
        "DistanceFromHome",
        "Education",
        "EducationField",
        "EnvironmentSatisfaction",
        "HourlyRate",
        "JobInvolvement",
        "JobLevel",
        "JobRole",
        "JobSatisfaction",
        "MonthlyIncome",
        "MonthlyRate",
        "NumCompaniesWorked",
        "OverTime",
        "PercentSalaryHike",
        "PerformanceRating",
        "RelationshipSatisfaction",
        "StockOptionLevel",
        "TotalWorkingYears",
        "TrainingTimesLastYear",
        "WorkLifeBalance",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
    ]

    return [
        column
        for column in required
        if column not in df.columns
    ]
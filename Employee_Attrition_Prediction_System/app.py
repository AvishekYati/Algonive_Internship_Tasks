from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

from src.preprocessing import (
    prepare_features,
    validate_input_columns,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent

MODEL_PATH = (
    BASE_DIR /
    "models" /
    "attrition_model.joblib"
)

DATA_PATH = (
    BASE_DIR /
    "data" /
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(

    page_title=
    "Algonive | Employee Attrition AI",

    page_icon="📊",

    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource

def load_bundle():

    return joblib.load(
        MODEL_PATH
    )


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data

def load_data():

    return pd.read_csv(
        DATA_PATH
    )


bundle = load_bundle()

model = bundle[
    "model"
]

threshold = float(
    bundle[
        "threshold"
    ]
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "Employee Attrition Prediction System"
)

st.caption(
    "Algonive Internship Project • "
    "Explainable Early-Warning Analytics"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "System Information"
    )

    st.metric(
        "Decision Threshold",
        f"{threshold:.2f}"
    )

    st.info(
        "This application is a portfolio/research "
        "prototype and should not be used as an "
        "automated employment decision system."
    )

    st.markdown(
        """
**Excluded from model scoring**

• Age  
• Gender  
• Marital Status  
• Employee identifiers  
• Constant columns
"""
    )


# ============================================================
# NAVIGATION
# ============================================================

page = st.radio(

    "Application Module",

    [
        "Single Employee Risk",
        "Batch Scoring",
        "Workforce Analytics"
    ],

    horizontal=True
)


# ============================================================
# SINGLE EMPLOYEE
# ============================================================

if page == "Single Employee Risk":

    st.subheader(
        "Single Employee Early-Warning Score"
    )

    st.write(
        "Enter operational, job and work-environment "
        "attributes. Sensitive demographic variables "
        "are intentionally excluded from model scoring."
    )

    with st.form(
        "prediction_form"
    ):

        # ====================================================
        # JOB
        # ====================================================

        st.markdown(
            "### Job Information"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            business_travel = st.selectbox(
                "Business Travel",
                [
                    "Travel_Rarely",
                    "Travel_Frequently",
                    "Non-Travel"
                ]
            )

            department = st.selectbox(
                "Department",
                [
                    "Sales",
                    "Research & Development",
                    "Human Resources"
                ]
            )

            education_field = st.selectbox(
                "Education Field",
                [
                    "Life Sciences",
                    "Medical",
                    "Marketing",
                    "Technical Degree",
                    "Human Resources",
                    "Other"
                ]
            )

            job_role = st.selectbox(
                "Job Role",
                [
                    "Sales Executive",
                    "Research Scientist",
                    "Laboratory Technician",
                    "Manufacturing Director",
                    "Healthcare Representative",
                    "Manager",
                    "Sales Representative",
                    "Research Director",
                    "Human Resources"
                ]
            )

        with col2:

            job_level = st.slider(
                "Job Level",
                1,
                5,
                2
            )

            monthly_income = st.number_input(
                "Monthly Income",
                min_value=100,
                max_value=100000,
                value=5000,
                step=100
            )

            monthly_rate = st.number_input(
                "Monthly Rate",
                min_value=2000,
                max_value=30000,
                value=15000,
                step=100
            )

            daily_rate = st.number_input(
                "Daily Rate",
                min_value=100,
                max_value=1500,
                value=800,
                step=10
            )

        with col3:

            hourly_rate = st.number_input(
                "Hourly Rate",
                min_value=30,
                max_value=150,
                value=70,
                step=1
            )

            distance = st.slider(
                "Distance From Home",
                1,
                30,
                5
            )

            total_years = st.slider(
                "Total Working Years",
                0,
                40,
                8
            )

            years_company = st.slider(
                "Years At Company",
                0,
                40,
                5
            )

        # ====================================================
        # WORK EXPERIENCE
        # ====================================================

        st.markdown(
            "### Experience & Career"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            years_role = st.slider(
                "Years In Current Role",
                0,
                20,
                3
            )

            years_since_promotion = st.slider(
                "Years Since Last Promotion",
                0,
                20,
                1
            )

            years_manager = st.slider(
                "Years With Current Manager",
                0,
                20,
                3
            )

        with col2:

            companies_worked = st.slider(
                "Companies Worked",
                0,
                15,
                2
            )

            stock_option_level = st.slider(
                "Stock Option Level",
                0,
                3,
                1
            )

            training = st.slider(
                "Training Times Last Year",
                0,
                10,
                3
            )

        with col3:

            salary_hike = st.slider(
                "Percent Salary Hike",
                10,
                30,
                14
            )

            overtime = st.selectbox(
                "Overtime",
                [
                    "Yes",
                    "No"
                ]
            )

        # ====================================================
        # SATISFACTION
        # ====================================================

        st.markdown(
            "### Satisfaction & Work Environment"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            education = st.slider(
                "Education Level",
                1,
                5,
                3
            )

        with col2:

            job_involvement = st.slider(
                "Job Involvement",
                1,
                4,
                3
            )

        with col3:

            job_satisfaction = st.slider(
                "Job Satisfaction",
                1,
                4,
                3
            )

        with col4:

            environment_satisfaction = st.slider(
                "Environment Satisfaction",
                1,
                4,
                3
            )

        col1, col2, col3 = st.columns(3)

        with col1:

            relationship_satisfaction = st.slider(
                "Relationship Satisfaction",
                1,
                4,
                3
            )

        with col2:

            work_life_balance = st.slider(
                "Work-Life Balance",
                1,
                4,
                3
            )

        with col3:

            performance_rating = st.slider(
                "Performance Rating",
                1,
                4,
                3
            )

        # ====================================================
        # SUBMIT
        # ====================================================

        submitted = st.form_submit_button(
            "Calculate Attrition Risk",
            type="primary"
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    if submitted:

        row = pd.DataFrame(

            [
                {

                    "BusinessTravel":
                        business_travel,

                    "DailyRate":
                        daily_rate,

                    "Department":
                        department,

                    "DistanceFromHome":
                        distance,

                    "Education":
                        education,

                    "EducationField":
                        education_field,

                    "EnvironmentSatisfaction":
                        environment_satisfaction,

                    "HourlyRate":
                        hourly_rate,

                    "JobInvolvement":
                        job_involvement,

                    "JobLevel":
                        job_level,

                    "JobRole":
                        job_role,

                    "JobSatisfaction":
                        job_satisfaction,

                    "MonthlyIncome":
                        monthly_income,

                    "MonthlyRate":
                        monthly_rate,

                    "NumCompaniesWorked":
                        companies_worked,

                    "OverTime":
                        overtime,

                    "PercentSalaryHike":
                        salary_hike,

                    "PerformanceRating":
                        performance_rating,

                    "RelationshipSatisfaction":
                        relationship_satisfaction,

                    "StockOptionLevel":
                        stock_option_level,

                    "TotalWorkingYears":
                        total_years,

                    "TrainingTimesLastYear":
                        training,

                    "WorkLifeBalance":
                        work_life_balance,

                    "YearsAtCompany":
                        years_company,

                    "YearsInCurrentRole":
                        years_role,

                    "YearsSinceLastPromotion":
                        years_since_promotion,

                    "YearsWithCurrManager":
                        years_manager,

                }
            ]
        )

        # ----------------------------------------------------
        # PREPARE
        # ----------------------------------------------------

        X_input = prepare_features(
            row
        )

        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        probability = float(
            model.predict_proba(
                X_input
            )[:, 1][0]
        )

        # ----------------------------------------------------
        # RISK BAND
        # ----------------------------------------------------

        if probability >= threshold:

            risk_band = (
                "High Early-Warning Risk"
            )

        else:

            risk_band = (
                "Lower Early-Warning Risk"
            )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        st.markdown(
            "### Prediction"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Attrition Probability",
                f"{probability:.1%}"
            )

        with col2:

            st.metric(
                "Model Threshold",
                f"{threshold:.1%}"
            )

        with col3:

            st.metric(
                "Risk Band",
                risk_band
            )

        st.progress(
            min(
                probability,
                1.0
            )
        )

        st.warning(
            "This is an early-warning analytics signal, "
            "not a recommendation to terminate, reject, "
            "promote or penalize an employee."
        )

        # ----------------------------------------------------
        # SHAP EXPLANATION
        # ----------------------------------------------------

        st.markdown(
            "### Why did the model produce this score?"
        )

        try:

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

            transformed = (
                preprocessor
                .transform(
                    X_input
                )
            )

            feature_names = (
                preprocessor
                .get_feature_names_out()
            )

            explainer = (
                shap.TreeExplainer(
                    estimator
                )
            )

            shap_values = (
                explainer
                .shap_values(
                    transformed
                )
            )

            if isinstance(
                shap_values,
                list
            ):

                values = shap_values[1][0]

            else:

                values = shap_values

                if values.ndim == 3:

                    values = (
                        values[0, :, 1]
                    )

                else:

                    values = (
                        values[0]
                    )

            explanation = pd.DataFrame(

                {
                    "Feature":
                        feature_names,

                    "Contribution":
                        values,
                }
            )

            explanation[
                "AbsoluteContribution"
            ] = explanation[
                "Contribution"
            ].abs()

            explanation = (
                explanation
                .sort_values(
                    "AbsoluteContribution",
                    ascending=False
                )
                .head(10)
            )

            fig = px.bar(

                explanation.sort_values(
                    "Contribution"
                ),

                x="Contribution",

                y="Feature",

                orientation="h",

                title=
                "Top Model Contributions"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        except Exception as error:

            st.info(
                "SHAP explanation could not be rendered "
                f"in this session: {error}"
            )


# ============================================================
# BATCH SCORING
# ============================================================

elif page == "Batch Scoring":

    st.subheader(
        "Batch Employee Scoring"
    )

    st.write(
        "Upload an HR CSV containing the raw fields "
        "used by the model."
    )

    uploaded = st.file_uploader(
        "Upload HR CSV",
        type=["csv"]
    )

    if uploaded:

        batch = pd.read_csv(
            uploaded
        )

        missing = (
            validate_input_columns(
                batch
            )
        )

        if missing:

            st.error(
                "Missing columns: "
                +
                ", ".join(
                    missing
                )
            )

        else:

            X_batch = (
                prepare_features(
                    batch
                )
            )

            probability = (
                model
                .predict_proba(
                    X_batch
                )[:, 1]
            )

            result = (
                batch
                .copy()
            )

            result[
                "AttritionProbability"
            ] = probability

            result[
                "EarlyWarningFlag"
            ] = (
                probability >=
                threshold
            )

            result[
                "RiskBand"
            ] = pd.cut(

                probability,

                bins=[
                    -0.01,
                    threshold * 0.60,
                    threshold,
                    1.0
                ],

                labels=[
                    "Lower",
                    "Watch",
                    "High"
                ]
            )

            st.dataframe(
                result.head(100),
                use_container_width=True
            )

            st.download_button(

                "Download Scored CSV",

                data=
                result
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8"
                ),

                file_name=
                "employee_attrition_scored.csv",

                mime=
                "text/csv"
            )


# ============================================================
# WORKFORCE ANALYTICS
# ============================================================

else:

    st.subheader(
        "Workforce Analytics"
    )

    df = load_data()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Employees",
            len(df)
        )

    with col2:

        st.metric(

            "Attrition Rate",

            f"{
                df['Attrition']
                .eq('Yes')
                .mean()
            :.1%}"
        )

    with col3:

        st.metric(

            "Average Monthly Income",

            f"₹{
                df['MonthlyIncome']
                .mean()
            :,.0f}"
        )

    with col4:

        st.metric(

            "Average Years At Company",

            f"{
                df['YearsAtCompany']
                .mean()
            :.1f}"
        )

    # --------------------------------------------------------
    # ROLE ANALYSIS
    # --------------------------------------------------------

    role = (

        df
        .groupby(
            "JobRole"
        )["Attrition"]

        .apply(
            lambda series:
            (series == "Yes")
            .mean()
        )

        .reset_index(
            name="AttritionRate"
        )
    )

    role = role.sort_values(
        "AttritionRate"
    )

    fig_role = px.bar(

        role,

        x="AttritionRate",

        y="JobRole",

        orientation="h",

        title=
        "Attrition Rate by Job Role"
    )

    st.plotly_chart(
        fig_role,
        use_container_width=True
    )

    # --------------------------------------------------------
    # OVERTIME ANALYSIS
    # --------------------------------------------------------

    overtime = (

        df
        .groupby(
            "OverTime"
        )["Attrition"]

        .apply(
            lambda series:
            (series == "Yes")
            .mean()
        )

        .reset_index(
            name="AttritionRate"
        )
    )

    fig_overtime = px.bar(

        overtime,

        x="OverTime",

        y="AttritionRate",

        title=
        "Attrition Rate by Overtime"
    )

    st.plotly_chart(
        fig_overtime,
        use_container_width=True
    )

    st.caption(
        "These charts describe patterns in the sample. "
        "They do not establish that a factor causes attrition."
    )
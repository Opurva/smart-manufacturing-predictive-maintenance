import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Smart Manufacturing Predictive Maintenance",
    page_icon="🏭",
    layout="wide"
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "trained_models"
PREPROCESSING_PATH = BASE_DIR / "models" / "preprocessing"
REPORTS_PATH = BASE_DIR / "reports"
DATASET_PATH = BASE_DIR / "dataset"

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.main-title {
    font-size: 48px;
    font-weight: 800;
}
.card {
    padding: 22px;
    border-radius: 14px;
    background-color: #111827;
    border: 1px solid #374151;
}
.metric-card {
    padding: 20px;
    border-radius: 14px;
    background-color: #0f172a;
    border: 1px solid #334155;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Helper Functions
# -----------------------------
@st.cache_resource
def load_model_assets():
    best_model_name = joblib.load(MODEL_PATH / "best_model_name.pkl")

    model_paths = {
        "Logistic Regression": MODEL_PATH / "logistic_regression.pkl",
        "Decision Tree": MODEL_PATH / "decision_tree.pkl",
        "Random Forest": MODEL_PATH / "random_forest.pkl",
        "XGBoost": MODEL_PATH / "xgboost.pkl"
    }

    model = joblib.load(model_paths[best_model_name])
    scaler = joblib.load(PREPROCESSING_PATH / "scaler.pkl")
    feature_names = joblib.load(PREPROCESSING_PATH / "feature_names.pkl")

    return model, scaler, feature_names, best_model_name


def get_risk_level(probability):
    if probability < 0.30:
        return "Low Risk", "🟢"
    elif probability < 0.70:
        return "Medium Risk", "🟡"
    else:
        return "High Risk", "🔴"


def get_recommendation(probability, tool_wear, torque, temp_diff):
    if probability >= 0.70:
        return "Immediate maintenance required. Inspect tool wear, torque load, and cooling system."
    elif tool_wear > 180:
        return "Tool wear is high. Schedule tool inspection or replacement."
    elif torque > 60:
        return "Torque load is high. Inspect mechanical load and transmission system."
    elif temp_diff > 12:
        return "Temperature difference is high. Check cooling and heat dissipation system."
    elif probability >= 0.30:
        return "Medium risk detected. Schedule preventive maintenance soon."
    else:
        return "Machine is operating normally. Continue routine monitoring."


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🏭 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "🤖 Predict Failure",
        "📊 Analytics",
        "🧠 Explainability",
        "ℹ️ About"
    ]
)

# -----------------------------
# Home Page
# -----------------------------
if page == "🏠 Home":
    st.markdown('<div class="main-title">🏭 Smart Manufacturing Predictive Maintenance System</div>', unsafe_allow_html=True)

    st.write("""
    An AI-powered manufacturing analytics platform that predicts machine failure risk,
    calculates machine health score, and provides maintenance recommendations.
    """)

    model, scaler, feature_names, best_model_name = load_model_assets()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Dataset Records", "10,000")
    col2.metric("Best Model", best_model_name)
    col3.metric("Features Used", len(feature_names))
    col4.metric("System Status", "Active ✅")

    st.divider()

    st.subheader("Project Workflow")

    st.markdown("""
    1. Data Collection using AI4I 2020 Predictive Maintenance Dataset  
    2. Exploratory Data Analysis  
    3. Data Cleaning and Feature Engineering  
    4. Model Training and Evaluation  
    5. Explainable AI using SHAP  
    6. Streamlit Dashboard Deployment  
    """)

# -----------------------------
# Prediction Page
# -----------------------------
elif page == "🤖 Predict Failure":
    model, scaler, feature_names, best_model_name = load_model_assets()

    st.title("🤖 Machine Failure Prediction")

    st.write(f"Using best trained model: **{best_model_name}**")

    col1, col2 = st.columns(2)

    with col1:
        machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
        air_temp = st.number_input("Air Temperature [K]", 250.0, 350.0, 298.0)
        process_temp = st.number_input("Process Temperature [K]", 250.0, 400.0, 308.0)

    with col2:
        rotational_speed = st.number_input("Rotational Speed [rpm]", 500, 3000, 1500)
        torque = st.number_input("Torque [Nm]", 0.0, 100.0, 40.0)
        tool_wear = st.number_input("Tool Wear [min]", 0, 300, 100)

    if st.button("Predict Machine Failure", type="primary"):
        type_mapping = {"L": 0, "M": 1, "H": 2}

        temp_diff = process_temp - air_temp
        mechanical_power = rotational_speed * torque
        wear_rate = tool_wear / (rotational_speed + 1e-6)
        thermal_stress = temp_diff * torque

        input_data = pd.DataFrame([{
            "Type": type_mapping[machine_type],
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "TWF": 0,
            "HDF": 0,
            "PWF": 0,
            "OSF": 0,
            "RNF": 0,
            "Temperature Difference": temp_diff,
            "Mechanical Power": mechanical_power,
            "Wear Rate": wear_rate,
            "Thermal Stress": thermal_stress
        }])

        input_data = input_data[feature_names]
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        health_score = round((1 - probability) * 100, 2)
        risk_level, risk_icon = get_risk_level(probability)

        recommendation = get_recommendation(
            probability,
            tool_wear,
            torque,
            temp_diff
        )

        st.divider()
        st.subheader("Prediction Results")

        c1, c2, c3 = st.columns(3)

        c1.metric("Failure Probability", f"{probability * 100:.2f}%")
        c2.metric("Machine Health Score", f"{health_score}%")
        c3.metric("Risk Level", f"{risk_icon} {risk_level}")

        if prediction == 1:
            st.error("⚠️ Machine Failure Risk Detected")
        else:
            st.success("✅ Machine Operating Normally")

        st.info(f"🔧 Recommendation: {recommendation}")

        with st.expander("View Engineered Input Features"):
            st.dataframe(input_data)

# -----------------------------
# Analytics Page
# -----------------------------
elif page == "📊 Analytics":
    st.title("📊 Model Analytics")

    eval_file = REPORTS_PATH / "model_evaluation_results.csv"
    training_file = REPORTS_PATH / "model_training_time_comparison.csv"

    if eval_file.exists():
        results_df = pd.read_csv(eval_file)

        st.subheader("Model Evaluation Results")
        st.dataframe(results_df)

        metric = st.selectbox(
            "Select Metric",
            ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=results_df, x="Model", y=metric, ax=ax)
        ax.set_title(f"{metric} Comparison")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=15)
        st.pyplot(fig)

    else:
        st.warning("Model evaluation results file not found.")

    st.divider()

    if training_file.exists():
        training_df = pd.read_csv(training_file)

        st.subheader("Training Time Comparison")
        st.dataframe(training_df)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=training_df,
            x="Model",
            y="Training Time (seconds)",
            ax=ax
        )
        ax.set_title("Training Time Comparison")
        ax.tick_params(axis="x", rotation=15)
        st.pyplot(fig)

    else:
        st.warning("Training time comparison file not found.")

# -----------------------------
# Explainability Page
# -----------------------------
elif page == "🧠 Explainability":
    st.title("🧠 Explainability & Feature Insights")

    feature_file = REPORTS_PATH / "best_model_feature_importance.csv"
    shap_file = REPORTS_PATH / "shap_feature_importance.csv"

    if feature_file.exists():
        importance_df = pd.read_csv(feature_file)

        st.subheader("Best Model Feature Importance")
        st.dataframe(importance_df)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=importance_df.head(10),
            x="Importance",
            y="Feature",
            ax=ax
        )
        ax.set_title("Top Feature Importance")
        st.pyplot(fig)

    else:
        st.warning("Feature importance file not found.")

    st.divider()

    if shap_file.exists():
        shap_df = pd.read_csv(shap_file)

        st.subheader("SHAP Feature Importance")
        st.dataframe(shap_df)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=shap_df.head(10),
            x="Mean SHAP Value",
            y="Feature",
            ax=ax
        )
        ax.set_title("Top SHAP Feature Contributions")
        st.pyplot(fig)

    else:
        st.warning("SHAP feature importance file not found.")

# -----------------------------
# About Page
# -----------------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About Project")

    st.markdown("""
    ## Smart Manufacturing Data Analytics and Predictive Maintenance System Using ML

    This project predicts machine failure risk using sensor-based manufacturing data.

    ### Dataset
    AI4I 2020 Predictive Maintenance Dataset

    ### Technologies Used
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - XGBoost
    - SHAP
    - Streamlit
    - GitHub

    ### Main Features
    - Machine failure prediction
    - Failure probability
    - Machine health score
    - Risk level classification
    - Maintenance recommendation
    - Model analytics
    - Explainable AI insights
    """)
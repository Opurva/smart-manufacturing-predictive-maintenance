import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Path Setup
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "trained_models"
PREPROCESSING_PATH = BASE_DIR / "models" / "preprocessing"

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


# -----------------------------
# Helper Functions
# -----------------------------
def calculate_health_score(failure_probability):
    health_score = round((1 - failure_probability) * 100, 2)
    return health_score


def get_risk_level(failure_probability):
    if failure_probability < 0.30:
        return "Low Risk", "🟢"
    elif failure_probability < 0.70:
        return "Medium Risk", "🟡"
    else:
        return "High Risk", "🔴"


def get_recommendation(failure_probability, tool_wear, torque, temp_diff):
    if failure_probability >= 0.70:
        return "Immediate maintenance required. Inspect tool wear, torque load, and cooling system."
    elif tool_wear > 180:
        return "Tool wear is high. Schedule tool inspection or replacement."
    elif torque > 60:
        return "Torque load is high. Inspect mechanical load and transmission system."
    elif temp_diff > 12:
        return "Temperature difference is high. Check cooling and heat dissipation system."
    elif failure_probability >= 0.30:
        return "Medium risk detected. Schedule preventive maintenance soon."
    else:
        return "Machine is operating normally. Continue routine monitoring."


# -----------------------------
# Page UI
# -----------------------------
st.title("🤖 Machine Failure Prediction")

st.markdown("""
Enter machine operating conditions to predict failure risk, machine health score,
and maintenance recommendation.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

    air_temp = st.number_input(
        "Air Temperature [K]",
        min_value=250.0,
        max_value=350.0,
        value=298.0
    )

    process_temp = st.number_input(
        "Process Temperature [K]",
        min_value=250.0,
        max_value=400.0,
        value=308.0
    )

with col2:
    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=500,
        max_value=3000,
        value=1500
    )

    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=100.0,
        value=40.0
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0,
        max_value=300,
        value=100
    )


type_mapping = {
    "L": 0,
    "M": 1,
    "H": 2
}


if st.button("Predict Machine Failure", type="primary"):

    type_encoded = type_mapping[machine_type]

    temperature_difference = process_temp - air_temp
    mechanical_power = rotational_speed * torque
    wear_rate = tool_wear / (rotational_speed + 1e-6)
    thermal_stress = temperature_difference * torque

    input_data = pd.DataFrame([{
        "Type": type_encoded,
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
        "Temperature Difference": temperature_difference,
        "Mechanical Power": mechanical_power,
        "Wear Rate": wear_rate,
        "Thermal Stress": thermal_stress
    }])

    input_data = input_data[feature_names]

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    failure_probability = model.predict_proba(input_scaled)[0][1]

    health_score = calculate_health_score(failure_probability)
    risk_level, risk_icon = get_risk_level(failure_probability)

    recommendation = get_recommendation(
        failure_probability,
        tool_wear,
        torque,
        temperature_difference
    )

    st.divider()

    st.subheader("Prediction Results")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Failure Probability",
            f"{failure_probability * 100:.2f}%"
        )

    with c2:
        st.metric(
            "Machine Health Score",
            f"{health_score}%"
        )

    with c3:
        st.metric(
            "Risk Level",
            f"{risk_icon} {risk_level}"
        )

    if prediction == 1:
        st.error("⚠️ Machine Failure Risk Detected")
    else:
        st.success("✅ Machine Operating Normally")

    st.info(f"🔧 Recommendation: {recommendation}")

    with st.expander("View Input Features"):
        st.dataframe(input_data)
import streamlit as st

st.set_page_config(
    page_title="Smart Manufacturing Predictive Maintenance",
    page_icon="🏭",
    layout="wide"
)

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

if page == "🏠 Home":
    st.title("🏭 Smart Manufacturing Predictive Maintenance System")
    st.write("Welcome to the Smart Manufacturing Predictive Maintenance dashboard.")
    st.success("System ready.")

elif page == "🤖 Predict Failure":
    import pandas as pd
    import joblib
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[1]

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

    st.title("🤖 Machine Failure Prediction")

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
        failure_probability = model.predict_proba(input_scaled)[0][1]
        health_score = round((1 - failure_probability) * 100, 2)

        if failure_probability < 0.30:
            risk = "🟢 Low Risk"
        elif failure_probability < 0.70:
            risk = "🟡 Medium Risk"
        else:
            risk = "🔴 High Risk"

        st.divider()
        st.subheader("Prediction Results")

        c1, c2, c3 = st.columns(3)
        c1.metric("Failure Probability", f"{failure_probability * 100:.2f}%")
        c2.metric("Machine Health Score", f"{health_score}%")
        c3.metric("Risk Level", risk)

        if prediction == 1:
            st.error("⚠️ Machine Failure Risk Detected")
        else:
            st.success("✅ Machine Operating Normally")

        if failure_probability >= 0.70:
            recommendation = "Immediate maintenance required. Inspect tool wear, torque load, and cooling system."
        elif tool_wear > 180:
            recommendation = "Tool wear is high. Schedule tool inspection or replacement."
        elif torque > 60:
            recommendation = "Torque load is high. Inspect mechanical load and transmission system."
        elif temp_diff > 12:
            recommendation = "Temperature difference is high. Check cooling and heat dissipation system."
        else:
            recommendation = "Machine is operating normally. Continue routine monitoring."

        st.info(f"🔧 Recommendation: {recommendation}")

        with st.expander("View Input Features"):
            st.dataframe(input_data)
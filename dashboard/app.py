import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from datetime import datetime
import plotly.express as px
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

def load_css(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css(BASE_DIR / "dashboard" / "assets" / "styles.css")

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
    
def create_health_gauge(health_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        title={"text": "Machine Health Score"},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#22c55e"},
            "steps": [
                {"range": [0, 40], "color": "#7f1d1d"},
                {"range": [40, 70], "color": "#713f12"},
                {"range": [70, 100], "color": "#064e3b"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "thickness": 0.75,
                "value": health_score
            }
        }
    ))

    fig.update_layout(
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"}
    )

    return fig


def save_prediction_history(input_data, probability, health_score, risk_level, recommendation):
    history_file = REPORTS_PATH / "prediction_history.csv"

    record = input_data.copy()
    record["Prediction Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record["Failure Probability"] = round(probability * 100, 2)
    record["Machine Health Score"] = health_score
    record["Risk Level"] = risk_level
    record["Recommendation"] = recommendation

    if history_file.exists():
        history_df = pd.read_csv(history_file)
        history_df = pd.concat([history_df, record], ignore_index=True)
    else:
        history_df = record

    history_df.to_csv(history_file, index=False)

    return history_df

def style_plotly_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.65)",
        font=dict(
            color="#e5e7eb",
            family="Arial"
        ),
        title=dict(
            font=dict(size=22, color="#f8fafc"),
            x=0.02
        ),
        margin=dict(l=40, r=30, t=70, b=40),
        hoverlabel=dict(
            bgcolor="#020617",
            font_size=13,
            font_color="#f8fafc"
        )
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        tickfont=dict(color="#cbd5e1")
    )

    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.18)",
        zeroline=False,
        tickfont=dict(color="#cbd5e1")
    )

    return fig

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="sidebar-title">🏭 SmartFactory AI</div>
    <div class="sidebar-subtitle">Company-Inspired Predictive Maintenance Prototype</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
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
    model, scaler, feature_names, best_model_name = load_model_assets()

    st.markdown("""
    <div class="hero">
        <div class="hero-badge">2026-Ready Industrial AI Prototype</div>
        <div class="hero-title">SmartFactory AI: Predictive Maintenance Intelligence Platform</div>
        <div class="hero-subtitle">
            A company-inspired industrial intelligence system that analyzes machine operating signals,
            predicts equipment failure risk, estimates health score, and recommends preventive maintenance
            actions before breakdown occurs.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Machine Records</div>
            <div class="metric-value">10,000</div>
            <div class="metric-subtitle">Representative industrial sensor records</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Active AI Model</div>
            <div class="metric-value" style="font-size:26px;">{best_model_name}</div>
            <div class="metric-subtitle">Selected using evaluation metrics</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Machine Signals</div>
            <div class="metric-value">{len(feature_names)}</div>
            <div class="metric-subtitle">Sensor + engineered features</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">System Mode</div>
            <div class="metric-value">Live</div>
            <div class="metric-subtitle">Prototype dashboard active</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Platform Capabilities</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Designed as a decision-support tool for smart manufacturing maintenance teams.</div>', unsafe_allow_html=True)

    cap1, cap2, cap3 = st.columns(3)

    with cap1:
        st.markdown("""
        <div class="premium-card">
            <div class="capability-icon">🤖</div>
            <div class="capability-title">Failure Risk Prediction</div>
            <div class="capability-text">
                Predicts whether a machine is likely to fail based on operating parameters such as temperature,
                speed, torque, and tool wear.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with cap2:
        st.markdown("""
        <div class="premium-card">
            <div class="capability-icon">🩺</div>
            <div class="capability-title">Machine Health Score</div>
            <div class="capability-text">
                Converts model output into a simple health score so maintenance teams can quickly understand
                machine condition.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with cap3:
        st.markdown("""
        <div class="premium-card">
            <div class="capability-icon">🔧</div>
            <div class="capability-title">Maintenance Recommendation</div>
            <div class="capability-text">
                Provides action-oriented maintenance suggestions based on risk probability and machine signal behavior.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Industrial ML Workflow</div>', unsafe_allow_html=True)

    workflow = [
        ("01", "Problem Framing", "Manufacturing downtime and machine failure risk analysis"),
        ("02", "Sensor Data Preparation", "Machine operating signals cleaned and transformed"),
        ("03", "Feature Engineering", "Thermal stress, mechanical power, wear rate, and temperature difference created"),
        ("04", "Model Training", "Multiple ML models trained and benchmarked"),
        ("05", "Model Evaluation", "Models compared using precision, recall, F1 score, ROC-AUC, and confusion matrix"),
        ("06", "Explainable AI", "Feature importance and SHAP-based insights generated"),
        ("07", "Dashboard System", "Interactive predictive maintenance dashboard built for decision support")
    ]

    for number, title, desc in workflow:
        st.markdown(f"""
        <div class="timeline-step">
            <div class="timeline-number">{number}</div>
            <div>
                <div class="timeline-title">{title}</div>
                <div class="timeline-desc">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# Prediction Page
# -----------------------------
elif page == "🤖 Predict Failure":
    model, scaler, feature_names, best_model_name = load_model_assets()

    st.title("🤖 Predictive Maintenance Engine")

    st.write(f"Active AI Model: **{best_model_name}**")
    st.caption("Enter machine operating parameters to estimate failure risk and maintenance priority.")
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

        history_df = save_prediction_history(
            input_data,
            probability,
            health_score,
            risk_level,
            recommendation
        )

        st.divider()

        st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)

        if probability < 0.30:
            card_class = "result-card-low"
        elif probability < 0.70:
            card_class = "result-card-medium"
        else:
            card_class = "result-card-high"

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class="{card_class}">
                <div class="result-title">Failure Probability</div>
                <div class="result-value">{probability * 100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="{card_class}">
                <div class="result-title">Machine Health Score</div>
                <div class="result-value">{health_score}%</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="{card_class}">
                <div class="result-title">Risk Level</div>
                <div class="result-value">{risk_icon} {risk_level}</div>
            </div>
            """, unsafe_allow_html=True)

        st.progress(int(health_score))

        g1, g2 = st.columns([1, 1])

        with g1:
            st.plotly_chart(
                create_health_gauge(health_score),
                use_container_width=True
            )

        with g2:
            st.markdown("### Failure Risk Indicator")
            st.progress(int(probability * 100))

            st.markdown(f"""
            <div class="glass-card">
                <b>Failure Probability:</b> {probability * 100:.2f}%<br>
                <b>Machine Health:</b> {health_score}%<br>
                <b>Risk Category:</b> {risk_icon} {risk_level}
            </div>
            """, unsafe_allow_html=True)

        if prediction == 1:
            st.error("⚠️ Machine Failure Risk Detected")
        else:
            st.success("✅ Machine Operating Normally")

        st.markdown(f"""
        <div class="recommendation-box">
            <b>🔧 Maintenance Recommendation</b><br>
            {recommendation}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("View Processed Machine Signal Features"):
            st.dataframe(input_data)

        st.divider()

        st.markdown('<div class="section-title">Prediction History</div>', unsafe_allow_html=True)

        history_file = REPORTS_PATH / "prediction_history.csv"

        if history_file.exists():
            history_df = pd.read_csv(history_file)
            st.dataframe(history_df.tail(10), use_container_width=True)

            csv_data = history_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction History",
                data=csv_data,
                file_name="prediction_history.csv",
                mime="text/csv"
            )
        else:
            st.info("No prediction history available yet.")

# -----------------------------
# Analytics Page
# -----------------------------
elif page == "📊 Analytics":
    st.title("📊 Predictive Maintenance Analytics")

    st.caption(
        "Model performance analysis, training efficiency, and predictive maintenance benchmarking."
    )

    eval_file = REPORTS_PATH / "model_evaluation_results.csv"
    training_file = REPORTS_PATH / "model_training_time_comparison.csv"

    if eval_file.exists():
        results_df = pd.read_csv(eval_file)

        # -----------------------------
        # KPI Cards
        # -----------------------------
        best_model_row = results_df.sort_values(
            by="F1 Score",
            ascending=False
        ).iloc[0]

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Best Model</div>
                <div class="metric-value">{best_model_row["Model"]}</div>
                <div class="metric-subtitle">Selected by F1 Score</div>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{best_model_row["Accuracy"]:.2%}</div>
                <div class="metric-subtitle">Overall correct predictions</div>
            </div>
            """, unsafe_allow_html=True)

        with k3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value">{best_model_row["Recall"]:.2%}</div>
                <div class="metric-subtitle">Failure detection ability</div>
            </div>
            """, unsafe_allow_html=True)

        with k4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">F1 Score</div>
                <div class="metric-value">{best_model_row["F1 Score"]:.2%}</div>
                <div class="metric-subtitle">Balanced model performance</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # -----------------------------
        # Evaluation Table
        # -----------------------------
        st.markdown(
            '<div class="section-title">Model Evaluation Results</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            results_df,
            use_container_width=True
        )

        # -----------------------------
        # Metric Comparison Chart
        # -----------------------------
        st.markdown(
            '<div class="section-title">Performance Comparison</div>',
            unsafe_allow_html=True
        )

        metric = st.selectbox(
            "Select Evaluation Metric",
            ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
        )

        plot_df = results_df.copy()
        plot_df["Metric Label"] = (plot_df[metric] * 100).round(2).astype(str) + "%"

        fig = px.bar(
            plot_df,
            x="Model",
            y=metric,
            text="Metric Label",
            color=metric,
            color_continuous_scale=["#ef4444", "#eab308", "#22c55e"],
            title=f"{metric} Comparison Across Models"
        )

        fig.update_traces(
            textposition="outside",
            marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>" + metric + ": %{y:.2%}<extra></extra>"
        )

        fig.update_layout(
            yaxis_range=[0, 1.08],
            coloraxis_showscale=False,
            height=460
        )

        fig = style_plotly_chart(fig)

        st.plotly_chart(fig, use_container_width=True)


        # -----------------------------
        # Full Metric Heatmap
        # -----------------------------
        st.markdown(
            '<div class="section-title">Model Performance Heatmap</div>',
            unsafe_allow_html=True
        )

        heatmap_data = results_df.set_index("Model")[
            ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
        ]

        fig = px.imshow(
            heatmap_data,
            text_auto=".3f",
            aspect="auto",
            color_continuous_scale="YlGnBu",
            title="Model Performance Heatmap",
            labels=dict(
                x="Evaluation Metric",
                y="Machine Learning Model",
                color="Score"
            )
        )

        fig.update_layout(
            height=430,
            coloraxis_colorbar=dict(
                title="Score"
            )
        )

        fig = style_plotly_chart(fig)

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Model evaluation results file not found.")

    st.divider()

    # -----------------------------
    # Training Time Section
    # -----------------------------
    if training_file.exists():
        training_df = pd.read_csv(training_file)

        st.markdown(
            '<div class="section-title">Training Time Analysis</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            training_df,
            use_container_width=True
        )

        training_plot_df = training_df.copy()
        training_plot_df["Time Label"] = training_plot_df["Training Time (seconds)"].round(4).astype(str) + "s"

        fig = px.bar(
            training_plot_df,
            x="Model",
            y="Training Time (seconds)",
            text="Time Label",
            color="Training Time (seconds)",
            color_continuous_scale=["#22c55e", "#38bdf8", "#6366f1"],
            title="Training Time Comparison"
        )

        fig.update_traces(
            textposition="outside",
            marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>Training Time: %{y:.4f}s<extra></extra>"
        )

        fig.update_layout(
            coloraxis_showscale=False,
            height=460
        )

        fig = style_plotly_chart(fig)

        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Training time helps compare computational efficiency. "
            "However, for predictive maintenance, recall and F1 score are usually more important than speed alone."
        )

    else:
        st.warning("Training time comparison file not found.")

# -----------------------------
# Explainability Page
# -----------------------------
elif page == "🧠 Explainability":
    st.title("🧠 Explainable AI Insights")

    st.caption(
        "Understand which machine signals influence failure prediction and maintenance decisions."
    )

    feature_file = REPORTS_PATH / "best_model_feature_importance.csv"
    shap_file = REPORTS_PATH / "shap_feature_importance.csv"

    # -----------------------------
    # Feature Importance Section
    # -----------------------------
    if feature_file.exists():
        importance_df = pd.read_csv(feature_file)

        st.markdown(
            '<div class="section-title">Key Machine Signals Influencing Prediction</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            importance_df,
            use_container_width=True
        )

        value_col = "Importance" if "Importance" in importance_df.columns else "Mean SHAP Value"

        top_importance = importance_df.head(10).copy()

        fig = px.bar(
            top_importance.sort_values(by=value_col, ascending=True),
            x=value_col,
            y="Feature",
            orientation="h",
            text=value_col,
            color=value_col,
            color_continuous_scale=["#38bdf8", "#6366f1", "#22c55e"],
            title="Top 10 Important Machine Signals"
        )

        fig.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside",
            marker_line_width=0,
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
        )

        fig.update_layout(
            height=520,
            coloraxis_showscale=False,
            xaxis_title="Importance Score",
            yaxis_title="Machine Signal"
        )

        fig = style_plotly_chart(fig)

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Feature importance file not found.")

    st.divider()

    # -----------------------------
    # SHAP Importance Section
    # -----------------------------
    if shap_file.exists():
        shap_df = pd.read_csv(shap_file)

        st.markdown(
            '<div class="section-title">SHAP-Based Sensor Contribution Analysis</div>',
            unsafe_allow_html=True
        )

        st.markdown("""
        SHAP values explain how much each feature contributes to the model's prediction.
        Higher SHAP values indicate stronger influence on machine failure risk prediction.
        """)

        st.dataframe(
            shap_df,
            use_container_width=True
        )

        top_shap = shap_df.head(10).copy()

        fig = px.bar(
            top_shap.sort_values(by="Mean SHAP Value", ascending=True),
            x="Mean SHAP Value",
            y="Feature",
            orientation="h",
            text="Mean SHAP Value",
            color="Mean SHAP Value",
            color_continuous_scale=["#0ea5e9", "#8b5cf6", "#ec4899"],
            title="Top 10 SHAP Feature Contributions"
        )

        fig.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside",
            marker_line_width=0,
            hovertemplate="<b>%{y}</b><br>Mean SHAP Value: %{x:.4f}<extra></extra>"
        )

        fig.update_layout(
            height=520,
            coloraxis_showscale=False,
            xaxis_title="Mean SHAP Value",
            yaxis_title="Machine Signal"
        )

        fig = style_plotly_chart(fig)

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # Top Signal Cards
        # -----------------------------
        st.markdown(
            '<div class="section-title">Top Predictive Signals</div>',
            unsafe_allow_html=True
        )

        top3 = shap_df.head(3)

        c1, c2, c3 = st.columns(3)

        cards = [c1, c2, c3]

        for i, (_, row) in enumerate(top3.iterrows()):
            with cards[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Rank {i + 1}</div>
                    <div class="metric-value" style="font-size:24px;">{row["Feature"]}</div>
                    <div class="metric-subtitle">SHAP Value: {row["Mean SHAP Value"]:.4f}</div>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.warning("SHAP feature importance file not found.")

    st.divider()

    st.info(
        "Explainable AI improves trust by showing which machine parameters influenced the prediction. "
        "This helps maintenance teams understand whether risk is driven by load, temperature, tool wear, or engineered stress features."
    )

# -----------------------------
# About Page
# -----------------------------
elif page == "ℹ️ About":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">Project Overview</div>
        <div class="hero-title">About SmartFactory AI</div>
        <div class="hero-subtitle">
            A company-inspired predictive maintenance prototype built to demonstrate how machine learning
            can support smart manufacturing operations through failure prediction, health monitoring,
            and maintenance intelligence.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Project Context</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        Manufacturing companies depend on continuous machine operation. Unexpected machine failures can lead to
        production delays, increased maintenance cost, quality issues, and operational downtime.
        <br><br>
        This project demonstrates how machine learning can be used to analyze machine operating parameters and
        identify possible failure risk before breakdown occurs. The system is designed as a prototype that can be
        adapted to real company machine logs when live sensor data or historical maintenance records become available.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Data Usage Note</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="note-card">
        Due to confidentiality and limited access to real company machine logs, representative industrial machine
        sensor records were used for prototype development and testing.
        <br><br>
        The same pipeline can be integrated with actual company machinery once real sensor logs, machine history,
        and maintenance records become available.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Core Features</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("""
        <div class="premium-card">
            <div class="capability-icon">📡</div>
            <div class="capability-title">Sensor-Based Monitoring</div>
            <div class="capability-text">
                Uses machine operating parameters such as temperature, rotational speed, torque, and tool wear.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown("""
        <div class="premium-card">
            <div class="capability-icon">🧠</div>
            <div class="capability-title">Machine Learning Engine</div>
            <div class="capability-text">
                Trains and evaluates multiple ML models to identify the best predictive maintenance model.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with f3:
        st.markdown("""
        <div class="premium-card">
            <div class="capability-icon">🔍</div>
            <div class="capability-title">Explainable AI</div>
            <div class="capability-text">
                Uses feature importance and SHAP-based analysis to explain which signals influence predictions.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Technology Stack</div>', unsafe_allow_html=True)

    tech1, tech2, tech3, tech4 = st.columns(4)

    tech_items = [
        ("Python", "Core programming"),
        ("Scikit-learn", "ML modeling"),
        ("XGBoost", "Boosting model"),
        ("Streamlit", "Dashboard UI")
    ]

    for col, (name, use) in zip([tech1, tech2, tech3, tech4], tech_items):
        with col:
            st.markdown(f"""
            <div class="tech-card">
                <div class="tech-title">{name}</div>
                <div class="tech-desc">{use}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Project Purpose</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        The purpose of this project is to reduce unplanned downtime, support preventive maintenance,
        and demonstrate how data analytics and machine learning can be applied in smart manufacturing environments.
        <br><br>
        This prototype can be extended with live IoT sensors, real-time dashboards, alert systems, and company-specific
        maintenance rules in future deployment.
    </div>
    """, unsafe_allow_html=True)

#app.py    

import streamlit as st
from model_loader import load_model_and_scaler
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from threshold import health_status

# ================= CONFIG =================
SEQUENCE_LENGTH = 50
N_FEATURES = 24
DATA_DIR = "data"
CAP_RUL = 125

# ================= PAGE =================
st.set_page_config(
    page_title="PrognosAI - Predictive Maintenance",
    layout="wide"
)

st.title("🔧 PrognosAI – Predictive Maintenance Dashboard")

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Configuration")

model_name = st.sidebar.selectbox(
    "Select Model",
    ["FD001", "FD002", "FD003", "FD004"]
)

@st.cache_resource
def get_model(fd):
    return load_model_and_scaler(fd.lower())

model, scaler = get_model(model_name)
st.sidebar.success(f"✅ {model_name} model loaded")

test_files = sorted(f for f in os.listdir(DATA_DIR) if f.lower().startswith("test"))
rul_files = sorted(f for f in os.listdir(DATA_DIR) if f.lower().startswith("rul"))

test_file = st.sidebar.selectbox("📂 Test File", test_files)
rul_file = st.sidebar.selectbox("⏱️ RUL File", rul_files)

run = st.sidebar.button("▶ RUN")

# ================= FUNCTIONS =================
def load_cmapss_test(path):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    engine_ids = df.iloc[:, 0].values
    X = df.iloc[:, 2:2 + N_FEATURES].values.astype("float32")
    return X, engine_ids


def predict_rul(model, scaler, X, engine_ids):
    preds = []
    for eng in np.unique(engine_ids):
        idx = np.where(engine_ids == eng)[0]
        data = scaler.transform(X[idx])

        if len(data) < SEQUENCE_LENGTH:
            pad = np.tile(data[0], (SEQUENCE_LENGTH - len(data), 1))
            data = np.vstack([pad, data])

        seq = data[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, N_FEATURES)
        preds.append(np.clip(model.predict(seq, verbose=0)[0][0], 0, CAP_RUL))
    return np.array(preds)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ================= MAIN =================
if run:
    st.markdown("---")
    st.header("📊 Model Results")

    X_test, engine_ids = load_cmapss_test(os.path.join(DATA_DIR, test_file))
    y_true = pd.read_csv(os.path.join(DATA_DIR, rul_file), header=None).iloc[:, 0].values
    y_true = np.clip(y_true, 0, CAP_RUL)

    y_pred = predict_rul(model, scaler, X_test, engine_ids)

    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    st.metric("RMSE", f"{rmse(y_true, y_pred):.2f}")

    # ================= SCATTER (SMALLER) =================
    st.subheader("🔹 True vs Predicted RUL")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, y_pred, alpha=0.6)
    ax.plot([0, CAP_RUL], [0, CAP_RUL], "k--", lw=2)
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    # ================= LINE PLOT =================
    st.subheader("📈 Predicted vs Actual RUL")

    n_plot = min(len(y_true), 200)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(y_true[:n_plot], label="Actual RUL", lw=2)
    ax.plot(y_pred[:n_plot], label="Predicted RUL", lw=2)
    ax.set_xlabel("Sample")
    ax.set_ylabel("RUL (capped)")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    # ================= DONUT + CONFIG TABLE =================
    st.subheader("🚨 Alert Distribution")

    alerts = [health_status(v).split()[0].capitalize() for v in y_pred]

    labels = ["Normal", "Warning", "Critical"]
    counts = [alerts.count(l) for l in labels]
    colors = ["#2ecc71", "#f1c40f", "#e74c3c"]

    col1, col2 = st.columns([2, 3])

    # ---- Donut ----
    with col1:
        fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=120)
        ax.pie(
            counts,
            colors=colors,
            autopct="%1.0f%%",
            pctdistance=0.75,   # 🔥 INSIDE
            startangle=90,
            wedgeprops=dict(width=0.4)
        )
        ax.text(0, 0, "Alerts", ha="center", va="center", fontsize=12, weight="bold")
        ax.axis("equal")
        st.pyplot(fig)

    # ---- Configuration Summary TABLE ----
    with col2:
        st.markdown("### 📌 Configuration Summary")
        config_df = pd.DataFrame({
            "Parameter": [
                "Selected Model",
                "Test File",
                "RUL File",
                "Warning Threshold",
                "Critical Threshold",
                "Total Engines"
            ],
            "Value": [
                model_name,
                test_file,
                rul_file,
                30,
                10,
                len(y_pred)
            ]
        })
        st.table(config_df)

    # ================= HEALTH TABLE =================
    st.markdown("---")
    st.subheader(f"🩺 HEALTH STATUS ALERTS FOR TEST UNITS ({model_name})")

    df_health = pd.DataFrame({
        "Unit": np.arange(1, len(y_pred) + 1),
        "True RUL": y_true.round(2),
        "Pred RUL": y_pred.round(2),
        "Error": np.abs(y_true - y_pred).round(2),
        "Status": [health_status(v) for v in y_pred]
    })

    st.dataframe(df_health, use_container_width=True, hide_index=True)
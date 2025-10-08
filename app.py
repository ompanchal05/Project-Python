import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# ==============================
# 🚦 Rule-Based Pump Model
# ==============================
class PumpModel:
    def predict(self, X):
        """
        X: numpy array of shape (n_samples, 3)
           Columns = [temperature, vibration, pressure]
        Returns: list of predictions (0 = Healthy, 1 = Failure Risk)
        """
        predictions = []
        for row in X:
            temp, vib, pres = row
            # Simple safety rules
            if temp > 85 or vib > 8 or pres < 45 or pres > 65:
                predictions.append(1)  # Failure risk
            else:
                predictions.append(0)  # Healthy
        return predictions

# ==============================
# 🎯 Load or Create Model
# ==============================
MODEL_FILE = "pump_model.pkl"

if os.path.exists(MODEL_FILE):
    try:
        model = pickle.load(open(MODEL_FILE, "rb"))
    except Exception as e:
        st.error("❌ Could not load model. Creating a new rule-based model...")
        model = PumpModel()
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model, f)
else:
    model = PumpModel()
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

# ==============================
# ⚙️ Page Configuration
# ==============================
st.set_page_config(page_title="PumpGuard – AI for Pumps", page_icon="🛠️", layout="centered")

# ==============================
# 🎨 Custom CSS
# ==============================
st.markdown("""
    <style>
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
    }
    .safe {color: green; font-weight: bold;}
    .warn {color: orange; font-weight: bold;}
    .danger {color: red; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# ==============================
# 🛠️ App Title
# ==============================
st.title("🛠️ PumpGuard – Safeguarding Pumps with AI")
st.write("An AI-powered predictive maintenance tool to keep pumps **safe, reliable, and efficient**.")

# ==============================
# 📊 Sensor Inputs
# ==============================
st.subheader("📊 Enter Sensor Readings")

col1, col2, col3 = st.columns(3)

with col1:
    temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=500.0, value=75.0, step=1.0)

with col2:
    vibration = st.number_input("📳 Vibration (mm/s)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

with col3:
    pressure = st.number_input("💨 Pressure (bar)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)

# ==============================
# 🚦 Safety Check Function
# ==============================
def check_safety(temp, vib, pres):
    reasons = []

    # Temperature
    if 40 <= temp <= 85:
        st.markdown(f"🌡️ Temperature: <span class='safe'>Safe ({temp} °C)</span>", unsafe_allow_html=True)
    elif 85 < temp <= 100:
        st.markdown(f"🌡️ Temperature: <span class='warn'>Warning ({temp} °C) – High</span>", unsafe_allow_html=True)
        reasons.append("High Temperature")
    else:
        st.markdown(f"🌡️ Temperature: <span class='danger'>Danger ({temp} °C)</span>", unsafe_allow_html=True)
        reasons.append("Critical Overheating")

    # Vibration
    if vib <= 4.5:
        st.markdown(f"📳 Vibration: <span class='safe'>Safe ({vib} mm/s)</span>", unsafe_allow_html=True)
    elif 4.5 < vib <= 7.1:
        st.markdown(f"📳 Vibration: <span class='warn'>Warning ({vib} mm/s) – Elevated</span>", unsafe_allow_html=True)
        reasons.append("Moderate Vibration")
    elif 7.1 < vib <= 11.2:
        st.markdown(f"📳 Vibration: <span class='warn'>Unsatisfactory ({vib} mm/s) – Possible Bearing Issue</span>", unsafe_allow_html=True)
        reasons.append("High Vibration")
    else:
        st.markdown(f"📳 Vibration: <span class='danger'>Danger ({vib} mm/s) – Severe</span>", unsafe_allow_html=True)
        reasons.append("Critical Vibration")

    # Pressure
    if 40 <= pres <= 60:
        st.markdown(f"💨 Pressure: <span class='safe'>Safe ({pres} bar)</span>", unsafe_allow_html=True)
    elif 30 <= pres < 40 or 60 < pres <= 70:
        st.markdown(f"💨 Pressure: <span class='warn'>Warning ({pres} bar)</span>", unsafe_allow_html=True)
        reasons.append("Abnormal Pressure")
    else:
        st.markdown(f"💨 Pressure: <span class='danger'>Danger ({pres} bar)</span>", unsafe_allow_html=True)
        reasons.append("Critical Pressure")

    return reasons

# ==============================
# 🤖 Prediction & Logging
# ==============================
if st.button("🔍 Predict Pump Health"):
    st.subheader("🧪 Sensor Safety Check")
    fail_reasons = check_safety(temperature, vibration, pressure)

    features = np.array([[temperature, vibration, pressure]])
    try:
        prediction = model.predict(features)
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
        st.stop()

    # Display Result
    st.subheader("🧠 AI Prediction Result")
    if prediction[0] == 1:
        st.markdown(f"""
            <div class="prediction-box" style="background-color:#ffe6e6; color:#b30000;">
            ⚠️ **Warning:** Pump is at **Risk of Failure**!  
            🔧 Issues detected: {", ".join(fail_reasons) if fail_reasons else "Unknown – check sensors."}
            </div>
        """, unsafe_allow_html=True)
        status = "Failure Risk"
    else:
        st.markdown(f"""
            <div class="prediction-box" style="background-color:#e6ffe6; color:#006600;">
            ✅ Pump is **Healthy & Stable**.  
            👍 No immediate issues detected.
            </div>
        """, unsafe_allow_html=True)
        status = "Healthy"

    # ==============================
    # 📂 Save History
    # ==============================
    log_entry = pd.DataFrame([{
        "Temperature": temperature,
        "Vibration": vibration,
        "Pressure": pressure,
        "Prediction": status
    }])

    if os.path.exists("history.csv"):
        log_entry.to_csv("history.csv", mode="a", header=False, index=False)
    else:
        log_entry.to_csv("history.csv", index=False)

# ==============================
# 📊 View History & Visualization
# ==============================
st.markdown("---")
st.subheader("📊 Prediction History")

if os.path.exists("history.csv"):
    history = pd.read_csv("history.csv")
    st.dataframe(history)

    # Plot Trends
    st.subheader("📈 Sensor Trends")
    fig, ax = plt.subplots()
    history[["Temperature", "Vibration", "Pressure"]].plot(ax=ax)
    plt.xlabel("Record #")
    plt.ylabel("Sensor Values")
    st.pyplot(fig)
else:
    st.info("No history available yet. Run a prediction first.")

# ==============================
# 📌 Footer
# ==============================
st.markdown("---")
st.caption("🚀 Developed with ❤️ using **Streamlit & Rule-Based AI** | Project: PumpGuard")
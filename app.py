
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Motor Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    package = joblib.load("model/motor_bearing_model.joblib")
    return package

try:
    package = load_model()
    model = package["model"]
    feature_names = package["feature_names"]
    class_names = package["class_names"]
    segment_size = package["segment_size"]
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# Dashboard title
st.title("⚙️ Motor Predictive Maintenance")
st.subheader("Machine Learning-Based Bearing Fault Detection")

st.write(
    "Upload vibration data to identify possible motor bearing conditions."
)

st.info(
    "This is an offline dataset-based demonstration. "
    "Live ESP32 sensor integration is planned as a future extension."
)

# Sidebar
st.sidebar.header("Project Information")
st.sidebar.write("Algorithm: Random Forest")
st.sidebar.write("Dataset: Case Western Reserve University (CWRU)")
st.sidebar.write(f"Window size: {segment_size} samples")

# Upload vibration data
uploaded_file = st.file_uploader(
    "Upload vibration data (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(df.head())

        # Select signal column
        signal_column = st.selectbox(
            "Select the vibration signal column",
            df.columns
        )

        signal = pd.to_numeric(
            df[signal_column],
            errors="coerce"
        ).dropna().to_numpy()

        if len(signal) < segment_size:
            st.warning(
                f"Please upload at least {segment_size} valid samples."
            )
        else:
            # Divide signal into segments
            features_list = []

            for start in range(0, len(signal) - segment_size + 1, segment_size):
                segment = signal[start:start + segment_size]

                features_list.append({
                    "Mean": np.mean(segment),
                    "Standard Deviation": np.std(segment),
                    "RMS": np.sqrt(np.mean(segment ** 2)),
                    "Variance": np.var(segment),
                    "Peak Amplitude": np.max(np.abs(segment))
                })

            feature_df = pd.DataFrame(features_list)
            feature_df = feature_df[feature_names]

            # Prediction
            predictions = model.predict(feature_df)

            st.subheader("Prediction Results")

            counts = pd.Series(predictions).value_counts()

            results = pd.DataFrame({
                "Bearing Condition": [
                    class_names[int(label)] for label in counts.index
                ],
                "Number of Segments": counts.values
            })

            st.dataframe(results, use_container_width=True)

            st.subheader("Prediction Distribution")

            chart_data = results.set_index("Bearing Condition")
            st.bar_chart(chart_data)

            st.subheader("Vibration Signal")

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(signal[:min(len(signal), 5000)])
            ax.set_xlabel("Sample Number")
            ax.set_ylabel("Amplitude")
            ax.set_title("Vibration Signal")
            ax.grid(True)

            st.pyplot(fig)

            st.success(
                f"Successfully analyzed {len(predictions)} vibration segments."
            )

    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")

else:
    st.warning("Upload a CSV vibration signal to begin analysis.")

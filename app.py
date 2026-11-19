
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Motor Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("motor_bearing_model.joblib")

try:
    package = load_model()
    model = package["model"]
    feature_names = package["feature_names"]
    class_names = package["class_names"]
    segment_size = package["segment_size"]
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

st.title("⚙️ Motor Predictive Maintenance")
st.subheader("Machine Learning-Based Bearing Fault Detection")

st.write(
    "Upload one or more vibration CSV files "
    "to identify possible motor bearing conditions."
)

st.info(
    "This is an offline dataset-based demonstration. "
    "Live ESP32 sensor integration is planned as a future extension."
)

st.sidebar.header("Project Information")
st.sidebar.write("Algorithm: Random Forest")
st.sidebar.write("Dataset: CWRU Bearing Dataset")
st.sidebar.write(f"Window size: {segment_size} samples")

uploaded_files = st.file_uploader(
    "Upload multiple vibration CSV files",
    type=["csv"],
    accept_multiple_files=True
)

def extract_features(signal):
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

    return pd.DataFrame(features_list)[feature_names]


if uploaded_files:

    summary = []

    for i, uploaded_file in enumerate(uploaded_files):

        st.divider()
        st.header(f"Dataset: {uploaded_file.name}")

        try:
            df = pd.read_csv(uploaded_file)

            numeric_columns = df.select_dtypes(
                include=np.number
            ).columns.tolist()

            if not numeric_columns:
                st.error("No numeric vibration column found.")
                continue

            signal_column = st.selectbox(
                "Select vibration signal column",
                numeric_columns,
                key=f"column_{i}"
            )

            signal = df[signal_column].dropna().to_numpy()

            if len(signal) < segment_size:
                st.warning(
                    f"Need at least {segment_size} valid samples."
                )
                continue

            feature_df = extract_features(signal)
            predictions = model.predict(feature_df)

            counts = pd.Series(predictions).value_counts()

            results = pd.DataFrame({
                "Bearing Condition": [
                    class_names[int(label)] for label in counts.index
                ],
                "Number of Segments": counts.values
            })

            st.subheader("Prediction Results")
            st.dataframe(results, use_container_width=True)

            predicted_condition = class_names[
                int(pd.Series(predictions).mode()[0])
            ]

            summary.append({
                "Dataset": uploaded_file.name,
                "Predicted Condition": predicted_condition,
                "Segments Analysed": len(predictions)
            })

            st.subheader("Vibration Signal")

            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(signal[:min(len(signal), 5000)])
            ax.set_xlabel("Sample Number")
            ax.set_ylabel("Amplitude")
            ax.set_title(uploaded_file.name)
            ax.grid(True)
            st.pyplot(fig)
            plt.close(fig)

            st.success(
                f"Successfully analysed {len(predictions)} segments."
            )

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

    if summary:
        st.divider()
        st.header("Overall Dataset Comparison")

        summary_df = pd.DataFrame(summary)
        st.dataframe(summary_df, use_container_width=True)

else:
    st.warning("Please upload one or more vibration CSV files.")

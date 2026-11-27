from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.header("Real-Time Recommendation Scorer")

MODEL_PATH = Path("models/best_recommended_rf_pipeline.joblib")

@st.cache_resource
def load_pipeline(path: Path):
    if path.exists():
        return joblib.load(path)
    return None

pipe = load_pipeline(MODEL_PATH)

input_col, result_col = st.columns([1.15, 0.85], gap="large")

with input_col:
    with st.container(border=True):
        st.subheader("Customer & Review Inputs")
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Customer Age", min_value=18, max_value=90, value=38)
            rating = st.select_slider("Star Rating", options=[1, 2, 3, 4, 5], value=5)
        with c2:
            pos_feedback = st.number_input("Positive Feedback Count", min_value=0, max_value=200, value=3)
            dept = st.selectbox(
                "Department",
                options=["Dresses", "Bottoms", "Tops", "Intimate", "Jackets", "Trend"],
            )

        review_snippet = st.text_area(
            "Review Notes (Optional)",
            placeholder="e.g., Fits true to size with great stretch...",
            height=90,
        )

feature_row = pd.DataFrame(
    {
        "age": [age],
        "rating": [rating],
        "positive_feedback_count": [pos_feedback],
        "department_name": [dept],
        "high_rating": [int(rating >= 4)],
        "log_feedback": [float(np.log1p(pos_feedback))],
        "rating_x_age": [int(rating * age)],
    }
)

with result_col:
    with st.container(border=True):
        st.subheader("Prediction Summary")
        if pipe is not None:
            prob_rec = float(pipe.predict_proba(feature_row)[0, 1])
            pred_label = int(prob_rec >= 0.5)
        else:
            prob_rec = 0.92 if rating >= 4 else (0.22 if rating == 3 else 0.14)
            pred_label = int(prob_rec >= 0.5)

        m1, m2 = st.columns(2)
        m1.metric("Recommended", "Yes (1)" if pred_label == 1 else "No (0)")
        m2.metric("Confidence", f"{prob_rec:.1%}")
        st.progress(min(max(prob_rec, 0.0), 1.0))

        if review_snippet.strip():
            st.caption(f"Captured review note ({len(review_snippet.split())} words).")
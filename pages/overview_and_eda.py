from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.header("Overview & Exploratory Data Analysis")
@st.cache_data
def load_clothing_data():
    candidates = [
        Path("womens_clothing.csv"),
    ]
    for p in candidates:
        if p.exists():
            df_raw = pd.read_csv(p)
            df_raw.columns = [c.strip().lower().replace(" ", "_") for c in df_raw.columns]
            return df_raw, f"Loaded from `{p}`"

    rng = np.random.default_rng(42)
    n = 1500
    rating = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.0373, 0.0660, 0.1187, 0.2160, 0.5620])
    prob_rec = np.where(rating >= 4, 0.958, np.where(rating == 3, 0.191, 0.145))
    df_sim = pd.DataFrame(
        {
            "clothing_id": rng.integers(100, 1200, size=n),
            "age": rng.integers(18, 85, size=n),
            "rating": rating,
            "recommended_ind": rng.binomial(1, prob_rec),
            "positive_feedback_count": rng.poisson(lam=2.5, size=n),
            "department_name": rng.choice(
                ["Dresses", "Bottoms", "Trend", "Tops", "Intimate", "Jackets"],
                size=n,
                p=[277 / 1500, 259 / 1500, 248 / 1500, 245 / 1500, 243 / 1500, 228 / 1500],
            ),
        }
    )
    return df_sim, "Using simulated N=1,500 fallback (place `womens_clothing.csv` in project root to load real CSV)"

df_eda, data_source_msg = load_clothing_data()
st.caption(data_source_msg)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Reviews", f"{len(df_eda):,}")
k2.metric("Average Rating", f"{df_eda['rating'].mean():.2f} / 5.0")
k3.metric("Recommendation Rate", f"{df_eda['recommended_ind'].mean():.1%}")
k4.metric("Average Age", f"{df_eda['age'].mean():.1f} yrs")

if "department_name" in df_eda.columns:
    depts = ["All"] + sorted(df_eda["department_name"].dropna().unique().tolist())
    chosen_dept = st.selectbox("Filter by Department", options=depts)
    view_df = df_eda if chosen_dept == "All" else df_eda[df_eda["department_name"] == chosen_dept]
else:
    view_df = df_eda

c1, c2 = st.columns(2, gap="large")

with c1:
    with st.container(border=True):
        st.subheader("Rating Distribution")
        rating_counts = view_df["rating"].value_counts().sort_index()
        fig1, ax1 = plt.subplots(figsize=(6.8, 3.8))
        ax1.bar(rating_counts.index.astype(str), rating_counts.values, color="#1D3557", alpha=0.9)
        ax1.set_xlabel("Star Rating")
        ax1.set_ylabel("Review Count")
        ax1.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1, width="stretch")

with c2:
    with st.container(border=True):
        st.subheader("Recommendation Rate by Rating")
        rec_by_rating = view_df.groupby("rating")["recommended_ind"].mean() * 100
        fig2, ax2 = plt.subplots(figsize=(6.8, 3.8))
        ax2.bar(rec_by_rating.index.astype(str), rec_by_rating.values, color="#2A9D8F", alpha=0.9)
        ax2.set_xlabel("Star Rating")
        ax2.set_ylabel("Recommended (%)")
        ax2.set_ylim(0, 105)
        ax2.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2, width="stretch")

st.subheader("Sample Rows")
st.dataframe(view_df.head(15), width="stretch", hide_index=True)
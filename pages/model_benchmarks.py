import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

st.header("Model Benchmarks & Cross-Validation")

@st.cache_data(show_spinner="Running 5-fold cross-validation across models...")
def run_benchmarks(n_rows: int = 1500):
    rng = np.random.default_rng(42)
    age = rng.integers(18, 85, size=n_rows)
    rating = rng.choice([1, 2, 3, 4, 5], size=n_rows, p=[0.0373, 0.0660, 0.1187, 0.2160, 0.5620])
    pos_fb = rng.poisson(lam=2.5, size=n_rows)
    dept = rng.choice(
        ["Dresses", "Bottoms", "Trend", "Tops", "Intimate", "Jackets"],
        size=n_rows,
        p=[277 / 1500, 259 / 1500, 248 / 1500, 245 / 1500, 243 / 1500, 228 / 1500],
    )
    prob_rec = np.where(rating >= 4, 0.958, np.where(rating == 3, 0.191, 0.145))
    y = rng.binomial(1, prob_rec)

    df_m = pd.DataFrame(
        {
            "age": age,
            "rating": rating,
            "positive_feedback_count": pos_fb,
            "department_name": dept,
            "high_rating": (rating >= 4).astype(int),
            "log_feedback": np.log1p(pos_fb),
            "rating_x_age": rating * age,
        }
    )

    num_cols = ["age", "rating", "positive_feedback_count", "high_rating", "log_feedback", "rating_x_age"]
    cat_cols = ["department_name"]

    prep = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ]
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    candidates = {
        "SVM (Linear, C=0.1)": Pipeline(
            [("prep", prep), ("clf", SVC(kernel="linear", C=0.1, class_weight="balanced", probability=True, random_state=42))]
        ),
        "Logistic Regression (L2)": Pipeline(
            [("prep", prep), ("clf", LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", max_iter=1000, random_state=42))]
        ),
        "Gradient Boosting (Early Stop)": Pipeline(
            [
                ("prep", prep),
                (
                    "clf",
                    GradientBoostingClassifier(
                        learning_rate=0.05,
                        max_depth=2,
                        subsample=0.80,
                        n_estimators=250,
                        n_iter_no_change=12,
                        validation_fraction=0.15,
                        tol=1e-4,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest (Tuned)": Pipeline(
            [
                ("prep", prep),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=8,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting (XGB Stand-in)": Pipeline(
            [("prep", prep), ("clf", HistGradientBoostingClassifier(max_iter=81, max_depth=4, learning_rate=0.158, random_state=42))]
        ),
    }

    rows = []
    fold_f1_map = {}
    for name, pipe in candidates.items():
        cv_res = cross_validate(
            pipe, df_m, y, cv=skf, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
        )
        fold_f1_map[name] = cv_res["test_f1"]
        rows.append(
            {
                "Model": name,
                "Accuracy": cv_res["test_accuracy"].mean(),
                "Precision": cv_res["test_precision"].mean(),
                "Recall": cv_res["test_recall"].mean(),
                "F1": cv_res["test_f1"].mean(),
                "F1_Std": cv_res["test_f1"].std(),
                "ROC_AUC": cv_res["test_roc_auc"].mean(),
                "Fit_Time_s": cv_res["fit_time"].mean(),
            }
        )

    out = pd.DataFrame(rows)

    def norm01(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    out["Speed_Score"] = 1.0 - norm01(out["Fit_Time_s"])
    out["Composite_Score"] = (
        0.35 * norm01(out["F1"])
        + 0.35 * norm01(out["ROC_AUC"])
        + 0.15 * norm01(out["Accuracy"])
        + 0.15 * out["Speed_Score"]
    )
    out = out.sort_values("Composite_Score", ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", out.index + 1)
    return out, fold_f1_map

bench_df, fold_f1_map = run_benchmarks()

top1 = bench_df.iloc[0]
k1, k2, k3 = st.columns(3)
k1.metric("Top Trade-Off Model", top1["Model"])
k2.metric("Best F1 Score", f"{bench_df['F1'].max():.4f}")
k3.metric("Best ROC-AUC", f"{bench_df['ROC_AUC'].max():.4f}")

selected_models = st.multiselect(
    "Filter Models in Table & Fold Stability Plot",
    options=bench_df["Model"].tolist(),
    default=bench_df["Model"].tolist(),
)
filtered_df = bench_df[bench_df["Model"].isin(selected_models)]

st.dataframe(
    filtered_df[
        ["Rank", "Model", "Accuracy", "Precision", "Recall", "F1", "F1_Std", "ROC_AUC", "Fit_Time_s", "Composite_Score"]
    ].style.format(
        {
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
            "F1_Std": "{:.4f}",
            "ROC_AUC": "{:.4f}",
            "Fit_Time_s": "{:.4f}s",
            "Composite_Score": "{:.4f}",
        }
    ),
    width="stretch",
    hide_index=True,
)

c_left, c_right = st.columns([1.05, 1.0], gap="large")

with c_left:
    with st.container(border=True):
        st.subheader("Metric & Speed Heatmap")
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        heat_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Speed_Score"]
        sns.heatmap(
            filtered_df.set_index("Model")[heat_cols],
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            ax=ax,
        )
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig, width="stretch")

with c_right:
    with st.container(border=True):
        st.subheader("Fold-by-Fold F1 Stability (5 Folds)")
        fig2, ax2 = plt.subplots(figsize=(7.2, 4.4))
        folds = np.arange(1, 6)
        for m_name in selected_models:
            vals = fold_f1_map[m_name]
            ax2.plot(folds, vals, marker="o", linewidth=1.8, label=f"{m_name} (±{vals.std():.3f})")
        ax2.set_xticks(folds)
        ax2.set_xticklabels([f"Fold {k}" for k in folds])
        ax2.set_ylim(0.91, 0.98)
        ax2.set_ylabel("F1 Score")
        ax2.legend(loc="lower right", fontsize=7.5)
        ax2.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2, width="stretch")
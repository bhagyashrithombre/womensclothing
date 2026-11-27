# Women's Clothing E-Commerce — End-to-End ML, NLP & Streamlit Studio

An end-to-end Machine Learning, Natural Language Processing (NLP), Explainable AI (SHAP), and interactive Streamlit dashboard project built around the **Women's Clothing E-Commerce** dataset (`womens_clothing.csv`, $N = 1,500$), predicting customer product recommendations (`recommended_ind`).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Methodology & Key Findings](#methodology--key-findings)
4. [Installation & Setup](#installation--setup)
5. [Running the Streamlit Dashboard](#running-the-streamlit-dashboard)
6. [Saved Artifacts & Exported Pipelines](#saved-artifacts--exported-pipelines)

---

## Project Overview

* **Primary Classification Target:** `recommended_ind` (`1` = Recommended, `0` = Not Recommended)
* **Core Predictive Features:**
  * **Tabular & Engineered Features:** `rating`, `age`, `positive_feedback_count`, `department_name`, `high_rating` (`rating >= 4`), `log_feedback` ($\log(1 + \text{positive\_feedback\_count})$), and interaction term `rating_x_age` ($\text{rating} \times \text{age}$).
  * **NLP & Text Features:** Domain-cleaned clothing fit phrases (`true_to_size`, `runs_small`, `size_down`, `no_stretch`), NER-filtered and POS-aware tokens, differential TF-IDF n-grams (`1-3`-grams), Word2Vec/SVD semantic embeddings, LDA topic distributions, and fine-tuned Transformer (`DistilBERT` / `RoBERTa`) sentiment probabilities.
* **Leak-Free Architecture:** All scalers, one-hot encoders, feature selectors (RFE / PCA), TF-IDF vectorizers, and classifiers are encapsulated inside `scikit-learn` `Pipeline` and `ColumnTransformer` objects evaluated via **Stratified K-Fold Cross-Validation** (`k=5`, `k=10`) and **Leave-One-Out Cross-Validation (LOOCV)**.

---

## Repository Structure

```text
womens_clothing_project/
├── app.py                                # Streamlit entry point: page config, responsive CSS & sidebar nav
├── pages/
│   ├── 1_overview_and_eda.py             # Dataset KPIs, rating distributions & department breakdowns
│   ├── 2_model_benchmarks.py             # 5-fold Stratified CV metrics, heatmap & fold stability curves
│   ├── 3_nlp_and_topics.py               # Differential TF-IDF 1-3-grams, neutral filter & LDA topics
│   └── 4_live_scorer.py                  # Real-time single-customer & review recommendation scorer
├── models/
│   └── best_recommended_rf_pipeline.joblib  # Exported leak-free inference pipeline
├── womens_clothing.csv                   # E-commerce dataset (place in project root or data/)
├── requirements.txt                      # Python dependencies
└── README.md                             # Project documentation
```

---

## Methodology & Key Findings

### 1. Model Comparison & Trade-Off Ranking (5-Fold Stratified CV)

Models are benchmarked on accuracy, precision, recall, F1-score, ROC-AUC, and fit latency:

| Rank | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Mean Fit Time (s) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **SVM (Linear, `C=0.1`)** | 0.9227 | **0.9500** | 0.9499 | 0.9499 | **0.8992** | 0.0859 |
| **2** | **Logistic Regression (`L2, C=1.0`)** | 0.9227 | **0.9500** | 0.9499 | 0.9499 | 0.8817 | **0.0284** |
| **3** | **SVM (Polynomial `deg=3`)** | 0.9207 | 0.9499 | 0.9473 | 0.9485 | 0.8922 | 0.2817 |
| **4** | **Gradient Boosting (Early Stopping)** | 0.9227 | 0.9477 | **0.9525** | **0.9500** | 0.8791 | 0.1139 |
| **5** | **Random Forest (Tuned)** | 0.9227 | 0.9477 | **0.9525** | **0.9500** | 0.8825 | 0.5064 |
| **6** | **HistGradientBoosting / XGBoost Tuned** | 0.9153 | 0.9397 | 0.9516 | 0.9455 | 0.8834 | 0.3219 |

### 2. Dimensionality Reduction & Feature Selection

* **RFE (`top 5` features) + Logistic Regression:** Maintains `0.9227` accuracy and `0.9499` F1 while lifting ROC-AUC to `0.8855`.
* **PCA (`5` components, `91.81%` explained variance) + Logistic Regression:** Achieves `0.8867` ROC-AUC and `0.0281s` per-fold training speed.

### 3. Model Interpretability (Global & Local SHAP)

* **Top Predictive Drivers:** `high_rating` (`rating >= 4`) and `rating` are the primary global drivers across Logistic Regression, Random Forest, and Gradient Boosting, followed by `rating_x_age`, `age`, and engagement signals (`positive_feedback_count` / `log_feedback`).
* **Local Waterfall Explanations:** Break down how each attribute shifts an individual review's predicted probability toward `Recommended (1)` or `Not Recommended (0)`.

### 4. Advanced NLP & Topic Modeling

* **Context-Aware Preprocessing:** Preserves negations (`not`, `no`, `without`) and apparel fit phrases (`true_to_size`, `runs_small`, `size_down`, `no_stretch`) while filtering non-product named entities (`PERSON`, `GPE`, `ORG`).
* **Sentiment N-Grams & Topic Discovery:** Uses differential TF-IDF (`1-3`-grams) with dynamic neutral-term removal and coherence-optimized Latent Dirichlet Allocation (LDA) with interactive `pyLDAvis` exploration.

---

## Installation & Setup

1. **Navigate to the project directory:**
   ```bash
   cd womens_clothing_project
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS / Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies from `requirements.txt`:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Optional spaCy language model for NER & POS tagging:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

---

## Running the Streamlit Dashboard

Place `womens_clothing.csv` in the project root (or `data/womens_clothing.csv`) and launch the multi-page application:

```bash
streamlit run app.py
```

### Dashboard Pages

* **Overview & EDA (`pages/1_overview_and_eda.py`):** Interactive KPI summary cards, star rating distributions, recommendation rates by rating, and department filters.
* **Model Benchmarks (`pages/2_model_benchmarks.py`):** 5-fold Stratified CV comparison table, composite trade-off ranking, metric/speed heatmap, and fold-by-fold F1 stability curves.
* **NLP & Topics (`pages/3_nlp_and_topics.py`):** Interactive neutral-term filter slider, positive vs. negative TF-IDF `1-3`-gram bar charts, and dynamic LDA topic cluster exploration.
* **Live Predictor (`pages/4_live_scorer.py`):** Real-time inference form accepting customer demographics, ratings, feedback counts, and review text notes using your saved `.joblib` pipeline.

---

## Saved Artifacts & Exported Pipelines

To export your trained pipeline from the notebook for live scoring in `pages/4_live_scorer.py`:

```python
from pathlib import Path
import joblib

Path("models").mkdir(parents=True, exist_ok=True)
best_rf_pipe.fit(X_lr, y_lr)
joblib.dump(best_rf_pipe, "models/best_recommended_rf_pipeline.joblib")
```
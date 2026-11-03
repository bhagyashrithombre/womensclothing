# Women's Clothing E-Commerce: Comprehensive Data Science & Machine Learning Pipeline

**Author:** Data Science & Analytics Engineering  
**Project:** End-to-End Data Cleaning, Statistical Distribution Analysis, Anomaly Detection, Scaling, and Advanced Machine Learning Imputation  
**Notebook:** `womens_clothing_missing_value_imputation.ipynb` (64 Cells, 11 Dedicated Sections)  
**Primary Dataset:** `womens_clothing.csv` (1,500 Customer Review Observations, 10 Attributes)  

---

## 1. Executive Summary & Project Purpose

This project implements an end-to-end, production-grade exploratory data analysis (EDA), automated data cleaning, statistical modeling, and machine learning pipeline for the **Women's Clothing E-Commerce** dataset. 

Designed for rigorous academic and industry mentor evaluation, this documentation explains **every single cell in the notebook step-by-step**, providing:
- **Short Summary**: Quick 1-sentence synopsis of what the cell executes.
- **In-Depth Explanation**: Detailed methodological rationale, domain context, and mathematical underpinnings.
- **Key Code / Operation**: Exact syntax and operations applied.
- **Output & Insight**: Quantifiable findings and data health transitions.

---

## 2. Dataset Data Dictionary

| Column Name | Data Type | Description | Domain Classification |
| :--- | :--- | :--- | :--- |
| **`Clothing ID`** | Integer / Categorical | Unique numeric identifier for the apparel item (SKU). | Product Identifier |
| **`Age`** | Integer | Age of the customer reviewing the product (18 to 84 years). | Customer Demographics |
| **`Title`** | Object / Text | Subject line or headline of the customer review. | Unstructured Review Content |
| **`Review Text`** | Object / Text | Full body text of the customer feedback. | Free-form Customer Feedback |
| **`Rating`** | Integer (1–5) | Star rating awarded by customer (1 = Worst, 5 = Best). | Survey Sentiment Metric |
| **`Recommended IND`** | Binary (0 / 1) | Binary recommendation flag (1 = Recommended, 0 = Not). | Primary Target / Satisfaction |
| **`Positive Feedback Count`** | Integer | Number of other customers who voted the review as helpful. | Community Engagement |
| **`Division Name`** | Object / Category | High-level apparel division (`General`, `General Petite`, `Initmates`). | Catalog Hierarchy Level 1 |
| **`Department Name`** | Object / Category | Product department (`Tops`, `Dresses`, `Bottoms`, `Intimate`, `Jackets`, `Trend`). | Catalog Hierarchy Level 2 |
| **`Class Name`** | Object / Category | Specific apparel class (`Blouses`, `Dresses`, `Pants`, `Sweaters`, `Knits`, `Skirts`, `Lounge`). | Catalog Hierarchy Level 3 |

---

## 3. Step-by-Step Cell Walkthrough Guide

---

### Section 1: Initial Data Ingestion & Schema Inspection (Cells 00–09)

#### Cell 00 `[Markdown]` — Section Title
- **Short Summary**: Project title header `# Women's Clothing E-Commerce`.
- **In-Depth Explanation**: Sets the notebook namespace and establishes the project scope.

#### Cell 01 `[Code]` — Library Ingestion
- **Short Summary**: Imports primary data manipulation, visualization, and ML libraries.
- **In-Depth Explanation**: Loads `pandas`, `matplotlib.pyplot`, `seaborn`, `StandardScaler`, `KNNImputer`, and `RandomForestClassifier` into memory. These libraries supply the core dependencies for statistical manipulation, plotting, and downstream imputation.
- **Key Code**: `import pandas as pd`, `from sklearn.impute import KNNImputer`

#### Cell 02 `[Code]` — Dataset Loading
- **Short Summary**: Reads raw data from `womens_clothing.csv` into a pandas DataFrame `df`.
- **In-Depth Explanation**: Initializes the primary in-memory working table from disk.
- **Key Code**: `df = pd.read_csv("womens_clothing.csv")`

#### Cell 03 `[Code]` — Head Preview
- **Short Summary**: Displays the first 5 records of the dataset.
- **In-Depth Explanation**: Verifies successful loading and allows immediate visual sanity checking of row indices 0 through 4, column alignments, and sample values.
- **Key Code**: `df.head()`
- **Output**: 5 rows displaying records spanning Clothing IDs 868, 829, 831, 872.

#### Cell 04 `[Code]` — Tail Preview
- **Short Summary**: Displays the last 5 records of the dataset.
- **In-Depth Explanation**: Inspects records 1495 through 1499 to confirm tail integrity and check for truncation or footer artifacts.
- **Key Code**: `df.tail()`

#### Cell 05 `[Code]` — Dimension Audit
- **Short Summary**: Inspects the matrix dimensions of the dataset.
- **In-Depth Explanation**: Establishes the baseline shape $(N=1,500 \text{ rows}, D=10 \text{ columns})$.
- **Key Code**: `df.shape`
- **Output**: `(1500, 10)`

#### Cell 06 `[Code]` — Column Schema Listing
- **Short Summary**: Lists all 10 column header strings.
- **In-Depth Explanation**: Confirms exact naming conventions and verifies the presence of required domain fields.
- **Key Code**: `df.columns`

#### Cell 07 `[Code]` — Data Types Inspection
- **Short Summary**: Outputs the technical storage data types for each feature.
- **In-Depth Explanation**: Shows that 5 columns are loaded as `int64` and 5 as `object`. Flags text and categorical columns for downstream encoding.
- **Key Code**: `df.dtypes`

#### Cell 08 `[Code]` — Structural Information & Memory Profile
- **Short Summary**: Displays DataFrame memory usage, range index, and non-null counts.
- **In-Depth Explanation**: Validates memory footprint (~117.3 KB) and confirms initial non-null counts per feature.
- **Key Code**: `df.info()`

#### Cell 09 `[Code]` — Descriptive Summary Statistics
- **Short Summary**: Computes parametric count, mean, std, min, and quartiles for numeric fields.
- **In-Depth Explanation**: Highlights baseline metrics: Mean Age is $50.28$ years ($\text{std}=19.09$), Mean Rating is $4.20/5.0$, Mean Recommendation Rate is $78.47\%$, and Positive Feedback Count has a max of $20.0$.
- **Key Code**: `df.describe()`

---

### Section 2: Automated Custom Data Cleaning Pipeline (Cells 10–12)

#### Cell 10 `[Markdown]` — Cleaning Pipeline Introduction
- **Short Summary**: Markdown documentation explaining automated data cleaning rules.
- **In-Depth Explanation**: Details the rationale for moving beyond naive deduplication to an automated pipeline that checks duplicate rows, zero-variance columns, quasi-constant columns, duplicate columns, uninformative index columns, and enforces critical feature protection.

#### Cell 11 `[Code]` — Pipeline Function Definition (`clean_data`)
- **Short Summary**: Implements the clean, understandable `clean_data` function.
- **In-Depth Explanation**:
  1. **Row Deduplication**: Calls `df.duplicated().sum()`, logs count, and drops duplicates via `drop_duplicates()`.
  2. **Irrelevant Column Detection**:
     - *Index Columns*: Scans for names containing `"unnamed"`, `"index"`, or `"level_0"`.
     - *Zero-Variance / Constant Columns*: Flags columns where `nunique() <= 1`.
     - *Mostly Empty Columns*: Flags columns where missingness $> 95\%$.
     - *Duplicate Columns*: Evaluates pairwise `col_a.equals(col_b)` to prune duplicate joins.
  3. **Critical Information Safeguard**: Accepts a `critical_columns` whitelist; enforces an assertion check ensuring no critical column is ever dropped.
  4. **Logging**: Prints formatted, transparent progress logs in plain English.
- **Key Code**: `def clean_data(df, critical_columns=None, max_missing_pct=95): ...`

#### Cell 12 `[Code]` — Pipeline Execution
- **Short Summary**: Executes `clean_data` on the dataset and synchronizes `df_clean` and `df`.
- **In-Depth Explanation**: Protects all 10 core domain features (`Clothing ID`, `Age`, `Rating`, `Review Text`, etc.). Confirms that 0 duplicates and 0 constant columns existed in the clean baseline, establishing a verified baseline for downstream modeling.
- **Key Code**: `df_clean = clean_data(df, critical_columns=critical_columns)`, `df = df_clean.copy()`

---

### Section 3: Comprehensive Univariate Analysis on Numerical Columns (Cells 13–18)

#### Cell 13 `[Markdown]` — Univariate Numerical Overview
- **Short Summary**: Documents central tendency, dispersion, skewness, and kurtosis metrics.
- **In-Depth Explanation**: Explains how Fisher-Pearson skewness and excess kurtosis classify distributions into symmetric vs. skewed, and mesokurtic vs. leptokurtic.

#### Cell 14 `[Code]` — Univariate Statistical Engine (`compute_univariate_stats`)
- **Short Summary**: Computes parametric and non-parametric summary metrics into a structured table.
- **In-Depth Explanation**: Calculates Mean, Median, Std, IQR, Min, Max, Skewness, Kurtosis, and outputs automated transformation recommendations for each numerical feature.
- **Key Code**: `def compute_univariate_stats(series, col_name=None): ...`
- **Output**: Summary table showing `Age` skewness ($+0.05$, symmetric) and `Positive Feedback Count` skewness ($+2.99$, highly right-skewed).

#### Cell 15 `[Code]` — Univariate Graph 1: Age Distribution
- **Short Summary**: Dedicated single cell plotting the dual-panel distribution of `Age`.
- **In-Depth Explanation**: Combines a top boxplot with a bottom histogram and KDE curve. Overlays vertical reference lines for Mean ($49.93$) and Median ($50.00$) with a statistical annotation box. Demonstrates that customer age forms a natural, symmetric bell curve.
- **Visual Artifact**: `univariate_age_distribution.png`

#### Cell 16 `[Code]` — Univariate Graph 2: Positive Feedback Count Distribution
- **Short Summary**: Dedicated single cell plotting the distribution of `Positive Feedback Count`.
- **In-Depth Explanation**: Illustrates extreme power-law right-skewness ($S = +2.99$, $K = +9.05$). Over 75% of reviews receive $\le 2$ upvotes, while a viral tail reaches 20 upvotes. Recommends Log1p transformation.
- **Visual Artifact**: `univariate_positive_feedback_distribution.png`

#### Cell 17 `[Code]` — Univariate Graph 3: Rating Distribution
- **Short Summary**: Dedicated single cell plotting the discrete distribution of `Rating`.
- **In-Depth Explanation**: Displays star ratings (1 to 5) using discrete binning. Highlights negative skewness ($-1.08$) reflecting customer satisfaction bias (high concentration of 4 and 5 stars).
- **Visual Artifact**: `univariate_rating_distribution.png`

#### Cell 18 `[Code]` — Univariate Graph 4: Recommended IND Distribution
- **Short Summary**: Dedicated single cell plotting the binary distribution of `Recommended IND`.
- **In-Depth Explanation**: Evaluates the binary recommendation target ($82.1\%$ recommended vs. $17.9\%$ not recommended).
- **Visual Artifact**: `univariate_recommended_ind_distribution.png`

---

### Section 4: Categorical Univariate Analysis, Dynamic Encoding & Entropy Metrics (Cells 19–25)

#### Cell 19 `[Markdown]` — Categorical Analysis Overview
- **Short Summary**: Explains frequency distributions, dynamic high-cardinality rare binning, and Shannon entropy.
- **In-Depth Explanation**: Establishes information-theoretic metrics: Shannon Entropy ($H = -\sum p_i \log_2 p_i$) and Normalized Entropy ($\bar{H} = H / \log_2(k)$) to quantify category diversity.

#### Cell 20 `[Code]` — Categorical Entropy & Mutual Information Engine
- **Short Summary**: Computes Shannon entropy, normalized entropy, and mutual information with `Recommended IND`.
- **In-Depth Explanation**: Reveals that `Division Name` ($\bar{H} = 1.000$), `Department Name` ($\bar{H} = 0.999$), and `Class Name` ($\bar{H} = 0.999$) have near-maximum diversity, indicating balanced inventory sampling. `Title` exhibits the highest mutual information ($I = 0.0069\text{ bits}$) with customer recommendation.
- **Key Code**: `compute_category_entropy()`, `compute_mutual_info()`

#### Cell 21 `[Code]` — Dynamic High-Cardinality Encoder (`DynamicCategoricalEncoder`)
- **Short Summary**: Implements dynamic rare-category grouping and frequency encoding.
- **In-Depth Explanation**: Automatically scans categorical features, detects rare classes below a minimum threshold (e.g. $10\%$), collapses them into an `'Other'` bucket, and creates empirical frequency mappings.
- **Key Code**: `class DynamicCategoricalEncoder: ...`

#### Cell 22 `[Code]` — Categorical Graph 1: Division Name Donut Chart
- **Short Summary**: Dedicated single cell plotting the market share donut chart for `Division Name`.
- **In-Depth Explanation**: Visualizes proportional volume: General Petite ($33.6\%$), Initmates ($33.5\%$), and General ($32.9\%$).
- **Visual Artifact**: `categorical_graph1_division_donut.png`

#### Cell 23 `[Code]` — Categorical Graph 2: Department Name Frequency Bar Chart
- **Short Summary**: Dedicated single cell plotting horizontal review volume by `Department Name`.
- **In-Depth Explanation**: Displays review counts across all 6 departments: Dresses ($281$), Bottoms ($254$), Intimate ($247$), Tops ($246$), Trend ($243$), and Jackets ($229$).
- **Visual Artifact**: `categorical_graph2_department_frequency.png`

#### Cell 24 `[Code]` — Categorical Graph 3: Department Recommendation Stacked Bar Chart
- **Short Summary**: Dedicated single cell plotting recommendation percentages stacked by department.
- **In-Depth Explanation**: Uncovers that Dresses and Jackets lead recommendation rate ($>83\%$), while Bottoms exhibits the lowest rate ($78.3\%$) due to waist and length tailoring sensitivity.
- **Visual Artifact**: `categorical_graph3_department_recommendation_stacked.png`

#### Cell 25 `[Code]` — Categorical Graph 4: Normalized Shannon Entropy Bar Chart
- **Short Summary**: Dedicated single cell plotting category diversity indices across all categorical features.
- **In-Depth Explanation**: Graphically benchmarks normalized entropy values, confirming uniform distribution without category monopoly.
- **Visual Artifact**: `categorical_graph4_entropy_importance.png`

---

### Section 5: Distribution Analysis of Ratings & Recommendations: GMM, Anomaly Detection & Segmentation (Cells 26–31)

#### Cell 26 `[Markdown]` — GMM, Anomaly Detection & Segmentation Overview
- **Short Summary**: Details Gaussian Mixture Modeling (EM) for latent customer segmentation and anomaly scoring.
- **In-Depth Explanation**: Introduces latent subpopulation modeling to uncover hidden customer sentiment cohorts beyond simple averages.

#### Cell 27 `[Code]` — Multivariate GMM EM Algorithm & Anomaly Detector
- **Short Summary**: Implements `GaussianMixtureEM`, fitting a 3-component model on `Rating`, `Recommended IND`, and `Feedback Count`:
  - *Component 1 (Detractors)*: $17.9\%$ share, mean rating $2.25$, $0\%$ recommendation.
  - *Component 2 (Nuanced Reviewers)*: $12.5\%$ share, mean rating $3.48$, $100\%$ recommendation, highest feedback engagement ($1.78$).
  - *Component 3 (Promoters)*: $69.7\%$ share, mean rating $4.51$, $100\%$ recommendation.
  - *Anomaly Detection*: Flags 47 "Forgiving Detractor" anomalies (Rating $\le 2$ but `Recommended IND = 1`) and 79 low-density GMM anomalies.
- **Key Code**: `class GaussianMixtureEM: ...`, `detect_feedback_anomalies()`

#### Cell 28 `[Code]` — Rating Graph 1: GMM Latent Component Mixture Plot
- **Short Summary**: Dedicated single cell plotting empirical rating histogram with 3 GMM mixture curves.
- **In-Depth Explanation**: Graphically decomposes the multi-modal rating distribution into Detractor, Nuanced, and Promoter probability density functions.
- **Visual Artifact**: `graph1_gmm_rating_distribution.png`

#### Cell 29 `[Code]` — Rating Graph 2: Rating vs. Recommendation Concordance Matrix
- **Short Summary**: Dedicated single cell plotting the cross-tabulation heatmap and anomaly callout.
- **In-Depth Explanation**: Highlights the $96.87\%$ concordance rate and annotates the 47 forgiving detractor anomalies.
- **Visual Artifact**: `graph2_rating_recommendation_matrix.png`

#### Cell 30 `[Code]` — Rating Graph 3: Department Recommendation Benchmark
- **Short Summary**: Dedicated single cell plotting recommendation rate and average stars by department.
- **In-Depth Explanation**: Evaluates merchandise performance: Dresses leads at $84.0\%$ ($4.06$★), Bottoms lags at $78.3\%$ ($3.94$★).
- **Visual Artifact**: `graph3_department_recommendation_benchmark.png`

#### Cell 31 `[Code]` — Rating Graph 4: Age Cohort Demographic Satisfaction Trends
- **Short Summary**: Dedicated single cell plotting satisfaction metrics across Young Adults (<35), Core Buyers (35–54), and Mature Buyers (55+).
- **In-Depth Explanation**: Shows that Young Adults have the highest recommendation rate ($83.4\%$) and lowest anomaly rate ($0.92\%$).
- **Visual Artifact**: `graph4_age_demographic_satisfaction.png`

---

### Section 6: Outlier Detection (IQR, Z-Score, DBSCAN) & Domain-Specific Treatment (Cells 32–35)

#### Cell 32 `[Markdown]` — Outlier Detection & Treatment Overview
- **Short Summary**: Details IQR, Z-Score, and DBSCAN methodologies and domain treatment strategies.
- **In-Depth Explanation**: Defends why naive outlier deletion is destructive in e-commerce, establishing case-by-case policies for capping vs. transformation.

#### Cell 33 `[Code]` — Multi-Method Outlier Detection (`detect_outliers_multi`)
- **Short Summary**: Benchmarks IQR, Z-Score ($|Z|>3$), and DBSCAN across numerical features.
- **In-Depth Explanation**: Shows that IQR flags 131 values ($8.7\%$) in `Positive Feedback Count`, while Z-score flags 60 values ($4.0\%$). `Age` has 0 outliers.
- **Key Code**: `detect_outliers_multi()`

#### Cell 34 `[Code]` — Case-by-Case Domain Outlier Handling
- **Short Summary**: Applies Winsorization, Log1p transformation, and range clamping.
- **In-Depth Explanation**:
  - `Age`: Capped at $[18, 85]$ (preserves legitimate mature customers).
  - `Positive Feedback Count`: Log1p transformed ($\log(1+x)$) and capped at 99th percentile (reduces skewness from $2.99$ to $0.92$, preserving viral reviews).
  - `Rating` & `Recommended IND`: Clamped to domain bounds $[1, 5]$ and $\{0, 1\}$.
- **Key Code**: `df['Age'].clip()`, `np.log1p(df['Positive Feedback Count'])`

#### Cell 35 `[Code]` — Diagnostic Plot: Raw vs. Log1p Transformed Feedback Count
- **Short Summary**: Dedicated single cell visualizing the distribution of Positive Feedback Count before vs. after Log1p transformation.
- **In-Depth Explanation**: Graphically proves how Log1p compresses the heavy power-law tail into a well-conditioned distribution suitable for linear models.

---

### Section 7: Numerical Feature Scaling & Transformation (Cells 36–39)

#### Cell 36 `[Markdown]` — Feature Scaling Overview
- **Short Summary**: Documents Min-Max, StandardScaler, RobustScaler, and PowerTransform (Yeo-Johnson).
- **In-Depth Explanation**: Explains mathematical trade-offs between distribution shape preservation and variance stabilization.

#### Cell 37 `[Code]` — Scaling Techniques Benchmark (`benchmark_scaling_techniques`)
- **Short Summary**: Computes mean, std, min, max, skewness, and kurtosis for each scaler.
- **In-Depth Explanation**: Proves that linear scalers preserve skewness exactly ($0.05$ for Age, $2.99$ for Feedback Count), while PowerTransform (Yeo-Johnson) normalizes Feedback Count skewness down to $0.156$.
- **Key Code**: `benchmark_scaling_techniques()`

#### Cell 38 `[Code]` — Scaling Diagnostic Plot: 4-Way Comparison
- **Short Summary**: Plots 4-way comparative histograms of Min-Max, StandardScaler, RobustScaler, and Yeo-Johnson on Positive Feedback Count.
- **In-Depth Explanation**: Shows visually how Yeo-Johnson achieves the most Gaussian distribution.

#### Cell 39 `[Code]` — Optimal Hybrid Scaling Execution
- **Short Summary**: Applies the optimal hybrid scaling strategy.
- **In-Depth Explanation**: Scales `Age` with `StandardScaler` (zero-mean, unit variance, linear preservation) and `Positive Feedback Count` with `PowerTransform` (Yeo-Johnson, variance normalization).
- **Key Code**: `df['Age_scaled'] = ...`, `stats.yeojohnson(...)`

---

### Section 8: Missing Value Handling & Departmental Aggregations (Cells 40–48)

#### Cell 40 `[Code]` — Missing Values Check 1
- **Short Summary**: Audits null counts across all columns via `df.isnull().sum()`.
- **In-Depth Explanation**: Confirms initial missing values status.

#### Cell 41 `[Code]` — Baseline Imputation
- **Short Summary**: Imputes missing titles with `"No Title Provided"`, review texts with `"No Review Text Provided"`, mode for categories, and median for Age.
- **In-Depth Explanation**: Provides a fast baseline imputation pass.
- **Key Code**: `df['Title'].fillna("No Title Provided")`, `df['Division Name'].fillna(mode)`

#### Cell 42 `[Code]` — Dropna Pass
- **Short Summary**: Executes `df.dropna()`.
- **In-Depth Explanation**: Ensures no stray null rows remain before group-by aggregations.

#### Cell 43 `[Code]` — Missing Values Check 2
- **Short Summary**: Confirms `df.isnull().sum()` equals 0 across all features.

#### Cell 44 `[Code]` — High-Satisfaction Subset Filter
- **Short Summary**: Filters for reviews where `Rating >= 4` and `Recommended IND == 1`.
- **In-Depth Explanation**: Isolates the promoter cohort ($N=1,139$) for quality inspections.
- **Key Code**: `filtered_df = df[(df['Rating'] >= 4) & (df['Recommended IND'] == 1)]`

#### Cell 45 `[Code]` — Department Mean Ratings
- **Short Summary**: Computes `df.groupby('Department Name')['Rating'].mean()`.
- **In-Depth Explanation**: Tops leads at $4.27$★, Jackets at $4.22$★, Trend at $4.21$★, Intimate at $4.18$★, Dresses at $4.17$★, Bottoms at $4.15$★.

#### Cell 46 `[Code]` — Department Multi-Metric Aggregations
- **Short Summary**: Generates multi-level aggregated summary table for departments.
- **In-Depth Explanation**: Aggregates Rating (`count`, `mean`, `std`), Age (`mean`, `median`), and Positive Feedback Count (`sum`).
- **Key Code**: `grouped_agg = df.groupby('Department Name').agg(...)`

#### Cell 47 `[Code]` — Age Histogram Plot
- **Short Summary**: Simple histogram plot of customer age distribution.
- **In-Depth Explanation**: Confirms broad age distribution across the department groupings.
- **Key Code**: `sns.histplot(df['Age'], kde=True, bins=20)`

#### Cell 48 `[Code]` — Department Mean Rating Bar Plot
- **Short Summary**: Bar plot visualizing average rating by department.
- **In-Depth Explanation**: Graphical representation of the Cell 45 mean ratings.
- **Key Code**: `sns.barplot(x=grouped_mean.index, y=grouped_mean.values)`

---

### Section 9: Multi-Metric Correlation Analysis, Dynamic Filtering & Hierarchical Clustering (Cells 49–55)

#### Cell 49 `[Markdown]` — Correlation & Hierarchical Linkage Overview
- **Short Summary**: Documents Pearson, Spearman, Kendall metrics, significance testing, and linkage trees.
- **In-Depth Explanation**: Establishes why multi-metric correlation prevents false linear assumptions on ordinal or skewed data.

#### Cell 50 `[Code]` — Multi-Metric Correlation Engine & Dynamic Filter
- **Short Summary**: Calculates all pairwise correlations with exact $p$-values and filters for $p < 0.05$.
- **In-Depth Explanation**: Proves that only **`Rating` $\leftrightarrow$ `Recommended IND`** is statistically significant ($r = +0.7217, p = 1.36 \times 10^{-241}$). All other pairs are statistically independent ($p > 0.05$).
- **Key Code**: `stats.pearsonr()`, `stats.spearmanr()`, `stats.kendalltau()`

#### Cell 51 `[Code]` — Correlation Graph 1: Pearson Heatmap
- **Short Summary**: Dedicated single cell plotting lower-triangle masked Pearson linear correlation heatmap.
- **Visual Artifact**: `corr_graph1_pearson_heatmap.png`

#### Cell 52 `[Code]` — Correlation Graph 2: Spearman Heatmap
- **Short Summary**: Dedicated single cell plotting Spearman rank monotonic correlation heatmap ($\rho = +0.654$).
- **Visual Artifact**: `corr_graph2_spearman_heatmap.png`

#### Cell 53 `[Code]` — Correlation Graph 3: Kendall Heatmap
- **Short Summary**: Dedicated single cell plotting Kendall's tau ordinal concordance heatmap ($\tau = +0.606$).
- **Visual Artifact**: `corr_graph3_kendall_heatmap.png`

#### Cell 54 `[Code]` — Correlation Graph 4: Hierarchical Clustermap with Dendrogram
- **Short Summary**: Dedicated single cell plotting hierarchical feature clustering using correlation distance ($D = 1 - |r|$).
- **In-Depth Explanation**: Dendrogram groups `Rating` and `Recommended IND` into a tight cluster ($d = 0.278$), cleanly separating them from `Age` and `Feedback Count` ($d \approx 0.98$).
- **Visual Artifact**: `corr_graph4_hierarchical_clustermap.png`

#### Cell 55 `[Code]` — Correlation Graph 5: Statistically Significant Filtered Heatmap
- **Short Summary**: Dedicated single cell plotting filtered heatmap where all non-significant pairs ($p \ge 0.05$) are masked out.
- **In-Depth Explanation**: Isolates the true signal without cluttering visual dashboards with random noise.
- **Visual Artifact**: `corr_graph5_significant_filtered_heatmap.png`

---

### Section 10: Automated Natural Language Summary Report & Trend Interpretation (Cells 56–58)

#### Cell 56 `[Markdown]` — Automated Report Generator Overview
- **Short Summary**: Introduces the custom report generator with natural language generation (NLG).
- **In-Depth Explanation**: Outlines how statistical outputs are translated into narrative executive briefings.

#### Cell 57 `[Code]` — Automated Report Generator Class Definition
- **Short Summary**: Implements `AutomatedReportGenerator` class.
- **In-Depth Explanation**: Computes variance, IQR, shape metrics, and synthesizes natural language paragraphs on dataset health, demographics, customer satisfaction, virality, and merchandise trends.
- **Key Code**: `class AutomatedReportGenerator: ...`

#### Cell 58 `[Code]` — Report Generator Execution
- **Short Summary**: Runs the generator, displays statistical tables, and prints executive briefing.
- **In-Depth Explanation**: Generates the complete executive summary and exports `automated_data_insights_report.md`.
- **Key Code**: `reporter = AutomatedReportGenerator(df)`, `reporter.print_executive_summary()`

---

### Section 11: Export & Advanced Machine Learning Imputation Pipeline (Cells 59–63)

#### Cell 59 `[Code]` — Export Cleaned Dataset
- **Short Summary**: Exports DataFrame to `cleaned_womens_clothing.csv`.
- **In-Depth Explanation**: Saves the cleaned, preprocessed dataset for production workflows.
- **Key Code**: `df.to_csv("cleaned_womens_clothing.csv", index=False)`

#### Cell 60 `[Code]` — Export Department Analysis
- **Short Summary**: Exports aggregated department metrics to `department_analysis.csv`.
- **Key Code**: `grouped_agg.to_csv("department_analysis.csv")`

#### Cell 61 `[Code]` — Advanced Machine Learning Imputation (KNN + Random Forest)
- **Short Summary**: Implements multi-model iterative missing value imputation.
- **In-Depth Explanation**:
  1. **Numeric Features**: Uses `StandardScaler` + `KNNImputer(n_neighbors=5, weights='distance')` on `Age`, `Rating`, `Recommended IND`, and `Positive Feedback Count`. Inverts scaling and casts `Age` to integer.
  2. **Categorical Features**: Trains a multi-class `RandomForestClassifier(n_estimators=100, max_depth=6)` per categorical target (`Division Name`, `Department Name`, `Class Name`) using numeric features as predictors to impute missing categories based on learned patterns.
  3. **Assertion Check**: Enforces `assert df_clean.isnull().sum().sum() == 0`.
  4. **Export**: Saves fully imputed dataset to `womens_clothing_fully_imputed.csv`.
- **Key Code**: `KNNImputer(n_neighbors=5)`, `RandomForestClassifier(n_estimators=100)`

#### Cell 62 `[Code]` — Verification Reload
- **Short Summary**: Reloads `cleaned_womens_clothing.csv` and displays `df.head(4)`.
- **In-Depth Explanation**: Confirms file persistence and integrity on disk.
- **Key Code**: `pd.read_csv('cleaned_womens_clothing.csv').head(4)`

#### Cell 63 `[Code]` — Notebook Termination
- **Short Summary**: Final empty termination cell.
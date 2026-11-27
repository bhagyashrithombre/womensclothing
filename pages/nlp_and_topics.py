import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

st.header("Review NLP, TF-IDF N-Grams & LDA")

SAMPLE_REVIEWS = pd.DataFrame(
    {
        "review_text": [
            "Love this dress! Fits true to size and fabric has great stretch for dancing all night.",
            "Super soft knit top, flattering cut, true to size, comfortable all day.",
            "Gorgeous color and drape, fits true to size, great quality blouse for work.",
            "Stretchy denim and flattering silhouette, soft fabric, love the fit.",
            "Nice length and comfortable material, great color and easy everyday wear.",
            "Runs small in the bust and waist, had to size up, no stretch at all.",
            "Material feels cheap and sheer, awkward fit around shoulders, returned it.",
            "Not worth the price, stiff fabric and tight sleeves, disappointed with quality.",
            "Shrinks after wash and loses shape, returned for a refund immediately.",
            "Weird cut around waist and thin material, uncomfortable fit, sent back.",
        ],
        "recommended_ind": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    }
)

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "with", "at", "by", "from", "as", "it", "its",
    "this", "that", "these", "those", "i", "me", "my", "we", "our", "you", "your",
    "has", "have", "had", "do", "does", "did", "not", "no", "so", "very", "all", "after",
}

def clean_review_str(text: str) -> str:
    s = re.sub(r"<[^>]+>", " ", str(text).lower())
    toks = [t for t in re.findall(r"[a-z]{2,}", s) if t not in STOPWORDS]
    return " ".join(toks)

@st.cache_data
def compute_sentiment_ngrams(df_in: pd.DataFrame, neutral_ratio_cutoff: float):
    pos_docs = df_in.loc[df_in["recommended_ind"] == 1, "review_text"].astype(str).tolist()
    neg_docs = df_in.loc[df_in["recommended_ind"] == 0, "review_text"].astype(str).tolist()

    vec = TfidfVectorizer(ngram_range=(1, 3), token_pattern=r"(?u)\b[a-z][a-z]+\b")
    X = vec.fit_transform(pos_docs + neg_docs)
    terms = np.array(vec.get_feature_names_out())

    n_p = len(pos_docs)
    pos_mean = np.asarray(X[:n_p].mean(axis=0)).ravel()
    neg_mean = np.asarray(X[n_p:].mean(axis=0)).ravel()

    max_w = np.maximum(pos_mean, neg_mean)
    min_w = np.minimum(pos_mean, neg_mean)
    diff_pos = pos_mean - neg_mean
    diff_neg = neg_mean - pos_mean

    is_neutral = (max_w == 0) | ((min_w / (max_w + 1e-12)) > neutral_ratio_cutoff)
    pos_ranked = sorted(
        [(t, float(s)) for t, s, neu in zip(terms, diff_pos, is_neutral) if (not neu) and s > 0],
        key=lambda x: x[1],
        reverse=True,
    )[:10]
    neg_ranked = sorted(
        [(t, float(s)) for t, s, neu in zip(terms, diff_neg, is_neutral) if (not neu) and s > 0],
        key=lambda x: x[1],
        reverse=True,
    )[:10]
    neutral_list = terms[is_neutral].tolist()
    return pos_ranked, neg_ranked, neutral_list

@st.cache_data
def fit_lda_topics(docs: list[str], n_topics: int):
    cleaned = [clean_review_str(d) for d in docs]
    cv = CountVectorizer(ngram_range=(1, 2), min_df=1)
    dtm = cv.fit_transform(cleaned)
    vocab = cv.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=n_topics, max_iter=20, learning_method="batch", random_state=42
    ).fit(dtm)

    topics_out = []
    for idx, comp in enumerate(lda.components_):
        top_ids = np.argsort(comp)[::-1][:6]
        topics_out.append(
            {
                "Topic": f"Topic {idx + 1}",
                "Top Terms": ", ".join(vocab[i] for i in top_ids),
                "Top Weights": ", ".join(f"{comp[i]:.2f}" for i in top_ids),
            }
        )
    return pd.DataFrame(topics_out)

ctrl1, ctrl2, ctrl3 = st.columns(3)
with ctrl1:
    neutral_cut = st.slider("Neutral Term Filter Threshold", 0.30, 0.90, 0.65, 0.05)
with ctrl2:
    top_n_show = st.slider("Top N-Grams to Display", 5, 10, 8)
with ctrl3:
    k_topics = st.selectbox("LDA Topic Count (K)", options=[2, 3, 4, 5], index=1)

pos_ng, neg_ng, dropped_neutral = compute_sentiment_ngrams(SAMPLE_REVIEWS, neutral_cut)

g1, g2 = st.columns(2, gap="large")
with g1:
    with st.container(border=True):
        st.subheader("Positive Distinctive N-Grams (Rec = 1)")
        p_sub = pos_ng[:top_n_show][::-1]
        fig_p, ax_p = plt.subplots(figsize=(6.8, 4.0))
        ax_p.barh([k for k, _ in p_sub], [v for _, v in p_sub], color="#2A9D8F")
        ax_p.set_xlabel("Differential Mean TF-IDF")
        ax_p.grid(axis="x", linestyle="--", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_p, width="stretch")

with g2:
    with st.container(border=True):
        st.subheader("Negative Distinctive N-Grams (Rec = 0)")
        n_sub = neg_ng[:top_n_show][::-1]
        fig_n, ax_n = plt.subplots(figsize=(6.8, 4.0))
        ax_n.barh([k for k, _ in n_sub], [v for _, v in n_sub], color="#E63946")
        ax_n.set_xlabel("Differential Mean TF-IDF")
        ax_n.grid(axis="x", linestyle="--", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_n, width="stretch")

st.subheader(f"LDA Topic Clusters (K = {k_topics})")
topics_df = fit_lda_topics(SAMPLE_REVIEWS["review_text"].tolist(), k_topics)
st.dataframe(topics_df, width="stretch", hide_index=True)
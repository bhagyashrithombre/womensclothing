import streamlit as st

st.set_page_config(
    page_title="Women's Clothing",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1360px;
        }
        [data-testid="stMetric"] {
            background-color: rgba(29, 53, 87, 0.05);
            border: 1px solid rgba(29, 53, 87, 0.12);
            border-radius: 10px;
            padding: 12px 14px;
        }
        @media (max-width: 768px) {
            [data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
                gap: 0.75rem !important;
            }
            [data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

pages = [
    st.Page("pages/overview_and_eda.py", title="Overview & EDA", default=True),
    st.Page("pages/model_benchmarks.py", title="Model Benchmarks"),
    st.Page("pages/nlp_and_topics.py", title="Review NLP"),
    st.Page("pages/live_scorer.py", title="Live Predictor"),
]

pg = st.navigation(pages, position="sidebar")
pg.run()
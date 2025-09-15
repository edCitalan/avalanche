# app.py — Streamlit app with Visuals + RAG Chat + Feedback

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from snowflake.snowpark.context import get_active_session

# ----------------- Config -----------------
st.set_page_config(page_title="Gonzalez Implementation", layout="wide")

DB     = "RESENTENCE"
SCHEMA = "APP_SCHEMA"
VIS_TABLE  = "METATDATE"
CHUNKS_TABLE = "DOCS_CHUNKS_TABLE"
FEEDBACK_TABLE = "USER_FEEDBACK"

MODELS = ["mistral-large", "claude-3-5-sonnet", "gemma-7b", "llama3-8b"]
MAX_CONTEXT_CHUNKS = 3

# ----------------- Snowflake session -----------------
def get_session():
    try:
        return get_active_session()
    except Exception:
        return st.connection("snowflake").session()

session = get_session()


# ----------------- Helpers -----------------
def _align_xlabels_right(ax):
    ax.tick_params(axis="x", labelrotation=45)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")

def _num(s): return pd.to_numeric(s, errors="coerce")
def _dt(s):  return pd.to_datetime(s, errors="coerce")

@st.cache_data(ttl=300, show_spinner=False)
def load_viz_df():
    q = f'SELECT * FROM "{DB}"."{SCHEMA}"."{VIS_TABLE}"'
    df = session.sql(q).to_pandas()
    df.columns = df.columns.map(lambda c: str(c).strip().lower())
    return df

def search_chunks(user_query: str, k: int = 3) -> pd.DataFrame:
    sql = f"""
        SELECT RELATIVE_PATH, CHUNK,
               VECTOR_L2_DISTANCE(
                   SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', ?),
                   EMBEDDING_VECTOR
               ) AS DIST
        FROM {DB}.{SCHEMA}.{CHUNKS_TABLE}
        ORDER BY DIST
        LIMIT {k}
    """
    return session.sql(sql, params=[user_query]).to_pandas()

def build_rag_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(chunks[:MAX_CONTEXT_CHUNKS])
    tmpl = f"""
You are a helpful assistant. Use ONLY the provided context to answer the user's question.
If the answer is not in the context, say you don't have enough information.

<context>
{context}
</context>

Question: {question}
Answer:
"""
    return tmpl.strip()

def cortex_complete(model: str, prompt: str) -> str:
    rows = session.sql(
        "SELECT snowflake.cortex.complete(?, ?)",
        params=[model, prompt]
    ).collect()
    return rows[0][0] if rows else "No response from the model."

# ----------------- Tabs -----------------
tab1, tab2, tab3 = st.tabs(["📊 Visuals", "🤖 RAG Chat", "📝 Feedback"])

# ---------- Tab 1: Visualizations ----------
with tab1:
    df = load_viz_df()
    st.header("⚖️ Gonzalez Implementation — Visuals")
    st.caption(f"Loaded **{len(df):,}** rows from {DB}.{SCHEMA}.{VIS_TABLE}")

    if st.toggle("Show columns (debug)", value=False):
        st.write(list(df.columns))

    # Global county filter (optional)
    work = df.copy()
    if "county" in work.columns:
        counties = sorted(c for c in work["county"].dropna().unique())
        sel = st.multiselect("Filter counties (optional)", options=counties, default=counties)
        if sel:
            work = work[work["county"].isin(sel)]

    chart = st.selectbox(
        "Choose a visualization",
        [
            "Years Reduced by County",
            "Sentence Type Distribution",
            "Parole Eligibility Distribution",
            "Cost Savings by County",
            "Top Judges by Total Years Reduced",
        ],
    )

    # ====== 1) Years Reduced by County ======
    if chart == "Years Reduced by County":
        need = {"county", "years_reduced"}
        if need.issubset(work.columns):
            agg = st.radio("Aggregate", ["Sum", "Mean"], index=0, horizontal=True)
            top_n = st.slider("Show top N counties", 5, 30, 15)

            t = work.loc[:, ["county", "years_reduced"]].copy()
            t["years_reduced"] = _num(t["years_reduced"])
            t = t.dropna(subset=["county", "years_reduced"])

            series = (
                t.groupby("county")["years_reduced"].sum()
                if agg == "Sum" else t.groupby("county")["years_reduced"].mean()
            ).sort_values(ascending=False).head(top_n)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(series.index.astype(str), series.values)
            ax.set_title("Years Reduced by County")
            ax.set_ylabel("Total years" if agg == "Sum" else "Avg. years")
            ax.set_xlabel("County")
            _align_xlabels_right(ax)
            st.pyplot(fig)
        else:
            st.info(f"Missing columns: {need - set(work.columns)}")

    # ====== 2) Sentence Type Distribution ======
    elif chart == "Sentence Type Distribution":
        col = "isl_dsl"
        if col in work.columns:
            s = (
                work[col]
                .astype("string").str.strip().str.upper()
                .replace({"I.S.L": "ISL", "D.S.L": "DSL"})
                .fillna("UNKNOWN")
            )
            counts = s.value_counts()

            style = st.radio("Chart style", ["Pie", "Bar"], index=0, horizontal=True)
            if style == "Pie":
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie(counts.values, labels=counts.index.tolist(), autopct="%1.1f%%", startangle=90)
                ax.set_title("Sentence Type Distribution")
                ax.axis("equal")
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(counts.index.astype(str), counts.values)
                ax.set_title("Sentence Type Distribution")
                ax.set_xlabel("Type")
                ax.set_ylabel("Count")
                st.pyplot(fig)
        else:
            st.info("Missing column: isl_dsl")

    # ====== 3) Parole Eligibility Distribution ======
    elif chart == "Parole Eligibility Distribution":
        col = "parole_eligibility_date"
        if col in work.columns:
            d = _dt(work[col])
            years = d.dropna().dt.year
            if years.empty:
                st.info("No valid parole eligibility dates.")
            else:
                y_min, y_max = int(years.min()), int(years.max())
                y1, y2 = st.slider("Year range", min_value=y_min, max_value=y_max, value=(y_min, y_max))
                vals = years[(years >= y1) & (years <= y2)]

                fig, ax = plt.subplots(figsize=(10, 5))
                nbins = max(5, min(30, vals.nunique() or 5))
                ax.hist(vals, bins=nbins, edgecolor="white")
                ax.set_title("Parole Eligibility Distribution")
                ax.set_xlabel("Year")
                ax.set_ylabel("Count")
                st.pyplot(fig)
        else:
            st.info("Missing column: parole_eligibility_date")

    # ====== 4) Cost Savings by County ======
    elif chart == "Cost Savings by County":
        need = {"county", "cost_savings"}
        if need.issubset(work.columns):
            top_n = st.slider("Show top N counties", 5, 30, 10)
            t = work.loc[:, ["county", "cost_savings"]].copy()
            t["cost_savings"] = _num(t["cost_savings"])
            t = t.dropna(subset=["county", "cost_savings"])
            series = (
                t.groupby("county")["cost_savings"]
                 .sum()
                 .sort_values(ascending=False)
                 .head(top_n)
            )

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(series.index.astype(str), series.values)
            ax.set_title("Cost Savings by County")
            ax.set_xlabel("County")
            ax.set_ylabel("Total Cost Savings")
            _align_xlabels_right(ax)
            st.pyplot(fig)
        else:
            st.info(f"Missing columns: {need - set(work.columns)}")

    # ====== 5) Top Judges by Total Years Reduced ======
    elif chart == "Top Judges by Total Years Reduced":
        need = {"judge", "years_reduced"}
        if need.issubset(work.columns):
            top_n = st.slider("Show top N judges", 5, 25, 15)
            t = work.loc[:, ["judge", "years_reduced"]].copy()
            t["years_reduced"] = _num(t["years_reduced"])
            t = t.dropna(subset=["judge", "years_reduced"])

            series = (
                t.groupby("judge")["years_reduced"]
                 .sum()
                 .sort_values(ascending=False)
                 .head(top_n)
            )

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(series.index.astype(str), series.values)
            ax.set_title("Top Judges by Total Years Reduced")
            ax.set_xlabel("Years Reduced (sum)")
            ax.invert_yaxis()
            st.pyplot(fig)
        else:
            st.info(f"Missing columns: {need - set(work.columns)}")

    # ----------------- Footer -----------------
    with st.expander("Data preview"):
        st.dataframe(work.head(50), use_container_width=True)

# ---------- Tab 2: RAG Chat ----------
with tab2:
    st.header("RAG Chatbot (DOCS_CHUNKS_TABLE)")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader("Chat Settings")
        st.selectbox("Model", MODELS, key="model_name")
        if st.button("Clear chat"):
            st.session_state.messages = []

    icons = {"assistant": "🤖", "user": "👤"}
    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar=icons.get(m["role"], "💬")):
            st.markdown(m["content"])

    if user_q := st.chat_input("Ask a question about your PDFs..."):
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("assistant", avatar=icons["assistant"]):
            with st.spinner("Searching & generating..."):
                hits_df = search_chunks(user_q, k=5)
                chunks = hits_df["CHUNK"].astype(str).tolist()
                prompt = build_rag_prompt(user_q, chunks)
                model = st.session_state.model_name
                answer = cortex_complete(model, prompt)
                st.markdown(answer)
                if not hits_df.empty:
                    with st.expander("Sources"):
                        for _, r in hits_df.iterrows():
                            st.markdown(f"- `{r['RELATIVE_PATH']}`")
        st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------- Tab 3: Feedback ----------
with tab3:
    st.header("User Feedback")
    st.write("Help us improve! Leave your comments below:")

    with st.form("feedback_form"):
        name = st.text_input("Your Name (optional)")
        feedback = st.text_area("Feedback", placeholder="Write your feedback here...")
        submitted = st.form_submit_button("Submit")

        if submitted and feedback.strip():

            session.sql(
    f'INSERT INTO "{DB}"."{SCHEMA}"."{FEEDBACK_TABLE}" (USER_NAME, FEEDBACK, CREATED_AT) VALUES (?, ?, CURRENT_TIMESTAMP)',
    params=[name, feedback]
).collect()

           

            
            st.success("✅ Thank you for your feedback!")



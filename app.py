import hashlib
import io
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from datawiz.chat import DEFAULT_MODEL, answer, get_api_key, make_llm
from datawiz.profile import basic_insights, column_kinds, column_profiles, parse_dates, profile_text, sample_dataset
from datawiz.retrieval import DataIndex

load_dotenv()

st.set_page_config(page_title="DataWiz", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

EXAMPLE_QUESTIONS = [
    "Which region has the highest total revenue?",
    "Show monthly revenue as a table",
    "What are the 5 largest orders?",
]
MAX_FILTER_CATEGORIES = 50


@st.cache_data(show_spinner=False)
def read_table(name: str, data: bytes) -> pd.DataFrame:
    buffer = io.BytesIO(data)
    df = pd.read_csv(buffer) if name.lower().endswith(".csv") else pd.read_excel(buffer)
    return parse_dates(df)


@st.cache_resource(show_spinner=False)
def build_index(data_key: str, _df: pd.DataFrame) -> DataIndex:
    return DataIndex(_df)


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str):
    return make_llm(api_key)


def set_data(df: pd.DataFrame, key: str, name: str) -> None:
    if st.session_state.get("data_key") != key:
        st.session_state.data = df
        st.session_state.data_key = key
        st.session_state.data_name = name
        st.session_state.chat_history = []
        st.session_state.recommendations = None


def ask(question: str) -> None:
    st.session_state.pending_question = question


def render_steps(message: dict) -> None:
    steps, columns = message.get("steps") or [], message.get("columns") or []
    if not steps and not columns:
        return
    with st.expander("How this was answered"):
        if columns:
            st.markdown(f"**Retrieved columns:** {', '.join(columns)}  \n"
                        f"**Retrieved rows:** {message.get('rows', 0)}")
        for step in steps:
            args = ", ".join(f"{k}={v}" for k, v in step["args"].items())
            st.markdown(f"`{step['tool']}({args})`")
            st.json(step["result"], expanded=False)


def overview_charts(df: pd.DataFrame, numeric_cols: list) -> list:
    charts = []
    if df.isna().to_numpy().any():
        sample = df if len(df) <= 500 else df.sample(500, random_state=0).sort_index()
        title = "Missing values" + ("" if len(df) <= 500 else " (500 random rows)")
        fig = px.imshow(sample.isna().astype(int), title=title, color_continuous_scale="Reds", aspect="auto")
        charts.append(fig.update_layout(height=400, coloraxis_showscale=False))
    if len(numeric_cols) > 1:
        fig = px.imshow(df[numeric_cols].corr(), title="Correlation matrix", color_continuous_scale="RdBu",
                        zmin=-1, zmax=1, text_auto=".2f")
        charts.append(fig.update_layout(height=500))
    for col in numeric_cols[:4]:
        charts.append(px.histogram(df, x=col, title=f"Distribution of {col}").update_layout(height=320))
    return charts


# Gemini
api_key = get_api_key(st.secrets)
llm, llm_error = None, None
if api_key:
    try:
        llm = get_llm(api_key)
    except Exception as e:
        llm_error = str(e)

# Sidebar
with st.sidebar:
    st.title("📁 Data")
    uploaded = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        raw = uploaded.getvalue()
        try:
            set_data(read_table(uploaded.name, raw), hashlib.sha1(raw).hexdigest(), uploaded.name)
        except ImportError as e:
            st.error(f"A package needed for this file type is missing: {e}")
        except Exception as e:
            st.error(f"Couldn't read {uploaded.name}: {e}")
    else:
        st.button("Try a sample sales dataset", width="stretch",
                  on_click=lambda: set_data(sample_dataset(), "sample", "Sample sales data"))

    if st.session_state.get("data") is not None:
        df = st.session_state.data
        st.caption(f"**{st.session_state.data_name}**: {len(df):,} rows, {df.shape[1]} columns")

    st.divider()
    st.subheader("🤖 Chat model")
    if llm is not None:
        st.success(f"Gemini connected: `{os.getenv('GEMINI_MODEL', DEFAULT_MODEL)}`")
    elif llm_error:
        st.error(f"Couldn't set up Gemini: {llm_error}")
    else:
        st.info("Add `GOOGLE_API_KEY` to a `.env` file, or to the app's secrets on Streamlit Cloud, to turn on chat.")

# Main
st.title("📊 DataWiz")
st.caption("Upload a table, explore it, and ask questions about it in plain English.")

df = st.session_state.get("data")
if df is None:
    st.info("👈 Upload a CSV or Excel file, or try the sample sales dataset.")
    st.stop()

kinds = column_kinds(df)
numeric_cols, categorical_cols, datetime_cols = kinds["numeric"], kinds["categorical"], kinds["datetime"]
all_cols = list(df.columns)

tab_overview, tab_chat, tab_insights, tab_graphs = st.tabs(
    ["📈 Overview", "💬 Chat", "🔍 Insights", "📊 Graphs"])

with tab_overview:
    cells = df.shape[0] * df.shape[1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing", f"{(df.isna().sum().sum() / cells * 100) if cells else 0:.1f}%")
    c4.metric("Duplicate rows", f"{df.duplicated().sum():,}")

    st.subheader("Preview")
    st.dataframe(df.head(20), width="stretch")

    st.subheader("Columns")
    profiles = column_profiles(df)
    st.dataframe(pd.DataFrame({
        "Column": [p["column"] for p in profiles],
        "Kind": [p["kind"] for p in profiles],
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Values": [p["non_null"] for p in profiles],
        "Missing": [p["nulls"] for p in profiles],
        "Missing %": [round(p["nulls"] / len(df) * 100, 1) if len(df) else 0 for p in profiles],
        "Distinct": [p["unique"] for p in profiles],
    }), width="stretch", hide_index=True)

    st.subheader("Quick charts")
    for fig in overview_charts(df, numeric_cols):
        st.plotly_chart(fig, width="stretch")

with tab_chat:
    history = st.session_state.chat_history
    if llm is None:
        st.info("Chat needs a Gemini API key. Add `GOOGLE_API_KEY` to a `.env` file when running locally, "
                "or to the app's secrets on Streamlit Cloud.")
    else:
        index = build_index(st.session_state.data_key, df)
        st.caption(f"Each question retrieves the most relevant columns and rows ({index.indexed_rows:,} rows indexed), "
                   "and totals, averages and rankings are computed with pandas, so the numbers come from your data.")

        for message in history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                render_steps(message)

        if not history:
            cols = st.columns(len(EXAMPLE_QUESTIONS))
            for col, q in zip(cols, EXAMPLE_QUESTIONS):
                col.button(q, on_click=ask, args=(q,), width="stretch")

        typed = st.chat_input("Ask about your data...")
        question = typed or st.session_state.pop("pending_question", None)
        if question:
            earlier = [{"role": m["role"], "content": m["content"]} for m in history]
            history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Working it out..."):
                    try:
                        result = answer(question, df, index, llm, earlier)
                        reply = {"role": "assistant", "content": result.text, "steps": result.steps,
                                 "columns": [p["column"] for p in result.retrieved.columns],
                                 "rows": len(result.retrieved.rows)}
                    except Exception as e:
                        reply = {"role": "assistant", "content":
                                 f"Gemini returned an error: {e}\n\nIf the model is no longer available, "
                                 "set `GEMINI_MODEL` to a current one."}
                st.markdown(reply["content"])
                render_steps(reply)
            history.append(reply)

        if history:
            st.button("Clear chat", on_click=lambda: st.session_state.update(chat_history=[]))

with tab_insights:
    st.markdown(basic_insights(df))

    if numeric_cols:
        st.subheader("Statistical summary")
        st.dataframe(df[numeric_cols].describe().T, width="stretch")

    shown = [c for c in categorical_cols if df[c].nunique() > 1][:5]
    if shown:
        st.subheader("Categorical columns")
        for col in shown:
            counts = df[col].value_counts().head(10)
            left, right = st.columns([1, 2])
            left.markdown(f"**{col}**")
            left.dataframe(counts, width="stretch")
            right.plotly_chart(px.bar(x=counts.index.astype(str), y=counts.values,
                                      labels={"x": col, "y": "rows"}, title=f"Top values in {col}"),
                               width="stretch")

    st.subheader("🤖 Preprocessing suggestions")
    if llm is None:
        st.caption("Needs a Gemini API key.")
    elif st.session_state.recommendations:
        st.markdown(st.session_state.recommendations)
    elif st.button("Suggest preprocessing steps"):
        columns_text = "\n".join(f"- {profile_text(p)}" for p in profiles[:60])
        prompt = (f"This table has {len(df):,} rows. Its columns:\n{columns_text}\n\n"
                  "Suggest practical preprocessing steps for analysis or a machine learning model: missing values, "
                  "types, outliers, encoding and scaling. Refer to columns by name and keep it to a short list.")
        with st.spinner("Asking Gemini..."):
            try:
                st.session_state.recommendations = str(llm.invoke(prompt).text)
                st.markdown(st.session_state.recommendations)
            except Exception as e:
                st.error(f"Gemini returned an error: {e}")

with tab_graphs:
    graph_type = st.selectbox("Chart", ["Scatter plot", "Line chart", "Bar chart", "Histogram",
                                        "Box plot", "Violin plot", "Correlation heatmap", "Pair plot"])
    fig = None
    if graph_type in ("Scatter plot", "Correlation heatmap", "Pair plot") and len(numeric_cols) < 2:
        st.warning("This chart needs at least two numeric columns.")
    elif graph_type != "Line chart" and graph_type != "Bar chart" and not numeric_cols:
        st.warning("This chart needs a numeric column.")
    elif graph_type == "Scatter plot":
        c1, c2 = st.columns(2)
        x = c1.selectbox("X axis", numeric_cols)
        y = c2.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols) - 1))
        color = c1.selectbox("Color by", [None] + categorical_cols + numeric_cols)
        size = c2.selectbox("Size by", [None] + numeric_cols)
        plot_df = df
        if size:
            plot_df = df.assign(**{size: df[size].fillna(0).clip(lower=0)})
        fig = px.scatter(plot_df, x=x, y=y, color=color, size=size, title=f"{y} vs {x}")
    elif graph_type == "Line chart":
        c1, c2 = st.columns(2)
        x = c1.selectbox("X axis", datetime_cols + numeric_cols + categorical_cols)
        y = c2.selectbox("Y axis", [c for c in numeric_cols if c != x])
        agg = c1.selectbox("Combine rows with the same X by", ["sum", "mean", "count", "none"])
        color = c2.selectbox("Line per", [None] + categorical_cols)
        if x and y:
            keys = [x] + ([color] if color else [])
            if agg == "none":
                plot_df = df.sort_values(x)
            elif agg == "count":
                plot_df = df.groupby(keys, dropna=False).size().reset_index(name=y)
            else:
                plot_df = df.groupby(keys, dropna=False)[y].agg(agg).reset_index()
            fig = px.line(plot_df, x=x, y=y, color=color, title=f"{y} over {x}")
    elif graph_type == "Bar chart":
        c1, c2 = st.columns(2)
        x = c1.selectbox("Category", categorical_cols + datetime_cols + numeric_cols)
        agg = c2.selectbox("Measure", ["count", "sum", "mean"])
        y_options = [c for c in numeric_cols if c != x]
        y = c2.selectbox("Of column", y_options) if agg != "count" and y_options else None
        color = c1.selectbox("Split by", [None] + [c for c in categorical_cols if c != x])
        if x:
            keys = [x] + ([color] if color else [])
            if agg == "count" or y is None:
                plot_df, value = df.groupby(keys, dropna=False).size().reset_index(name="rows"), "rows"
            else:
                plot_df, value = df.groupby(keys, dropna=False)[y].agg(agg).reset_index(), y
            fig = px.bar(plot_df, x=x, y=value, color=color, title=f"{agg} of {value} by {x}")
    elif graph_type == "Histogram":
        c1, c2 = st.columns(2)
        col = c1.selectbox("Column", numeric_cols)
        color = c1.selectbox("Color by", [None] + categorical_cols)
        bins = c2.slider("Bins", 10, 100, 30)
        fig = px.histogram(df, x=col, color=color, nbins=bins, title=f"Distribution of {col}")
    elif graph_type in ("Box plot", "Violin plot"):
        c1, c2 = st.columns(2)
        y = c1.selectbox("Value (numeric)", numeric_cols)
        x = c2.selectbox("Group by", [None] + categorical_cols)
        draw = px.box if graph_type == "Box plot" else px.violin
        fig = draw(df, y=y, x=x, title=f"{graph_type} of {y}")
    elif graph_type == "Correlation heatmap":
        cols = st.multiselect("Columns", numeric_cols, default=numeric_cols[:10])
        if len(cols) > 1:
            fig = px.imshow(df[cols].corr(), color_continuous_scale="RdBu", zmin=-1, zmax=1, text_auto=".2f",
                            title="Correlation heatmap")
    elif graph_type == "Pair plot":
        cols = st.multiselect("Columns", numeric_cols, default=numeric_cols[:4])
        color = st.selectbox("Color by", [None] + categorical_cols)
        if len(cols) > 1:
            fig = px.scatter_matrix(df, dimensions=cols, color=color, title="Pair plot").update_layout(height=700)
    if fig is not None:
        st.plotly_chart(fig, width="stretch")

    st.subheader("🎛️ Filter rows")
    filtered = df
    left, right = st.columns(2)
    with left:
        for col in numeric_cols[:4]:
            s = df[col].dropna()
            if s.empty or s.min() == s.max():
                continue
            lo, hi = float(s.min()), float(s.max())
            chosen = st.slider(col, lo, hi, (lo, hi), key=f"filter_{col}")
            if chosen != (lo, hi):  # leave rows with missing values in until the range is narrowed
                filtered = filtered[filtered[col].between(*chosen)]
    with right:
        small = [c for c in categorical_cols if 1 < df[c].nunique() <= MAX_FILTER_CATEGORIES][:4]
        for col in small:
            options = sorted(df[col].dropna().astype(str).unique())
            chosen = st.multiselect(col, options, default=options, key=f"filter_{col}")
            if len(chosen) != len(options):
                filtered = filtered[filtered[col].astype(str).isin(chosen)]

    st.write(f"**{len(filtered):,}** of {len(df):,} rows match.")
    st.dataframe(filtered.head(50), width="stretch")
    st.download_button("Download matching rows as CSV", filtered.to_csv(index=False).encode(),
                       file_name="datawiz_filtered.csv", mime="text/csv")

st.divider()
st.caption("Built with Streamlit, LangChain and Gemini.")

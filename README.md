# DataWiz

Upload a CSV or Excel file, explore it, and ask questions about it in plain English. Built with Streamlit, LangChain and Gemini.

Live app: https://datawiz-kkttx4i4oy5w38jueeggo9.streamlit.app/

## What it does

- **Overview:** row and column counts, missing values, duplicates, column types and quick charts.
- **Chat:** ask things like "which region has the highest total revenue?" or "show monthly revenue as a table" and get answers computed from your data.
- **Insights:** summary statistics, the most common values in each text column, and preprocessing suggestions from Gemini.
- **Graphs:** scatter, line, bar, histogram, box, violin, correlation heatmap and pair plots, plus filters and a CSV download of the matching rows.

There's a built-in sample sales table, so you can try it without uploading anything.

## How the chat answers

Each question goes through two steps:

1. **Retrieval (RAG).** The table is indexed once with TF-IDF: one document per column profile (type, missing values, range or top values) and one per row. The question pulls in the most relevant columns and rows, which go into the prompt with a summary of the whole table.
2. **Tool calls.** Gemini can call four pandas tools: `aggregate` (count, sum, mean, median, min, max, distinct, with grouping, filters and monthly or yearly buckets), `filter_rows`, `value_counts` and `correlation`. The math runs in pandas, so numbers come from the data rather than the model. The model only chooses tools and arguments; nothing it writes is run as code.

Under each answer, "How this was answered" shows the retrieved columns and every tool call with its result.

## Run it locally

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
echo "GOOGLE_API_KEY=your-key" > .env
streamlit run app.py
```

Get a key from [Google AI Studio](https://aistudio.google.com/apikey). Without a key, everything except chat and the preprocessing suggestions still works.

## Configuration

| Variable | Default | |
|---|---|---|
| `GOOGLE_API_KEY` | none | Gemini API key, from `.env`, the environment or Streamlit secrets |
| `GEMINI_MODEL` | `gemini-flash-latest` | Any Gemini chat model that supports tool calling |

On Streamlit Community Cloud, add `GOOGLE_API_KEY` under the app's Settings → Secrets.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests check the pandas tools against plain pandas, retrieval, the chat loop (with a scripted stand-in for Gemini) and that every tab and chart renders.

## Layout

```
app.py              Streamlit UI
stremlit.py         old entry point, kept for the existing deployment; runs app.py
datawiz/
  profile.py        column types, column profiles, sample data
  retrieval.py      TF-IDF index over column profiles and rows
  tools.py          pandas tools the model can call
  chat.py           prompt, tool-calling loop, Gemini setup
tests/
```

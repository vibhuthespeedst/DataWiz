"""Retrieval for the chat: a TF-IDF index over column profiles and rows.

Each question pulls the most relevant column profiles and rows into the prompt, so the model
answers from the actual data instead of a fixed preview. TF-IDF runs locally, so indexing a file
costs no API calls.
"""

from dataclasses import dataclass, field

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .profile import column_profiles, profile_text

MAX_INDEXED_ROWS = 50_000


def row_text(row: pd.Series) -> str:
    return " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))


@dataclass
class Retrieved:
    columns: list[dict] = field(default_factory=list)
    rows: pd.DataFrame = field(default_factory=pd.DataFrame)


class DataIndex:
    def __init__(self, df: pd.DataFrame, max_rows: int = MAX_INDEXED_ROWS):
        self.profiles = column_profiles(df)
        rows = df if len(df) <= max_rows else df.sample(max_rows, random_state=0)
        self.rows = rows
        self.indexed_rows = len(rows)
        docs = [profile_text(p) for p in self.profiles]
        docs += [row_text(r) for _, r in rows.iterrows()]
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", sublinear_tf=True)
        try:
            self.matrix = self.vectorizer.fit_transform(docs)
        except ValueError:  # nothing to index, e.g. an empty file
            self.matrix = None

    def search(self, question: str, k_columns: int = 6, k_rows: int = 8) -> Retrieved:
        if self.matrix is None or not question.strip():
            return Retrieved()
        scores = (self.matrix @ self.vectorizer.transform([question]).T).toarray().ravel()
        n_cols = len(self.profiles)
        col_scores, row_scores = scores[:n_cols], scores[n_cols:]

        q = question.lower()
        named = [i for i, p in enumerate(self.profiles)
                 if p["column"].lower() in q or p["column"].lower().replace("_", " ") in q]
        ranked = [i for i in col_scores.argsort()[::-1] if col_scores[i] > 0 and i not in named]
        columns = [self.profiles[i] for i in (named + ranked)[:k_columns]]

        top = [i for i in row_scores.argsort()[::-1][:k_rows] if row_scores[i] > 0]
        return Retrieved(columns=columns, rows=self.rows.iloc[top])

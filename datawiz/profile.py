"""Column types, per-column profiles and the summary text the app and the chat share."""

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

DATE_HINTS = ("date", "time", "day", "month", "year", "timestamp")


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns that look like dates (by name, and mostly parseable) to datetime."""
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if not (ptypes.is_object_dtype(s) or ptypes.is_string_dtype(s)):
            continue
        if not any(h in str(col).lower() for h in DATE_HINTS):
            continue
        parsed = pd.to_datetime(s, errors="coerce", format="mixed")
        if s.notna().sum() and parsed.notna().sum() / s.notna().sum() >= 0.9:
            out[col] = parsed
    return out


def column_kinds(df: pd.DataFrame) -> dict:
    """Split columns into numeric, datetime, boolean and categorical (everything else)."""
    kinds = {"numeric": [], "datetime": [], "boolean": [], "categorical": []}
    for col in df.columns:
        s = df[col]
        if ptypes.is_bool_dtype(s):
            kinds["boolean"].append(col)
        elif ptypes.is_numeric_dtype(s):
            kinds["numeric"].append(col)
        elif ptypes.is_datetime64_any_dtype(s):
            kinds["datetime"].append(col)
        else:
            kinds["categorical"].append(col)
    return kinds


def _kind_of(kinds: dict, col) -> str:
    for kind, cols in kinds.items():
        if col in cols:
            return kind
    return "categorical"


def _num(x) -> float | None:
    return None if x is None or pd.isna(x) else round(float(x), 4)


def column_profiles(df: pd.DataFrame) -> list[dict]:
    """One dict per column: type, nulls, distinct values and the stats that fit its type."""
    kinds = column_kinds(df)
    profiles = []
    for col in df.columns:
        s = df[col]
        kind = _kind_of(kinds, col)
        p = {
            "column": str(col),
            "kind": kind,
            "non_null": int(s.notna().sum()),
            "nulls": int(s.isna().sum()),
            "unique": int(s.nunique(dropna=True)),
        }
        if kind == "numeric" and p["non_null"]:
            p.update(min=_num(s.min()), max=_num(s.max()), mean=_num(s.mean()),
                     median=_num(s.median()), std=_num(s.std()))
        elif kind == "datetime" and p["non_null"]:
            p.update(min=str(s.min()), max=str(s.max()))
        else:
            top = s.astype("string").value_counts(dropna=True).head(5)
            p["top_values"] = {str(k): int(v) for k, v in top.items()}
        profiles.append(p)
    return profiles


def profile_text(p: dict) -> str:
    """A profile as one line of text, used for retrieval and in the chat prompt."""
    name = p["column"]
    spaced = name.replace("_", " ")
    parts = [f"column {name}" + (f" ({spaced})" if spaced != name else ""),
             f"type {p['kind']}", f"{p['non_null']} values", f"{p['nulls']} missing",
             f"{p['unique']} distinct"]
    if p["kind"] == "numeric" and "mean" in p:
        parts.append(f"min {p['min']}, max {p['max']}, mean {p['mean']}, median {p['median']}")
    elif p["kind"] == "datetime" and "min" in p:
        parts.append(f"from {p['min']} to {p['max']}")
    elif p.get("top_values"):
        parts.append("top values: " + ", ".join(f"{k} ({v})" for k, v in p["top_values"].items()))
    return "; ".join(parts)


def dataset_summary(df: pd.DataFrame) -> str:
    kinds = column_kinds(df)
    lines = [f"{len(df):,} rows, {df.shape[1]} columns."]
    for kind, cols in kinds.items():
        if cols:
            lines.append(f"{kind.capitalize()} columns: {', '.join(map(str, cols))}")
    return "\n".join(lines)


def basic_insights(df: pd.DataFrame) -> str:
    """Markdown summary for the Insights tab."""
    n = len(df)
    kinds = column_kinds(df)
    out = ["**Dataset overview**",
           f"- Shape: {n:,} rows, {df.shape[1]} columns",
           f"- Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
           "", "**Column types**"]
    for kind, cols in kinds.items():
        out.append(f"- {kind.capitalize()}: {len(cols)}")

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    out += ["", "**Missing values**"]
    if missing.empty:
        out.append("- None")
    for col, count in missing.head(5).items():
        out.append(f"- {col}: {count:,} ({count / n * 100:.1f}%)")

    dupes = int(df.duplicated().sum())
    out += ["", f"**Duplicate rows:** {dupes:,}" + (f" ({dupes / n * 100:.1f}%)" if n else "")]

    if kinds["numeric"]:
        out += ["", "**Numeric columns**"]
        for col in kinds["numeric"][:5]:
            s = df[col].dropna()
            if s.empty:
                continue
            out.append(f"- {col}: mean {s.mean():,.2f}, std {s.std():,.2f}, range {s.min():,.2f} to {s.max():,.2f}")
    return "\n".join(out)


def sample_dataset(rows: int = 600, seed: int = 7) -> pd.DataFrame:
    """A small, made-up sales table so the app can be tried without uploading anything."""
    rng = np.random.default_rng(seed)
    products = {
        "Laptop": ("Electronics", 900), "Monitor": ("Electronics", 220), "Headphones": ("Electronics", 80),
        "Desk": ("Furniture", 300), "Chair": ("Furniture", 150), "Notebook": ("Stationery", 4),
        "Pen pack": ("Stationery", 6), "Backpack": ("Accessories", 45),
    }
    names = list(products)
    product = rng.choice(names, rows, p=[0.12, 0.12, 0.16, 0.08, 0.12, 0.16, 0.14, 0.10])
    units = rng.integers(1, 12, rows)
    base = np.array([products[p][1] for p in product], dtype=float)
    unit_price = np.round(base * rng.uniform(0.85, 1.15, rows), 2)
    df = pd.DataFrame({
        "order_id": np.arange(10001, 10001 + rows),
        "order_date": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, rows), unit="D"),
        "region": rng.choice(["North", "South", "East", "West"], rows, p=[0.3, 0.2, 0.25, 0.25]),
        "customer_segment": rng.choice(["Consumer", "Small business", "Enterprise"], rows, p=[0.5, 0.3, 0.2]),
        "product": product,
        "category": [products[p][0] for p in product],
        "units": units,
        "unit_price": unit_price,
    })
    df["revenue"] = np.round(df["units"] * df["unit_price"], 2)
    df.loc[rng.choice(rows, 12, replace=False), "customer_segment"] = None
    return df.sort_values("order_date", ignore_index=True)

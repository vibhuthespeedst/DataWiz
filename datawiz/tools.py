"""Pandas operations the chat model can call to compute answers.

The model only picks a tool and its arguments; the arithmetic runs here, in pandas. Nothing the
model writes is ever executed as code. Bad arguments come back as {"error": ...} so the model can
correct itself.
"""

from typing import Literal, Optional

import pandas as pd
from langchain_core.tools import StructuredTool
from pandas.api import types as ptypes
from pydantic import BaseModel, Field

Op = Literal["==", "!=", ">", ">=", "<", "<=", "contains", "in"]
Agg = Literal["count", "sum", "mean", "median", "min", "max", "nunique"]
Grain = Literal["day", "week", "month", "quarter", "year"]
GRAIN_CODES = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}
MAX_LIMIT = 50


class ToolError(ValueError):
    pass


class Filter(BaseModel):
    column: str = Field(description="Column name, exactly as in the dataset")
    op: Op = Field(description="Comparison. 'in' takes a comma-separated list; 'contains' is a case-insensitive text match")
    value: str = Field(description="Value to compare with, as text")


def _check_columns(df: pd.DataFrame, *cols) -> None:
    missing = [c for c in cols if c is not None and c not in df.columns]
    if missing:
        raise ToolError(f"Unknown column(s): {', '.join(missing)}. Columns are: {', '.join(map(str, df.columns))}")


def _coerce(series: pd.Series, value: str):
    value = value.strip()
    if ptypes.is_bool_dtype(series):
        return value.lower() in ("true", "1", "yes")
    if ptypes.is_numeric_dtype(series):
        try:
            return float(value)
        except ValueError:
            raise ToolError(f"'{value}' is not a number, but {series.name} is numeric")
    if ptypes.is_datetime64_any_dtype(series):
        try:
            return pd.Timestamp(value)
        except ValueError:
            raise ToolError(f"'{value}' is not a date, but {series.name} holds dates")
    return value


def apply_filters(df: pd.DataFrame, filters: Optional[list] = None) -> pd.DataFrame:
    for f in filters or []:
        f = f if isinstance(f, Filter) else Filter(**f)
        _check_columns(df, f.column)
        s = df[f.column]
        if f.op == "contains":
            mask = s.astype("string").str.contains(f.value, case=False, regex=False, na=False)
        elif f.op == "in":
            values = [_coerce(s, v) for v in f.value.split(",") if v.strip()]
            mask = s.isin(values)
        else:
            v = _coerce(s, f.value)
            mask = {"==": s == v, "!=": s != v, ">": s > v, ">=": s >= v, "<": s < v, "<=": s <= v}[f.op]
        df = df[mask.fillna(False).astype(bool)]
    return df


def _scalar(value):
    if value is None or pd.isna(value):
        return None
    if ptypes.is_number(value) and not isinstance(value, bool):
        return round(float(value), 4)
    return str(value)


def _records(frame: pd.DataFrame) -> list[dict]:
    return [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
            for row in frame.astype(object).where(frame.notna(), None).to_dict(orient="records")]


def aggregate(df, column, agg, group_by=None, filters=None, time_grain=None, top=20):
    _check_columns(df, column, *(group_by or []))
    data = apply_filters(df, filters)
    if agg in ("sum", "mean", "median") and not ptypes.is_numeric_dtype(data[column]):
        raise ToolError(f"Can't take the {agg} of {column}: it isn't numeric")
    if not group_by:
        value = len(data) if agg == "count" else data[column].agg(agg)
        return {"rows_matched": len(data), "result": _scalar(value)}

    keys = []
    for col in group_by:
        s = data[col]
        if time_grain and ptypes.is_datetime64_any_dtype(s):
            s = s.dt.to_period(GRAIN_CODES[time_grain]).astype(str)
        keys.append(s.rename(col))
    grouped = data.groupby(keys, dropna=False)[column]
    result = grouped.size() if agg == "count" else grouped.agg(agg)
    result = result.rename(f"{agg}_{column}").reset_index()
    if time_grain:
        result = result.sort_values(group_by)
    else:
        result = result.sort_values(f"{agg}_{column}", ascending=False)
    top = max(1, min(int(top), MAX_LIMIT))
    return {"rows_matched": len(data), "groups": len(result), "result": _records(result.head(top))}


def filter_rows(df, filters=None, columns=None, sort_by=None, descending=True, limit=10):
    _check_columns(df, *(columns or []), sort_by)
    data = apply_filters(df, filters)
    if sort_by:
        data = data.sort_values(sort_by, ascending=not descending)
    limit = max(1, min(int(limit), MAX_LIMIT))
    shown = data[columns] if columns else data
    return {"rows_matched": len(data), "rows": _records(shown.head(limit))}


def value_counts(df, column, filters=None, top=10):
    _check_columns(df, column)
    data = apply_filters(df, filters)
    counts = data[column].value_counts(dropna=False).head(max(1, min(int(top), MAX_LIMIT)))
    total = len(data)
    return {"rows_matched": total, "result": [
        {"value": None if pd.isna(k) else str(k), "count": int(v), "share": round(v / total, 4) if total else None}
        for k, v in counts.items()]}


def correlation(df, column_a, column_b, filters=None):
    _check_columns(df, column_a, column_b)
    data = apply_filters(df, filters)[[column_a, column_b]].dropna()
    for col in (column_a, column_b):
        if not ptypes.is_numeric_dtype(data[col]):
            raise ToolError(f"{col} isn't numeric")
    if len(data) < 3:
        raise ToolError("Fewer than 3 rows have both values")
    r = data[column_a].corr(data[column_b])
    return {"pairs": len(data), "pearson_r": None if pd.isna(r) else round(float(r), 4)}


# Argument schemas, written out so the model sees clear descriptions.

class AggregateArgs(BaseModel):
    column: str = Field(description="Column to aggregate")
    agg: Agg = Field(description="count counts rows; the others apply to the column's values")
    group_by: Optional[list[str]] = Field(None, description="Columns to group by, if any")
    filters: Optional[list[Filter]] = Field(None, description="Row filters applied first, all must match")
    time_grain: Optional[Grain] = Field(None, description="Bucket datetime group_by columns by this period")
    top: int = Field(20, description="Most groups to return, largest first (time buckets come in order)")


class FilterRowsArgs(BaseModel):
    filters: Optional[list[Filter]] = Field(None, description="Row filters, all must match")
    columns: Optional[list[str]] = Field(None, description="Columns to return; all if omitted")
    sort_by: Optional[str] = Field(None, description="Column to sort by")
    descending: bool = Field(True, description="Sort largest first")
    limit: int = Field(10, description=f"Most rows to return, up to {MAX_LIMIT}")


class ValueCountsArgs(BaseModel):
    column: str = Field(description="Column to count values in")
    filters: Optional[list[Filter]] = Field(None, description="Row filters applied first")
    top: int = Field(10, description="Most values to return")


class CorrelationArgs(BaseModel):
    column_a: str = Field(description="First numeric column")
    column_b: str = Field(description="Second numeric column")
    filters: Optional[list[Filter]] = Field(None, description="Row filters applied first")


def _safe(fn, df):
    def run(**kwargs):
        try:
            return fn(df, **kwargs)
        except (ToolError, ValueError, TypeError, KeyError) as e:
            return {"error": str(e)}
    return run


def make_tools(df: pd.DataFrame) -> list[StructuredTool]:
    specs = [
        (aggregate, AggregateArgs, "aggregate",
         "Compute count, sum, mean, median, min, max or distinct count of a column, optionally grouped "
         "and filtered. Use for any total, average, ranking or trend question."),
        (filter_rows, FilterRowsArgs, "filter_rows",
         "Return rows that match filters, optionally sorted. Use to look up specific records or the top N rows."),
        (value_counts, ValueCountsArgs, "value_counts",
         "Count how often each value appears in a column, with its share of rows."),
        (correlation, CorrelationArgs, "correlation",
         "Pearson correlation between two numeric columns."),
    ]
    return [StructuredTool.from_function(func=_safe(fn, df), name=name, description=desc, args_schema=schema)
            for fn, schema, name, desc in specs]

import pytest

from datawiz.profile import sample_dataset
from datawiz.tools import ToolError, aggregate, correlation, filter_rows, make_tools, value_counts


@pytest.fixture
def df():
    return sample_dataset()


def test_grouped_sum_matches_pandas(df):
    out = aggregate(df, "revenue", "sum", group_by=["region"])
    expected = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    assert [r["region"] for r in out["result"]] == list(expected.index)
    assert out["result"][0]["sum_revenue"] == pytest.approx(expected.iloc[0], abs=1e-3)


def test_monthly_buckets_come_in_order(df):
    out = aggregate(df, "revenue", "sum", group_by=["order_date"], time_grain="month")
    months = [r["order_date"] for r in out["result"]]
    assert months == sorted(months)
    assert months[0] == "2025-01" and out["groups"] == 12


def test_filters_coerce_numbers_and_lists(df):
    out = filter_rows(df, filters=[{"column": "units", "op": ">=", "value": "10"},
                                   {"column": "region", "op": "in", "value": "North, East"}])
    expected = df[(df["units"] >= 10) & df["region"].isin(["North", "East"])]
    assert out["rows_matched"] == len(expected)
    assert len(out["rows"]) == min(10, len(expected))


def test_filters_coerce_dates(df):
    out = aggregate(df, "order_id", "count", filters=[{"column": "order_date", "op": ">=", "value": "2025-12-01"}])
    assert out["result"] == (df["order_date"] >= "2025-12-01").sum()


def test_contains_is_case_insensitive(df):
    out = value_counts(df, "product", filters=[{"column": "product", "op": "contains", "value": "LAP"}])
    assert [r["value"] for r in out["result"]] == ["Laptop"]


def test_value_counts_include_missing_values(df):
    out = value_counts(df, "customer_segment")
    assert sum(r["count"] for r in out["result"]) == len(df)
    assert any(r["value"] is None for r in out["result"])


def test_correlation_of_units_and_revenue_is_positive(df):
    assert 0 < correlation(df, "units", "revenue")["pearson_r"] <= 1


def test_mean_of_a_text_column_is_refused(df):
    with pytest.raises(ToolError):
        aggregate(df, "region", "mean")


def test_tool_errors_come_back_as_data(df):
    tools = {t.name: t for t in make_tools(df)}
    out = tools["aggregate"].invoke({"column": "profit", "agg": "sum"})
    assert "Unknown column" in out["error"]
    out = tools["filter_rows"].invoke({"filters": [{"column": "units", "op": ">", "value": "lots"}]})
    assert "not a number" in out["error"]

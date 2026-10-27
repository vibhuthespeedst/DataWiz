import pandas as pd

from datawiz.profile import sample_dataset
from datawiz.retrieval import DataIndex


def test_finds_the_row_a_question_is_about():
    df = pd.DataFrame({"name": ["Alice Smith", "Bob Jones", "Carol White"],
                       "city": ["Pune", "Delhi", "Pune"], "salary": [10, 20, 30]})
    hit = DataIndex(df).search("What does Bob Jones earn?")
    assert hit.rows.iloc[0]["name"] == "Bob Jones"


def test_columns_named_in_the_question_come_first():
    hit = DataIndex(sample_dataset()).search("average unit price by product")
    first_two = {p["column"] for p in hit.columns[:2]}
    assert first_two == {"unit_price", "product"}


def test_large_tables_are_sampled():
    df = pd.DataFrame({"x": range(1000)})
    assert DataIndex(df, max_rows=100).indexed_rows == 100


def test_empty_table_returns_nothing():
    hit = DataIndex(pd.DataFrame()).search("anything")
    assert hit.columns == [] and hit.rows.empty

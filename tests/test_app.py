from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


# stremlit.py is the entry point the existing Streamlit Cloud deployment runs.
@pytest.mark.parametrize("entry", ["app.py", "stremlit.py"])
def test_app_loads_the_sample_and_renders_every_tab(monkeypatch, entry):
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    at = AppTest.from_file(str(ROOT / entry), default_timeout=60).run()
    assert not at.exception

    next(b for b in at.button if b.label.startswith("Try a sample")).click().run()
    assert not at.exception
    assert at.metric[0].value == "600"
    assert any("needs a Gemini API key" in i.value for i in at.info)


def test_every_chart_and_a_filter_render():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    next(b for b in at.button if b.label.startswith("Try a sample")).click().run()
    chart = next(s for s in at.selectbox if s.label == "Chart")
    for option in chart.options:
        chart.set_value(option).run()
        assert not at.exception, option
        chart = next(s for s in at.selectbox if s.label == "Chart")

    at.slider(key="filter_units").set_value((5.0, 11.0)).run()
    assert not at.exception
    matching = next(m for m in at.markdown if "rows match" in m.value).value
    assert not matching.startswith("**600**")

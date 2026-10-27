"""The app lives in app.py. This file stays so the existing Streamlit Cloud deployment, which runs it, keeps working."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")

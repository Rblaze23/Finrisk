"""Entry point for `streamlit run streamlit_app.py` and for Streamlit Community Cloud."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "src" / "finrisk" / "dashboard" / "app.py"),
               run_name="__main__")

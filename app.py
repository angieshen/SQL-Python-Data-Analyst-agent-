import asyncio
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from analyst_agent import AnalystDeps, analyst_agent, create_messy_sample_csv

st.set_page_config(
    page_title="Self-Correcting SQL & Python Data Analyst",
    page_icon="📊",
    layout="wide",
)

st.title("Self-Correcting SQL & Python Data Analyst")
st.caption("Powered by Pydantic AI, DuckDB, and Gemini — upload a messy CSV or Excel file and ask questions in plain English.")

# ---------------------------------------------------------------------------
# Sidebar: Dataset Selection & Preview
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Data Source")
    data_mode = st.radio(
        "Choose input dataset:",
        ["Upload CSV / Excel", "Use Built-in Messy Sample"],
        index=0,
    )

    uploaded_file = None
    if data_mode == "Upload CSV / Excel":
        uploaded_file = st.file_uploader(
            "Upload dataset (.csv, .xlsx, .xls, .parquet)",
            type=["csv", "xlsx", "xls", "parquet"],
        )
    else:
        st.info("Uses `sample_sales.csv` (10 rows with mixed dates, currency strings, and inconsistent region casing).")

# ---------------------------------------------------------------------------
# Helper: Async Agent Runner for Streamlit
# ---------------------------------------------------------------------------
async def execute_agent_ui(input_path: Path, question: str):
    with tempfile.TemporaryDirectory(prefix="streamlit_analyst_") as tmp_dir:
        workspace = Path(tmp_dir)
        target_csv = workspace / "data.csv"

        suffix = input_path.suffix.lower()
        if suffix == ".csv":
            shutil.copy2(input_path, target_csv)
        elif suffix in {".xlsx", ".xls"}:
            pd.read_excel(input_path).to_csv(target_csv, index=False)
        elif suffix == ".parquet":
            pd.read_parquet(input_path).to_csv(target_csv, index=False)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        deps = AnalystDeps(workspace_dir=workspace)
        result = await analyst_agent.run(question, deps=deps)
        report = result.output

        chart_bytes = None
        chart_path = workspace / "chart.png"
        if chart_path.exists():
            chart_bytes = chart_path.read_bytes()

        return report, chart_bytes


# ---------------------------------------------------------------------------
# Main Panel: Question Input & Results
# ---------------------------------------------------------------------------
default_prompt = (
    "Calculate total revenue and churn rate by region. Clean any formatting issues "
    "in region names, dates, or currency strings, and save a bar chart comparing "
    "total revenue across regions."
    if data_mode == "Use Built-in Messy Sample"
    else "What are the top 5 products by net revenue excluding cancellations? Plot a horizontal bar chart."
)

question = st.text_area(
    "2. Ask an analytical question:",
    value=default_prompt,
    height=90,
)

run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

if run_clicked:
    if data_mode == "Upload CSV / Excel" and uploaded_file is None:
        st.warning("Please upload a CSV or Excel file in the sidebar first.")
        st.stop()

    with st.spinner("Probing schema in DuckDB, writing SQL/Python, and self-correcting any errors..."):
        try:
            with tempfile.TemporaryDirectory(prefix="upload_stage_") as stage_dir:
                stage_path = Path(stage_dir)
                if data_mode == "Upload CSV / Excel" and uploaded_file is not None:
                    staged_file = stage_path / uploaded_file.name
                    staged_file.write_bytes(uploaded_file.getbuffer())
                else:
                    staged_file = stage_path / "sample_sales.csv"
                    create_messy_sample_csv(staged_file)

                report, chart_bytes = asyncio.run(execute_agent_ui(staged_file, question))

            st.success("Analysis complete!")

            col_left, col_right = st.columns([1.15, 1])

            with col_left:
                st.subheader("Executive Answer")
                st.markdown(report.answer)

                if report.caveats:
                    st.subheader("Data Quality Caveats")
                    for caveat in report.caveats:
                        st.markdown(f"- {caveat}")

            with col_right:
                st.subheader("Generated Visualization")
                if chart_bytes:
                    st.image(chart_bytes, use_container_width=True)
                    st.download_button(
                        label="Download Chart (PNG)",
                        data=chart_bytes,
                        file_name="analysis_chart.png",
                        mime="image/png",
                    )
                else:
                    st.info("No chart was generated for this query.")

            with st.expander("View Verified Python & DuckDB Code", expanded=False):
                st.code(report.final_code, language="python")

        except Exception as e:
            st.error(f"Analysis failed: {e}")
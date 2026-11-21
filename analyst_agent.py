import os
os.environ["PYDANTIC_AI_NO_BANNER"] = "1"

print("[1/4] Script started...", flush=True)

import asyncio
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google.genai import Client
from google.genai.types import HttpOptions, HttpRetryOptions
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider


# ---------------------------------------------------------------------------
# 1. Structured Output & Local Workspace Dependencies
# ---------------------------------------------------------------------------

@dataclass
class AnalystDeps:
    workspace_dir: Path
    data_filename: str = "data.csv"
    chart_filename: str = "chart.png"


class AnalysisReport(BaseModel):
    answer: str = Field(
        description="Direct, plain-English answer with exact numbers from the analysis."
    )
    final_code: str = Field(
        description="The final, working Python/DuckDB code that produced the result."
    )
    caveats: list[str] = Field(
        description="Data quality quirks discovered (e.g., nulls, string-formatted numbers, date formats)."
    )
    chart_saved: bool = Field(
        description="True if a PNG chart was successfully saved to chart.png in the current working directory."
    )


# ---------------------------------------------------------------------------
# 2. Fast-Failover Gemini Client (Prevents 60s 503 Hangs)
# ---------------------------------------------------------------------------

fast_client = Client(
    http_options=HttpOptions(
        timeout=15_000,  # 15-second timeout per LLM call
        retry_options=HttpRetryOptions(attempts=1),  # Fail over immediately on 503
    )
)
google_provider = GoogleProvider(client=fast_client)

resilient_model = FallbackModel(
    GoogleModel("gemini-3.5-flash-lite", provider=google_provider),
    GoogleModel("gemini-3.8-flash", provider=google_provider),
)

analyst_agent = Agent(
    resilient_model,
    deps_type=AnalystDeps,
    output_type=AnalysisReport,
    retries=4,
    system_prompt=(
        "You are an expert Staff Data Analyst operating inside an isolated local workspace directory.\n"
        "Follow this strict workflow for every request:\n"
        "1. SCHEMA PROBE FIRST: Call `inspect_dataset` before writing analytical queries.\n"
        "2. USE DUCKDB FOR SQL: Query `'data.csv'` via `duckdb.sql(\"SELECT ... FROM 'data.csv'\")`. "
        "Important: In this DuckDB version, relations do NOT have `.fetchone()`—always use `.fetchall()` or `.df()`.\n"
        "3. GENERATE CHARTS: Save matplotlib figures to `'chart.png'` in the current working directory "
        "(dpi=150, bbox_inches='tight'). Do not call `plt.show()`.\n"
        "4. VERIFY OUTPUTS: Always `print()` concise summary tables. Never print more than 30 rows."
    ),
)


# ---------------------------------------------------------------------------
# 3. Local Subprocess Execution Helper & Tools
# ---------------------------------------------------------------------------

def run_in_workspace(code: str, cwd: Path, timeout_sec: int = 30) -> subprocess.CompletedProcess:
    script_path = cwd / "_run_step.py"
    script_path.write_text(code, encoding="utf-8")

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    return subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )


@analyst_agent.tool
def inspect_dataset(ctx: RunContext[AnalystDeps]) -> str:
    """Probes data.csv using DuckDB to inspect schema, row count, and sample rows."""
    print("[Tool] Probing dataset schema in DuckDB...", flush=True)
    probe_code = f"""
import duckdb

path = "{ctx.deps.data_filename}"
row_count = duckdb.sql(f"SELECT COUNT(*) FROM '{{path}}'").fetchall()[0][0]
schema_df = duckdb.sql(f"DESCRIBE SELECT * FROM '{{path}}'").df()
sample_df = duckdb.sql(f"SELECT * FROM '{{path}}' LIMIT 5").df()

print(f"TOTAL ROWS: {{row_count}}\\n")
print("=== SCHEMA ===")
print(schema_df.to_string(index=False))
print("\\n=== SAMPLE ROWS ===")
print(sample_df.to_string(index=False))
"""
    try:
        proc = run_in_workspace(probe_code, ctx.deps.workspace_dir)
    except subprocess.TimeoutExpired:
        raise ModelRetry("Schema inspection timed out after 30 seconds.")

    if proc.returncode != 0:
        print(f"[Schema Probe Error] {proc.stderr.strip()}", flush=True)
        raise ModelRetry(f"Schema inspection failed:\n{proc.stderr}")

    print("[Tool] Schema probe finished! Sending results to LLM...", flush=True)
    return proc.stdout.strip()


@analyst_agent.tool
def execute_analysis_code(ctx: RunContext[AnalystDeps], python_code: str) -> str:
    """Runs Python + DuckDB + Matplotlib code in the local workspace with automatic self-correction."""
    print("[Tool] Executing analysis code locally...", flush=True)
    try:
        proc = run_in_workspace(python_code, ctx.deps.workspace_dir)
    except subprocess.TimeoutExpired:
        raise ModelRetry("Execution timed out after 30 seconds. Optimize your query.")

    if proc.returncode != 0:
        first_err_line = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "Unknown error"
        print(f"[Self-Correcting] Caught error: {first_err_line}", flush=True)
        raise ModelRetry(
            f"Code execution failed (exit code {proc.returncode}).\n"
            f"Stderr:\n{proc.stderr}\n"
            "Fix the error and try again."
        )

    stdout = proc.stdout.strip()
    if not stdout:
        raise ModelRetry(
            "Code executed without errors, but stdout was completely empty. "
            "Add `print()` statements outputting your computed metrics."
        )

    if len(stdout) > 4000:
        stdout = stdout[:4000] + "\n...[OUTPUT TRUNCATED AT 4000 CHARS]..."

    print("[Tool] Analysis script succeeded! Formatting final report...", flush=True)
    return stdout


# ---------------------------------------------------------------------------
# 4. Helper: Create Messy Sample CSV
# ---------------------------------------------------------------------------

def create_messy_sample_csv(filepath: Path) -> None:
    csv_content = """order_id,order_date,region,plan_tier,revenue_raw,churned
1001,2026-01-05,North America,Enterprise,"$4,500.00",false
1002,2026-01-12,north america,Pro,$850.00,false
1003,01/18/2026,EMEA,Enterprise,"$6,200.50",false
1004,2026-02-02,EMEA,Starter,$120.00,true
1005,2026-02-14,APAC,Pro,"$1,100.00",false
1006,2026-02-25,North America,Enterprise,NULL,true
1007,2026-03-03,emea,Pro,"$950.00",false
1008,2026-03-11,APAC,Enterprise,"$5,100.00",false
1009,2026-03-19,North America,Pro,"$1,250.00",false
1010,2026-03-28,EMEA,Enterprise,"$7,400.00",false
"""
    filepath.write_text(csv_content.strip(), encoding="utf-8")


# ---------------------------------------------------------------------------
# 5. Orchestrator
# ---------------------------------------------------------------------------

async def run_analyst(local_file: Path, question: str, chart_out: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="analyst_ws_") as tmp_dir:
        workspace = Path(tmp_dir)
        target_csv = workspace / "data.csv"

        suffix = local_file.suffix.lower()
        if suffix == ".csv":
            shutil.copy2(local_file, target_csv)
        elif suffix in {".xlsx", ".xls"}:
            import pandas as pd
            print(f"[2/4] Converting {local_file.name} to CSV for DuckDB...", flush=True)
            pd.read_excel(local_file).to_csv(target_csv, index=False)
        elif suffix == ".parquet":
            import pandas as pd
            pd.read_parquet(local_file).to_csv(target_csv, index=False)
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Use .csv, .xlsx, or .parquet.")

        print(f"[2/4] Prepared isolated workspace at {workspace}", flush=True)

        deps = AnalystDeps(workspace_dir=workspace)

        print("[3/4] Running Pydantic AI agent loop...", flush=True)
        result = await analyst_agent.run(question, deps=deps)
        report = result.output

        print("[4/4] Analysis complete!\n", flush=True)
        generated_chart = workspace / "chart.png"
        if generated_chart.exists():
            shutil.copy2(generated_chart, chart_out)
            print(f"[Chart Saved] Extracted chart to: {chart_out.resolve()}\n")

        print("=== FINAL ANSWER ===")
        print(report.answer)

        print("\n=== DATA QUALITY CAVEATS ===")
        for caveat in report.caveats:
            print(f"- {caveat}")

        print("\n=== VERIFIED EXECUTED CODE ===")
        print(report.final_code)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Self-Correcting SQL & Python Data Analyst Agent"
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("sample_sales.csv"),
        help="Path to the input CSV file (defaults to sample_sales.csv).",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=(
            "Calculate total revenue and churn rate by region. Clean any formatting issues "
            "in region names, dates, or currency strings, and save a bar chart comparing "
            "total revenue across regions."
        ),
        help="Natural-language analytical question to answer.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output_chart.png"),
        help="Output path for the generated PNG chart.",
    )

    args = parser.parse_args()

    # Auto-create sample_sales.csv only if using the default sample file
    if args.file == Path("sample_sales.csv") and not args.file.exists():
        create_messy_sample_csv(args.file)
    elif not args.file.exists():
        raise FileNotFoundError(f"Input file not found: {args.file.resolve()}")

    asyncio.run(
        run_analyst(
            local_file=args.file,
            question=args.question,
            chart_out=args.out,
        )
    )
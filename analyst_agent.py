import os
os.environ["PYDANTIC_AI_NO_BANNER"] = "1"

print("[1/4] Script started...", flush=True)

import argparse
import asyncio
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
import textwrap

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, ToolOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider


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
# 2. Local Ollama Agent Factory (No API Key Required)
# ---------------------------------------------------------------------------

DEFAULT_LOCAL_MODEL = "qwen3:8b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

SYSTEM_PROMPT = (
    "You are an expert Staff Data Analyst operating inside an isolated local workspace directory.\n"
    "Follow this strict workflow for every request:\n"
    "1. SCHEMA PROBE FIRST: Call `inspect_dataset` before writing analytical queries.\n"
    "2. RUN SQL + CHART VIA `execute_analysis_code`:\n"
    "   - Pass pure DuckDB SQL in `sql_query` querying `'data.csv'`.\n"
    "   - Always use `TRY_CAST(...)` (never `CAST(...)`) when parsing messy currency/numeric strings "
    "(e.g., `TRY_CAST(REGEXP_REPLACE(revenue_raw, '[$,]', '', 'g') AS DOUBLE)`), so `'NULL'` or `''` safely returns `NULL`.\n"
    "   - Pass short Matplotlib code in `plot_code` using the pre-loaded pandas DataFrame `df` to save `'chart.png'` "
    "(dpi=150, bbox_inches='tight'). Do not call `plt.show()` and do not use triple-quoted strings.\n"
)


def build_agent(model_name: str = DEFAULT_LOCAL_MODEL, base_url: str = OLLAMA_BASE_URL) -> Agent[AnalystDeps, AnalysisReport]:
    local_model = OpenAIChatModel(
        model_name=model_name,
        provider=OllamaProvider(base_url=base_url),
    )

    agent = Agent(
        local_model,
        deps_type=AnalystDeps,
        output_type=ToolOutput(AnalysisReport),
        retries=8,
        system_prompt=SYSTEM_PROMPT,
    )

    @agent.tool
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

        print("[Tool] Schema probe finished! Sending schema to local model...", flush=True)
        return proc.stdout.strip()

    @agent.tool
    def execute_analysis_code(ctx: RunContext[AnalystDeps], sql_query: str, plot_code: str = "") -> str:
        """Executes a DuckDB SQL query against data.csv into `df`, prints `df`, and optionally runs `plot_code` to save chart.png."""
        print("[Tool] Executing DuckDB SQL + Matplotlib locally...", flush=True)

        (ctx.deps.workspace_dir / "_query.sql").write_text(sql_query, encoding="utf-8")

        clean_plot_code = textwrap.dedent(plot_code).strip()
        runner_script = (
            "import duckdb\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n\n"
            "sql = open('_query.sql', encoding='utf-8').read()\n"
            "df = duckdb.sql(sql).df()\n"
            "print(df.head(30).to_string(index=False))\n\n"
            f"{clean_plot_code}\n"
        )

        try:
            proc = run_in_workspace(runner_script, ctx.deps.workspace_dir)
        except subprocess.TimeoutExpired:
            raise ModelRetry("Execution timed out after 30 seconds. Optimize your query.")

        if proc.returncode != 0:
            err_lines = [line for line in proc.stderr.strip().splitlines() if line.strip()]
            summary_err = err_lines[-1] if err_lines else "Unknown error"
            print(f"[Self-Correcting] Caught error: {summary_err}", flush=True)
            raise ModelRetry(
                f"Execution failed (exit code {proc.returncode}).\n"
                f"Stderr:\n{proc.stderr}\n"
                "Fix the SQL or plot_code and try again."
            )

        stdout = proc.stdout.strip()
        if len(stdout) > 4000:
            stdout = stdout[:4000] + "\n...[OUTPUT TRUNCATED AT 4000 CHARS]..."

        print("[Tool] Analysis script succeeded! Formatting final report...", flush=True)
        return stdout or "Query executed successfully (0 rows returned)."

    return agent


# Default agent instance so `from analyst_agent import analyst_agent` in app.py works out of the box
analyst_agent = build_agent()


# ---------------------------------------------------------------------------
# 3. Local Subprocess Execution Helper
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

async def run_analyst(
    local_file: Path,
    question: str,
    chart_out: Path,
    model_name: str = DEFAULT_LOCAL_MODEL,
) -> None:
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
        agent = analyst_agent if model_name == DEFAULT_LOCAL_MODEL else build_agent(model_name=model_name)

        print(f"[3/4] Running Pydantic AI agent loop via local Ollama ({model_name})...", flush=True)
        result = await agent.run(question, deps=deps)
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
    parser = argparse.ArgumentParser(
        description="100% Local Self-Correcting SQL & Python Data Analyst Agent (Ollama + DuckDB)"
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("sample_sales.csv"),
        help="Path to the input dataset (.csv, .xlsx, or .parquet).",
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
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LOCAL_MODEL,
        help=f"Local Ollama model name (default: {DEFAULT_LOCAL_MODEL}).",
    )

    args = parser.parse_args()

    if args.file == Path("sample_sales.csv") and not args.file.exists():
        create_messy_sample_csv(args.file)
    elif not args.file.exists():
        raise FileNotFoundError(f"Input file not found: {args.file.resolve()}")

    asyncio.run(
        run_analyst(
            local_file=args.file,
            question=args.question,
            chart_out=args.out,
            model_name=args.model,
        )
    )
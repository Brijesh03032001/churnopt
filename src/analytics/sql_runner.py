"""
Run the SQL analysis queries with DuckDB.

Each file in sql/ answers one business question against the raw CSV. The
runner executes `00_setup.sql` once (it defines the `customers` view), then
every numbered query, and writes each result to outputs/sql/.
"""

import re
from pathlib import Path
from typing import Dict

import duckdb
import pandas as pd

from ..utils import ensure_dir, load_config, setup_logging

logger = setup_logging(__name__)


def query_files(sql_dir: str) -> list:
    """Numbered analysis queries, excluding the setup script, in order."""
    return sorted(
        path for path in Path(sql_dir).glob("[0-9][0-9]_*.sql") if not path.name.startswith("00_")
    )


def run_sql_analysis(config: Dict, write_outputs: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Execute every SQL query against the raw CSV.

    Args:
        config: Configuration dictionary from project.yaml
        write_outputs: Whether to save each result as CSV under outputs/sql/

    Returns:
        Dictionary mapping query name to its result DataFrame
    """
    sql_dir = config["paths"]["sql"]
    raw_csv = str(Path(config["paths"]["raw_data"]).resolve())

    connection = duckdb.connect()
    setup = (Path(sql_dir) / "00_setup.sql").read_text().replace("{{raw_csv}}", raw_csv)
    connection.execute(setup)

    output_dir = ensure_dir(f"{config['paths']['outputs']}/sql") if write_outputs else None
    results = {}
    for path in query_files(sql_dir):
        name = re.sub(r"^\d+_", "", path.stem)
        frame = connection.execute(path.read_text()).df()
        results[name] = frame
        if output_dir is not None:
            frame.to_csv(output_dir / f"{path.stem}.csv", index=False)
        logger.info(f"SQL {path.name}: {len(frame)} rows")

    connection.close()
    return results


def main():
    config = load_config("configs/project.yaml")
    results = run_sql_analysis(config)
    pd.set_option("display.width", 200)
    for name, frame in results.items():
        print(f"\n== {name} ==")
        print(frame.head(12).to_string(index=False))


if __name__ == "__main__":
    main()

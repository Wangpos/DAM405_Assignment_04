"""Data-quality gate on the raw extract, built with Great Expectations (GX Core 1.x).

The gate sits between `extract` and `transform`: if any expectation fails, the
report is still written (for debugging) and QualityGateError stops the pipeline.
"""
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("GX_ANALYTICS_ENABLED", "False")
import great_expectations as gx  # noqa: E402
import great_expectations.expectations as gxe  # noqa: E402

from pipeline import QualityGateError, schema  # noqa: E402
from pipeline.storage import atomic_write_json  # noqa: E402

MIN_ROWS, MAX_ROWS = 500, 100_000
# Share of benign cases; the source has ~63%. A large swing means a broken feed.
TARGET_MEAN_RANGE = (0.3, 0.8)


def build_expectations() -> list:
    expectations = [
        gxe.ExpectTableColumnsToMatchSet(
            column_set=[*schema.RAW_FEATURES, schema.TARGET], exact_match=True
        ),
        gxe.ExpectTableRowCountToBeBetween(min_value=MIN_ROWS, max_value=MAX_ROWS),
        gxe.ExpectColumnValuesToNotBeNull(column=schema.TARGET),
        gxe.ExpectColumnValuesToBeInSet(column=schema.TARGET, value_set=[0, 1]),
        gxe.ExpectColumnMeanToBeBetween(
            column=schema.TARGET, min_value=TARGET_MEAN_RANGE[0], max_value=TARGET_MEAN_RANGE[1]
        ),
    ]
    for column in schema.RAW_FEATURES:
        # Every feature is a physical measurement: present and non-negative.
        expectations.append(gxe.ExpectColumnValuesToNotBeNull(column=column))
        expectations.append(gxe.ExpectColumnValuesToBeBetween(column=column, min_value=0))
    return expectations


def run_checks(df: pd.DataFrame) -> dict:
    context = gx.get_context(mode="ephemeral")
    batch = (
        context.data_sources.add_pandas("pipeline")
        .add_dataframe_asset("raw_extract")
        .add_batch_definition_whole_dataframe("full_frame")
        .get_batch(batch_parameters={"dataframe": df})
    )
    suite = context.suites.add(gx.ExpectationSuite(name="raw_breast_cancer"))
    for expectation in build_expectations():
        suite.add_expectation(expectation)

    result = batch.validate(suite)
    failed = [
        {
            "expectation": r.expectation_config.type,
            "column": r.expectation_config.kwargs.get("column"),
            "observed_value": r.result.get("observed_value"),
            "unexpected_count": r.result.get("unexpected_count"),
        }
        for r in result.results
        if not r.success
    ]
    return {
        "success": bool(result.success),
        "evaluated": len(result.results),
        "failed": failed,
        "statistics": result.statistics,
    }


def validate_raw(raw_path) -> str:
    """Run the gate on a raw CSV. Returns the report path, or raises QualityGateError."""
    report = run_checks(pd.read_csv(raw_path))
    report_path = Path(raw_path).with_name("quality_report.json")
    atomic_write_json(report, report_path)

    if not report["success"]:
        summary = ", ".join(
            f"{f['expectation']}({f['column']})" if f["column"] else f["expectation"]
            for f in report["failed"]
        )
        raise QualityGateError(
            f"{len(report['failed'])}/{report['evaluated']} expectations failed: {summary}. "
            f"See {report_path}"
        )
    return str(report_path)

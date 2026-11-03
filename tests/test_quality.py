import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer

from pipeline import QualityGateError, quality, steps


@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    return load_breast_cancer(as_frame=True).frame


def test_clean_data_passes(clean_df):
    report = quality.run_checks(clean_df)
    assert report["success"], report["failed"]


def test_injected_bad_data_fails(clean_df):
    report = quality.run_checks(steps._inject_bad_data(clean_df))
    failed = {(f["expectation"], f["column"]) for f in report["failed"]}
    assert not report["success"]
    assert ("expect_column_values_to_not_be_null", "mean radius") in failed
    assert ("expect_column_values_to_be_between", "mean area") in failed
    assert ("expect_column_values_to_be_in_set", "target") in failed


def test_missing_column_fails(clean_df):
    assert not quality.run_checks(clean_df.drop(columns=["worst area"]))["success"]


def test_validate_raw_raises_and_writes_report(tmp_path, clean_df):
    raw = tmp_path / "raw.csv"
    steps._inject_bad_data(clean_df).to_csv(raw, index=False)
    with pytest.raises(QualityGateError):
        quality.validate_raw(raw)
    assert (tmp_path / "quality_report.json").exists()

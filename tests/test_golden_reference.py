"""The golden questions' reference plans give exactly pandas' answers (executor check)."""

from pathlib import Path
from typing import Any

import pytest

from app.connectors.csv_xlsx import ingest_file
from app.query.executor import clear_cache, columns_from_profile, execute_plan, validate_plan
from app.schemas.query_plan import QueryPlan
from evals.golden import CASES, SAMPLE, TODAY, frame, matches

FILES = {"sales": "sales_orders.csv", "hr": "hr_employees.csv"}


@pytest.fixture(scope="module")
def parquet(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[Path, Any]]:
    out = {}
    for name, file in FILES.items():
        path = tmp_path_factory.mktemp(name) / "data.parquet"
        res = ingest_file(SAMPLE / file, "csv", path)
        out[name] = (path, columns_from_profile([c.to_dict() for c in res.profile.columns]))
    return out


def test_there_are_at_least_30_golden_questions() -> None:
    assert len(CASES) >= 30


@pytest.mark.parametrize("case", CASES, ids=[c.question for c in CASES])
def test_reference_plan_matches_pandas(case: Any, parquet: dict[str, Any]) -> None:
    clear_cache()
    path, columns = parquet[case.dataset]
    vp = validate_plan(QueryPlan.model_validate(case.plan), columns, TODAY)
    result = execute_plan(vp, path, dataset_id=case.dataset, version=1)
    assert matches(case.expected(frame(case.dataset)), result.rows, case.compare), result.rows

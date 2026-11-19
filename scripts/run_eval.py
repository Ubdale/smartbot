"""Score the configured LLM on the golden questions (planning accuracy, end to end).

    uv run python scripts/run_eval.py                       # uses .env (LLM_PROVIDER, ...)
    LLM_PROVIDER=groq uv run python scripts/run_eval.py      # another provider
    uv run python scripts/run_eval.py --only sales

A case passes when the model's plan, executed by AskData, gives pandas' answer.
Summaries are skipped (they don't change the numbers).
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.connectors.csv_xlsx import ingest_file
from app.core.config import get_settings
from app.llm.base import LLMUnavailableError
from app.llm.factory import NoSummaries, make_provider
from app.query.answer import answer_question
from app.query.executor import clear_cache
from evals.golden import CASES, SAMPLE, TODAY, frame, matches

FILES = {"sales": ("sales_orders.csv", "sales_orders"), "hr": ("hr_employees.csv", "hr_employees")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["sales", "hr"])
    parser.add_argument("--target", type=float, default=0.85)
    parser.add_argument("--pause", type=float, default=0.0, help="seconds between questions")
    args = parser.parse_args()

    settings = get_settings()
    llm = make_provider(settings)
    work = Path(tempfile.mkdtemp())
    data = {}
    for key, (file, name) in FILES.items():
        out = work / f"{key}.parquet"
        res = ingest_file(SAMPLE / file, "csv", out)
        data[key] = (out, name, res.profile.row_count, [c.to_dict() for c in res.profile.columns])

    cases = [c for c in CASES if not args.only or c.dataset == args.only]
    passed, started = 0, time.perf_counter()
    plan: object
    errors = 0
    print(f"Model: {llm.name} / {llm.model}  ({len(cases)} questions)\n")
    for i, case in enumerate(cases, start=1):
        clear_cache()
        path, name, rows, profile = data[case.dataset]
        t = time.perf_counter()
        try:
            answer = answer_question(
                llm, case.question, dataset_id=case.dataset, dataset_name=name,
                dataset_version=1, row_count=rows, profile=profile, parquet=path,
                history=[], today=TODAY, summarizer=NoSummaries(),
            )  # fmt: skip
            ok = answer.result is not None and matches(
                case.expected(frame(case.dataset)), answer.result.rows, case.compare
            )
            plan = (
                answer.validated.plan.model_dump(exclude_defaults=True)
                if answer.validated
                else answer.status
            )
        except LLMUnavailableError as exc:
            ok, plan = False, f"model error: {exc}"
            errors += 1
        passed += ok
        time.sleep(args.pause)
        mark = "PASS" if ok else "FAIL"
        print(f"{i:2}. {mark} {time.perf_counter() - t:5.1f}s  {case.question}")
        if not ok:
            print(f"          got plan: {plan}")
    answered = len(cases) - errors
    if errors:
        print(f"\n{errors} question(s) never reached the model (rate limits/outages): not scored.")
    score = passed / answered if answered else 0.0
    print(f"\nScore: {passed}/{answered} = {score:.0%}  (target {args.target:.0%}), "
          f"{time.perf_counter() - started:.0f}s total")  # fmt: skip
    return 0 if score >= args.target else 1


if __name__ == "__main__":
    raise SystemExit(main())

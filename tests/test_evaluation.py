from datetime import date
from pathlib import Path
from typing import Any, get_args

import pytest
from fakes import FakeLLM

from claim_extractor.evaluation import (
    EVALUATED_FIELDS,
    EvalCase,
    ExpectedFields,
    compare,
    format_report,
    load_cases,
    run_evaluation,
    summarize,
)
from claim_extractor.extractor import FEW_SHOT_EXAMPLES
from claim_extractor.models import ClaimExtraction, RequestType, Urgency

EVAL_SET = Path(__file__).parents[1] / "data" / "eval_set.jsonl"
EXPECTED = ExpectedFields(
    type_demande="remboursement",
    montant=86.4,
    date_incident=date(2026, 10, 3),
    urgence="moyenne",
)


def make_output(**changes: Any) -> ClaimExtraction:
    """A ClaimExtraction that matches EXPECTED, except for `changes`."""
    fields = EXPECTED.model_dump() | {"resume": "Double débit du pass.", "confiance": 0.9}
    return ClaimExtraction.model_validate(fields | changes)


def make_case(case_id: str) -> EvalCase:
    return EvalCase(
        id=case_id,
        received_on=date(2026, 10, 20),
        text="Mon pass a été débité deux fois le 3 octobre, 86,40 €.",
        expected=EXPECTED,
    )


def test_eval_set_is_valid_and_covers_every_category() -> None:
    cases = load_cases(EVAL_SET)

    assert len(cases) == 20
    assert len({case.id for case in cases}) == len(cases)
    assert {case.expected.type_demande for case in cases} == set(get_args(RequestType))
    assert {case.expected.urgence for case in cases} == set(get_args(Urgency))
    assert any(case.expected.montant is None for case in cases)
    assert any(case.expected.date_incident is None for case in cases)
    # Labelling sanity check: an incident cannot happen after the claim was received.
    assert all(
        case.expected.date_incident is None or case.expected.date_incident <= case.received_on
        for case in cases
    )


def test_few_shot_examples_are_not_taken_from_the_eval_set() -> None:
    eval_texts = {case.text for case in load_cases(EVAL_SET)}

    assert not eval_texts & {text for text, _ in FEW_SHOT_EXAMPLES}


def test_load_cases_points_at_the_faulty_line(tmp_path: Path) -> None:
    path = tmp_path / "eval_set.jsonl"
    path.write_text(make_case("ok").model_dump_json() + '\n\n{"id": "broken"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 3"):
        load_cases(path)


def test_matching_output_matches_every_field() -> None:
    assert compare(EXPECTED, make_output()) == dict.fromkeys(EVALUATED_FIELDS, True)


def test_invalid_output_is_wrong_on_every_field() -> None:
    assert compare(EXPECTED, None) == dict.fromkeys(EVALUATED_FIELDS, False)


@pytest.mark.parametrize(
    ("expected_amount", "actual_amount", "same"),
    [
        (86.4, 86.4, True),
        (86.4, 86.400001, True),  # float noise
        (86.4, 86.41, False),  # one cent off
        (None, None, True),
        (None, 0.0, False),  # "no amount" is not "zero euros"
        (20.0, None, False),
    ],
)
def test_amount_comparison(
    expected_amount: float | None, actual_amount: float | None, same: bool
) -> None:
    expected = EXPECTED.model_copy(update={"montant": expected_amount})

    assert compare(expected, make_output(montant=actual_amount))["montant"] is same


def test_run_evaluation_scores_valid_and_invalid_outputs() -> None:
    llm = FakeLLM([make_output().model_dump_json(), "pas du JSON", "toujours pas"])

    results = run_evaluation([make_case("c1"), make_case("c2")], llm)

    assert [result.output is not None for result in results] == [True, False]
    assert "2026-10-20" in llm.calls[0][0]["content"]  # received_on is used as "today"
    summary = summarize(results)
    assert summary.valid_json_rate == 0.5
    assert summary.accuracy == dict.fromkeys(EVALUATED_FIELDS, 0.5)


def test_report_shows_scores_and_mismatches() -> None:
    llm = FakeLLM([make_output(urgence="haute").model_dump_json()])

    report = format_report(run_evaluation([make_case("c1")], llm), model="fake-model")

    assert "fake-model" in report
    assert "Valid JSON" in report
    mismatches = [line.split() for line in report.splitlines() if "expected=" in line]
    assert mismatches == [["c1", "urgence", "expected=moyenne", "got=haute"]]

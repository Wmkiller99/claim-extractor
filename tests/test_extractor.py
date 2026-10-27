import json
import logging
from datetime import date
from typing import get_args

import pytest
from fakes import FakeLLM

from claim_extractor.extractor import (
    FEW_SHOT_EXAMPLES,
    ExtractionError,
    build_date_hints,
    extract_claim,
)
from claim_extractor.llm.base import LLMError
from claim_extractor.models import RequestType

CLAIM = "Mon pass Navigo a été débité deux fois le 3 octobre, 86,40 €, je veux être remboursé"
TODAY = date(2025, 10, 20)
VALID_REPLY = json.dumps(
    {
        "type_demande": "remboursement",
        "montant": 86.40,
        "date_incident": "2025-10-03",
        "urgence": "moyenne",
        "resume": "Le pass Navigo du client a été débité deux fois.",
        "confiance": 0.9,
    }
)
INVALID_REPLY = VALID_REPLY.replace('"remboursement"', '"plainte"')


def test_valid_first_reply_needs_a_single_call() -> None:
    llm = FakeLLM([VALID_REPLY])

    claim = extract_claim(CLAIM, llm, today=TODAY)

    assert claim.type_demande == "remboursement"
    assert claim.montant == 86.40
    assert len(llm.calls) == 1


def test_prompt_has_schema_dates_examples_then_the_claim() -> None:
    llm = FakeLLM([VALID_REPLY])

    extract_claim(CLAIM, llm, today=TODAY)

    system, *examples, user = llm.calls[0]
    assert system["role"] == "system"
    assert "lundi 2025-10-20" in system["content"]  # TODAY, in the date hints
    assert '"additionalProperties": false' in system["content"]  # the JSON schema
    assert [message["role"] for message in examples] == ["user", "assistant"] * len(
        FEW_SHOT_EXAMPLES
    )
    assert user == {"role": "user", "content": CLAIM}


def test_date_hints_resolve_relative_dates() -> None:
    hints = build_date_hints(date(2026, 10, 19))  # a Monday

    assert "- hier : dimanche 2026-10-18" in hints
    assert "- jeudi dernier : 2026-10-15" in hints
    assert "- lundi dernier : 2026-10-12" in hints  # a week ago, not today
    assert "en 2026 si elle tombe au plus tard aujourd'hui, sinon en 2025" in hints


def test_few_shot_examples_cover_every_category() -> None:
    categories = {example.type_demande for _, example in FEW_SHOT_EXAMPLES}

    assert categories == set(get_args(RequestType))


@pytest.mark.parametrize(
    ("bad_reply", "reported_field"),
    [
        (INVALID_REPLY, "type_demande"),  # valid JSON, wrong value
        ("Voici le JSON demandé : ...", "json"),  # not JSON at all
    ],
)
def test_invalid_reply_is_retried_with_the_errors(bad_reply: str, reported_field: str) -> None:
    llm = FakeLLM([bad_reply, VALID_REPLY])

    claim = extract_claim(CLAIM, llm, today=TODAY)

    assert claim.type_demande == "remboursement"
    assert len(llm.calls) == 2
    *_, previous_answer, correction = llm.calls[1]
    assert previous_answer == {"role": "assistant", "content": bad_reply}
    assert correction["role"] == "user"
    assert f"- {reported_field} :" in correction["content"]


def test_second_invalid_reply_raises_extraction_error() -> None:
    llm = FakeLLM([INVALID_REPLY, INVALID_REPLY, VALID_REPLY])

    with pytest.raises(ExtractionError, match="type_demande") as exc_info:
        extract_claim(CLAIM, llm, today=TODAY)

    assert len(llm.calls) == 2  # exactly one retry, never a third call
    assert exc_info.value.raw_output == INVALID_REPLY


def test_llm_error_is_logged_and_not_retried(caplog: pytest.LogCaptureFixture) -> None:
    llm = FakeLLM([LLMError("connection refused"), VALID_REPLY])

    with pytest.raises(LLMError):
        extract_claim(CLAIM, llm, today=TODAY)

    assert len(llm.calls) == 1
    assert "outcome=llm_error" in caplog.text


def test_empty_claim_is_rejected_without_calling_the_llm() -> None:
    llm = FakeLLM([])

    with pytest.raises(ValueError, match="empty"):
        extract_claim("   ", llm)

    assert llm.calls == []


def test_each_attempt_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="claim_extractor")
    llm = FakeLLM([INVALID_REPLY, VALID_REPLY])

    extract_claim(CLAIM, llm, today=TODAY)

    logs = [record.getMessage() for record in caplog.records]
    assert len(logs) == 2
    assert "model=fake-model attempt=1" in logs[0]
    assert "outcome=invalid_output" in logs[0]
    assert "attempt=2" in logs[1]
    assert "outcome=success" in logs[1]
    assert all("duration_ms=" in line for line in logs)

import json
from datetime import date
from typing import Any

import pytest
from pydantic import ValidationError

from claim_extractor.models import ClaimExtraction


@pytest.fixture
def payload() -> dict[str, Any]:
    """A valid extraction for the Navigo double-debit example."""
    return {
        "type_demande": "remboursement",
        "montant": 86.40,
        "date_incident": "2025-10-03",
        "urgence": "moyenne",
        "resume": "Le client demande le remboursement d'un double débit de son pass Navigo.",
        "confiance": 0.92,
    }


def test_valid_payload_is_parsed(payload: dict[str, Any]) -> None:
    claim = ClaimExtraction.model_validate(payload)

    assert claim.type_demande == "remboursement"
    assert claim.montant == 86.40
    assert claim.date_incident == date(2025, 10, 3)


def test_nullable_fields_accept_null(payload: dict[str, Any]) -> None:
    payload |= {"montant": None, "date_incident": None}

    claim = ClaimExtraction.model_validate(payload)

    assert claim.montant is None
    assert claim.date_incident is None


@pytest.mark.parametrize("field", list(ClaimExtraction.model_fields))
def test_missing_field_is_rejected(payload: dict[str, Any], field: str) -> None:
    del payload[field]

    with pytest.raises(ValidationError, match=field):
        ClaimExtraction.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("type_demande", "plainte"),  # not one of the allowed categories
        ("urgence", "critique"),
        ("montant", -5),
        ("montant", "86,40"),  # a French decimal comma is not a number
        ("date_incident", "3 octobre"),  # not ISO 8601
        ("date_incident", "2025-13-01"),  # month 13 does not exist
        ("confiance", 1.5),
        ("confiance", -0.1),
        ("resume", "   "),  # blank once whitespace is stripped
        ("resume", "x" * 301),
    ],
)
def test_invalid_value_is_rejected(payload: dict[str, Any], field: str, value: Any) -> None:
    payload[field] = value

    with pytest.raises(ValidationError, match=field):
        ClaimExtraction.model_validate(payload)


def test_unknown_field_is_rejected(payload: dict[str, Any]) -> None:
    payload["commentaire"] = "champ inventé par le LLM"

    with pytest.raises(ValidationError, match="commentaire"):
        ClaimExtraction.model_validate(payload)


def test_summary_whitespace_is_stripped(payload: dict[str, Any]) -> None:
    payload["resume"] = "  Double débit du pass Navigo.  "

    claim = ClaimExtraction.model_validate(payload)

    assert claim.resume == "Double débit du pass Navigo."


def test_json_round_trip(payload: dict[str, Any]) -> None:
    # LLM output arrives as a raw JSON string, not a Python dict.
    claim = ClaimExtraction.model_validate_json(json.dumps(payload))

    assert claim.model_dump(mode="json") == payload


def test_json_schema_requires_every_field() -> None:
    # This schema will be sent to the LLM as the output contract (step 3).
    schema = ClaimExtraction.model_json_schema()

    assert set(schema["required"]) == set(ClaimExtraction.model_fields)
    assert schema["additionalProperties"] is False

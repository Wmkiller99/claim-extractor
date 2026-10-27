import json

import pandas as pd
import pytest
from fakes import FakeLLM

from claim_extractor.batch import extract_rows, guess_text_column, read_csv, to_csv
from claim_extractor.llm.base import LLMError
from claim_extractor.models import ClaimExtraction

VALID_REPLY = json.dumps(
    {
        "type_demande": "remboursement",
        "montant": 86.4,
        "date_incident": "2026-10-03",
        "urgence": "moyenne",
        "resume": "Le pass Navigo du client a été débité deux fois.",
        "confiance": 0.9,
    }
)


@pytest.mark.parametrize(
    "content",
    [
        "id,texte\n007,Débité deux fois\n008,\n".encode(),
        "id;texte\n007;Débité deux fois\n008;\n".encode(),
        "id;texte\n007;Débité deux fois\n008;\n".encode("utf-8-sig"),
        "id;texte\n007;Débité deux fois\n008;\n".encode("cp1252"),
    ],
    ids=["comma_utf8", "semicolon_utf8", "utf8_with_bom", "excel_cp1252"],
)
def test_read_csv_handles_separators_and_encodings(content: bytes) -> None:
    claims = read_csv(content)

    assert list(claims.columns) == ["id", "texte"]
    assert claims["id"].tolist() == ["007", "008"]  # read as text: leading zeros are kept
    assert claims["texte"].tolist() == ["Débité deux fois", ""]  # empty cell is "", not NaN


@pytest.mark.parametrize(
    ("columns", "expected_index"),
    [(["id", " Réclamation "], 1), (["texte", "date"], 0), (["id", "body"], 0)],
)
def test_guess_text_column(columns: list[str], expected_index: int) -> None:
    assert guess_text_column(columns) == expected_index


def test_each_claim_gets_a_row_with_a_status() -> None:
    llm = FakeLLM([VALID_REPLY, "pas du JSON", "toujours pas"])

    rows = list(extract_rows(["Débité deux fois", "   ", "Autre réclamation"], llm))

    assert [row["statut"] for row in rows] == ["ok", "texte_vide", "sortie_invalide"]
    assert rows[0]["montant"] == 86.4
    assert rows[2]["erreur"] != ""
    assert len(llm.calls) == 3  # the empty claim never reaches the LLM
    expected_columns = ["statut", "erreur", *ClaimExtraction.model_fields]
    assert all(list(row) == expected_columns for row in rows)  # same columns, same order


def test_backend_error_stops_the_batch() -> None:
    llm = FakeLLM([LLMError("connection refused")])

    with pytest.raises(LLMError):
        list(extract_rows(["Débité deux fois", "Autre réclamation"], llm))


def test_to_csv_is_excel_friendly_and_round_trips() -> None:
    claims = pd.DataFrame({"id": ["007"], "texte": ["Débité, deux fois ; vraiment"]})

    content = to_csv(claims)

    assert content.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM: Excel detects the encoding
    assert content.decode("utf-8-sig").splitlines()[0] == "id;texte"
    pd.testing.assert_frame_equal(read_csv(content), claims)

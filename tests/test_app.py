import json
from collections.abc import Callable
from pathlib import Path

import pytest
import streamlit as st
from fakes import FakeLLM
from streamlit.testing.v1 import AppTest

from claim_extractor import config
from claim_extractor.llm.base import LLMError

APP_PATH = str(Path(__file__).parents[1] / "app.py")
CLAIM = "Mon pass Navigo a été débité deux fois le 3 octobre, 86,40 €, je veux être remboursé"
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

StartApp = Callable[[FakeLLM], AppTest]


@pytest.fixture
def start_app(monkeypatch: pytest.MonkeyPatch) -> StartApp:
    """Run app.py headless, with the LLM backend replaced by a FakeLLM."""

    def start(llm: FakeLLM) -> AppTest:
        monkeypatch.setattr(config, "build_client", lambda settings: llm)
        st.cache_resource.clear()  # otherwise get_client() returns a previous test's LLM
        return AppTest.from_file(APP_PATH, default_timeout=30).run()

    return start


def extract_one(app: AppTest, text: str) -> AppTest:
    app.text_area(key="claim_text").input(text)
    return app.button(key="extract_one").click().run()


def test_app_starts_and_shows_the_model(start_app: StartApp) -> None:
    app = start_app(FakeLLM([]))

    assert not app.exception
    assert app.title[0].value == "claim-extractor"
    assert "fake-model" in app.sidebar.markdown[0].value


def test_single_claim_is_shown_as_json(start_app: StartApp) -> None:
    app = extract_one(start_app(FakeLLM([VALID_REPLY])), CLAIM)

    assert json.loads(app.json[0].value)["type_demande"] == "remboursement"


def test_empty_claim_shows_a_warning(start_app: StartApp) -> None:
    llm = FakeLLM([])

    app = extract_one(start_app(llm), "   ")

    assert "Colle d'abord" in app.warning[0].value
    assert llm.calls == []


def test_backend_error_is_shown(start_app: StartApp) -> None:
    app = extract_one(start_app(FakeLLM([LLMError("connection refused")])), CLAIM)

    assert "ne répond pas" in app.error[0].value


def test_invalid_output_shows_the_raw_reply(start_app: StartApp) -> None:
    app = extract_one(start_app(FakeLLM(["pas du JSON", "toujours pas"])), CLAIM)

    assert "sortie valide" in app.error[0].value
    assert app.code[0].value == "toujours pas"


def test_csv_batch_shows_results_and_a_download(start_app: StartApp) -> None:
    app = start_app(FakeLLM([VALID_REPLY]))
    csv_content = "id;texte\n007;Débité deux fois\n008;\n".encode()

    app.file_uploader[0].upload("claims.csv", csv_content, "text/csv").run()
    app.button(key="extract_csv").click().run()

    assert not app.exception
    assert "1 réclamation(s) extraite(s), 1 en erreur" in app.success[0].value
    assert app.dataframe[-1].value["statut"].tolist() == ["ok", "texte_vide"]
    assert len(app.get("download_button")) == 1


def test_csv_result_survives_the_next_rerun(start_app: StartApp) -> None:
    llm = FakeLLM([VALID_REPLY])
    app = start_app(llm)
    app.file_uploader[0].upload("claims.csv", b"texte\nDebit en double\n", "text/csv").run()
    app.button(key="extract_csv").click().run()

    app.run()  # any later interaction reruns the script, with the button back to False

    assert "1 réclamation(s) extraite(s)" in app.success[0].value
    assert len(llm.calls) == 1  # shown from the session, not extracted again

"""Streamlit interface: paste one claim, or upload a CSV of claims.

Run with: streamlit run app.py
"""

import logging

import pandas as pd
import streamlit as st

from claim_extractor.batch import extract_rows, guess_text_column, read_csv, to_csv
from claim_extractor.config import Settings, build_client
from claim_extractor.extractor import ExtractionError, extract_claim
from claim_extractor.llm.base import LLMClient, LLMError

EXAMPLE = "Mon pass Navigo a été débité deux fois le 3 octobre, 86,40 €, je veux être remboursé."


@st.cache_resource
def get_client() -> LLMClient:
    """Built once per server process, then reused by every rerun and every user."""
    settings = Settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    return build_client(settings)


def single_claim_tab(client: LLMClient) -> None:
    text = st.text_area("Réclamation", placeholder=EXAMPLE, height=150, key="claim_text")
    if not st.button("Extraire", type="primary", key="extract_one"):
        return
    if not text.strip():
        st.warning("Colle d'abord le texte d'une réclamation.")
        return

    try:
        with st.spinner("Analyse en cours..."):
            claim = extract_claim(text, client)
    except LLMError as exc:
        st.error(f"Le fournisseur LLM ne répond pas : {exc}")
    except ExtractionError as exc:
        st.error(f"Le modèle n'a pas produit de sortie valide, même après relance.\n\n{exc}")
        with st.expander("Dernière réponse brute du modèle"):
            st.code(exc.raw_output)
    else:
        st.json(claim.model_dump(mode="json"))


def csv_tab(client: LLMClient) -> None:
    uploaded = st.file_uploader("Fichier CSV, une réclamation par ligne", type="csv")
    if uploaded is None:
        return
    try:
        claims = read_csv(uploaded.getvalue())
    except ValueError as exc:  # also covers UnicodeDecodeError and pandas parsing errors
        st.error(f"Impossible de lire ce fichier CSV : {exc}")
        return

    columns = list(claims.columns)
    column = st.selectbox("Colonne contenant le texte", columns, index=guess_text_column(columns))
    st.dataframe(claims.head(), hide_index=True)

    # Streamlit reruns the whole script on every interaction: keep the result in the
    # session, tied to this file and column, so it survives the next rerun.
    batch_key = (uploaded.file_id, column)
    if st.button(f"Extraire les {len(claims)} réclamations", type="primary", key="extract_csv"):
        results = _run_batch(claims[column].tolist(), client)
        if results is not None:
            st.session_state["batch"] = (batch_key, pd.concat([claims, results], axis=1))

    stored = st.session_state.get("batch")
    if stored is None or stored[0] != batch_key:
        return
    result = stored[1]
    ok = int((result["statut"] == "ok").sum())
    st.success(f"{ok} réclamation(s) extraite(s), {len(result) - ok} en erreur.")
    st.dataframe(result, hide_index=True)
    st.download_button(
        "Télécharger le résultat (CSV)",
        data=to_csv(result),
        file_name="reclamations_extraites.csv",
        mime="text/csv",
        on_click="ignore",  # downloading must not rerun the script
    )


def _run_batch(texts: list[str], client: LLMClient) -> pd.DataFrame | None:
    progress = st.progress(0.0)
    rows = []
    try:
        for done, row in enumerate(extract_rows(texts, client), start=1):
            rows.append(row)
            progress.progress(done / len(texts), text=f"{done} / {len(texts)}")
    except LLMError as exc:
        st.error(f"Le fournisseur LLM ne répond pas, traitement interrompu : {exc}")
        return None
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="claim-extractor", page_icon="📨", layout="wide")
    st.title("claim-extractor")
    st.caption("Transforme des réclamations clients en données structurées, validées par Pydantic.")

    try:
        client = get_client()
    except ValueError as exc:  # missing API key, or invalid value in .env
        st.error(f"Configuration invalide : {exc}")
        st.stop()

    st.sidebar.markdown(f"**Modèle :** `{client.model}`")
    st.sidebar.caption("Le fournisseur et le modèle se configurent dans .env (voir .env.example).")

    one, many = st.tabs(["Une réclamation", "Fichier CSV"])
    with one:
        single_claim_tab(client)
    with many:
        csv_tab(client)


main()

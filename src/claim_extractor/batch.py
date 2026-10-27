"""Batch extraction over a CSV of claims."""

import io
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import pandas as pd

from claim_extractor.extractor import ExtractionError, extract_claim
from claim_extractor.llm.base import LLMClient
from claim_extractor.models import ClaimExtraction

TEXT_COLUMN_NAMES = ("texte", "text", "reclamation", "réclamation", "message")


def read_csv(data: bytes) -> pd.DataFrame:
    """Read a CSV with a , or ; separator, in UTF-8 or Excel's Windows-1252 encoding.

    Every cell is read as text, and an empty cell stays an empty string (not NaN).
    """
    try:
        text = data.decode("utf-8-sig")  # -sig also strips the BOM Excel adds
    except UnicodeDecodeError:
        text = data.decode("cp1252")
    header = text.split("\n", 1)[0]
    separator = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=separator, dtype=str, keep_default_na=False)


def guess_text_column(columns: Sequence[str]) -> int:
    """Index of the column most likely to hold the claim text, or 0."""
    for index, name in enumerate(columns):
        if name.strip().lower() in TEXT_COLUMN_NAMES:
            return index
    return 0


def extract_rows(texts: Iterable[str], client: LLMClient) -> Iterator[dict[str, Any]]:
    """Yield one flat result row per claim, in order.

    A bad claim becomes a row with a status, so one failure never loses the batch.
    An LLMError is not caught: if the backend is down, every row would fail anyway.
    """
    for text in texts:
        if not text.strip():
            yield _error_row("texte_vide", "")
            continue
        try:
            claim = extract_claim(text, client)
        except ExtractionError as exc:
            yield _error_row("sortie_invalide", str(exc))
        else:
            yield {"statut": "ok", "erreur": "", **claim.model_dump(mode="json")}


def to_csv(df: pd.DataFrame) -> bytes:
    """Semicolon-separated UTF-8 with a BOM, so that a French Excel opens it correctly."""
    return df.to_csv(index=False, sep=";").encode("utf-8-sig")


def _error_row(status: str, detail: str) -> dict[str, Any]:
    return {"statut": status, "erreur": detail, **dict.fromkeys(ClaimExtraction.model_fields)}

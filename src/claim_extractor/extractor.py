"""Extract a validated ClaimExtraction from free text, with one corrective retry."""

import json
import logging
import time
from datetime import date
from typing import Literal

from pydantic import ValidationError

from claim_extractor.llm.base import LLMClient, LLMError, Message
from claim_extractor.models import ClaimExtraction

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2  # first try + one retry with the validation errors

SYSTEM_PROMPT = """\
Tu analyses des réclamations envoyées par des clients à un service client.
Réponds uniquement avec un objet JSON conforme à ce schéma :
{schema}

Règles :
- type_demande : "remboursement" si le client demande à être remboursé ou dédommagé ;
  "perte" s'il a perdu un objet ; "information" pour une simple question ;
  "reclamation_service" pour une plainte sur le service (retard, accueil, propreté)
  sans demande de remboursement ; "autre" sinon.
- montant : montant en euros cité dans la réclamation (avec un point décimal), sinon null.
- date_incident : date de l'incident au format AAAA-MM-JJ, sinon null.
  Nous sommes le {today} : si l'année manque, prends la date passée la plus proche.
- urgence : "haute" si le client est bloqué, en danger, ou a perdu des papiers ou un moyen
  de paiement ; "moyenne" pour un préjudice financier ou un service dégradé ;
  "faible" pour une question ou une gêne mineure.
- resume : une seule phrase en français.
- confiance : ta certitude sur l'ensemble de l'extraction, entre 0 et 1.
"""

RETRY_PROMPT = """\
Ta réponse précédente n'est pas valide :
{errors}
Réponds de nouveau avec uniquement l'objet JSON corrigé.
"""


class ExtractionError(Exception):
    """The LLM output was still invalid after the retry."""

    def __init__(self, message: str, raw_output: str) -> None:
        super().__init__(message)
        self.raw_output = raw_output


def extract_claim(text: str, client: LLMClient, *, today: date | None = None) -> ClaimExtraction:
    """Extract structured data from one free-text claim.

    Raises:
        ValueError: if the claim text is empty.
        LLMError: if the LLM backend fails (not retried: it is not the model's fault).
        ExtractionError: if the output is still invalid after one retry.
    """
    if not text.strip():
        raise ValueError("Claim text is empty.")

    schema = ClaimExtraction.model_json_schema()
    system_prompt = SYSTEM_PROMPT.format(
        schema=json.dumps(schema, ensure_ascii=False),
        today=(today or date.today()).isoformat(),
    )
    messages: list[Message] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        start = time.perf_counter()
        try:
            raw = client.complete(messages, schema)
        except LLMError:
            _log_attempt(client, attempt, start, "llm_error")
            raise

        try:
            claim = ClaimExtraction.model_validate_json(raw)
        except ValidationError as exc:
            _log_attempt(client, attempt, start, "invalid_output")
            errors = _format_errors(exc)
            if attempt == MAX_ATTEMPTS:
                raise ExtractionError(
                    f"LLM output still invalid after {MAX_ATTEMPTS} attempts:\n{errors}",
                    raw_output=raw,
                ) from exc
            # Retry: show the model its own answer and what is wrong with it.
            messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {"role": "user", "content": RETRY_PROMPT.format(errors=errors)},
            ]
        else:
            _log_attempt(client, attempt, start, "success")
            return claim

    raise AssertionError("unreachable: the last attempt always returns or raises")


def _log_attempt(
    client: LLMClient,
    attempt: int,
    start: float,
    outcome: Literal["success", "invalid_output", "llm_error"],
) -> None:
    duration_ms = (time.perf_counter() - start) * 1000
    level = logging.INFO if outcome == "success" else logging.WARNING
    logger.log(
        level,
        "llm_call model=%s attempt=%d duration_ms=%.0f outcome=%s",
        client.model,
        attempt,
        duration_ms,
        outcome,
    )


def _format_errors(exc: ValidationError) -> str:
    """One line per error, without Pydantic's documentation URLs."""
    lines: list[str] = []
    for error in exc.errors(include_url=False):
        field = ".".join(str(part) for part in error["loc"]) or "json"
        lines.append(f"- {field} : {error['msg']}")
    return "\n".join(lines)

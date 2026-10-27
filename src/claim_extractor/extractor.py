"""Extract a validated ClaimExtraction from free text, with one corrective retry."""

import json
import logging
import time
from datetime import date, timedelta
from typing import Literal

from pydantic import ValidationError

from claim_extractor.llm.base import LLMClient, LLMError, Message
from claim_extractor.models import ClaimExtraction

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2  # first try + one retry with the validation errors

WEEKDAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")

SYSTEM_PROMPT = """\
Tu analyses des réclamations envoyées par des clients à un service client de transport.
Réponds uniquement avec un objet JSON conforme à ce schéma :
{schema}

type_demande : applique la première règle qui correspond.
1. "perte" : le client a perdu ou oublié un objet, même s'il pose une question ou demande
   un remplacement.
2. "remboursement" : le client demande à être remboursé ou dédommagé.
3. "information" : le client pose une question, sans se plaindre.
4. "reclamation_service" : le client se plaint du service (retard, panne, accueil, sécurité,
   propreté) sans demander de remboursement.
5. "autre" : tout le reste (félicitations, suggestion, données personnelles...).

urgence :
- "haute" seulement si le client est bloqué (ne peut plus voyager ou travailler), en danger,
  ou a perdu des papiers d'identité ou un moyen de paiement. Un ton pressant ne suffit pas.
- "moyenne" pour une perte d'argent ou un service dégradé (débit, amende, retard, panne).
- "faible" dans les autres cas (question, objet sans valeur, gêne mineure, remerciements...).

montant : montant en euros cité dans la réclamation, en chiffres avec un point décimal
(« quinze euros » donne 15.0), sinon null. Si plusieurs montants sont cités, prends celui
que le client réclame.

date_incident : jour de l'incident au format AAAA-MM-JJ. Utilise ces repères :
{date_hints}
Mets null si aucun jour précis n'est donné (« depuis juillet ») ou si la date est future.

resume : une seule phrase en français.
confiance : ta certitude sur l'ensemble de l'extraction, entre 0 et 1.
"""

# Worked examples sent before the claim, one per category. They are deliberately NOT taken
# from the evaluation set; building them as ClaimExtraction validates them at import time.
FEW_SHOT_EXAMPLES: list[tuple[str, ClaimExtraction]] = [
    (
        "J'ai oublié mon écharpe grise dans le bus 27. Est-ce que quelqu'un l'a rapportée ?",
        ClaimExtraction(
            type_demande="perte",
            montant=None,
            date_incident=None,
            urgence="faible",
            resume="Le client a oublié son écharpe dans le bus 27 et demande si on l'a retrouvée.",
            confiance=0.9,
        ),
    ),
    (
        "Le 5 mars 2026, l'automate m'a facturé quinze euros mais le ticket n'est jamais sorti. "
        "Je veux être remboursé.",
        ClaimExtraction(
            type_demande="remboursement",
            montant=15.0,
            date_incident=date(2026, 3, 5),
            urgence="moyenne",
            resume="Le client a payé 15 € pour un ticket non délivré et demande un remboursement.",
            confiance=0.95,
        ),
    ),
    (
        "Les vélos sont-ils autorisés dans le tramway le dimanche ?",
        ClaimExtraction(
            type_demande="information",
            montant=None,
            date_incident=None,
            urgence="faible",
            resume="Le client demande si les vélos sont autorisés dans le tramway le dimanche.",
            confiance=0.95,
        ),
    ),
    (
        "Ma carte Navigo ne passe plus aux portiques et je ne peux plus aller travailler !",
        ClaimExtraction(
            type_demande="reclamation_service",
            montant=None,
            date_incident=None,
            urgence="haute",
            resume="La carte Navigo du client ne passe plus, il ne peut plus aller travailler.",
            confiance=0.85,
        ),
    ),
    (
        "Je propose d'installer des prises USB dans les rames, ce serait bien pratique.",
        ClaimExtraction(
            type_demande="autre",
            montant=None,
            date_incident=None,
            urgence="faible",
            resume="Le client suggère d'installer des prises USB dans les rames.",
            confiance=0.9,
        ),
    ),
]

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
        date_hints=build_date_hints(today or date.today()),
    )
    messages: list[Message] = [{"role": "system", "content": system_prompt}]
    for example_text, example_output in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example_text})
        messages.append({"role": "assistant", "content": example_output.model_dump_json()})
    messages.append({"role": "user", "content": text})

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


def build_date_hints(today: date) -> str:
    """A small pre-computed calendar: the model looks dates up instead of computing them."""
    lines = [
        f"- aujourd'hui (ce matin, ce soir) : {_day_label(today)}",
        f"- hier : {_day_label(today - timedelta(days=1))}",
        f"- avant-hier : {_day_label(today - timedelta(days=2))}",
    ]
    for weekday, name in enumerate(WEEKDAYS):
        days_back = (today.weekday() - weekday) % 7 or 7  # "lundi dernier" on a Monday: 7 days
        lines.append(f"- {name} dernier : {(today - timedelta(days=days_back)).isoformat()}")
    lines.append(
        f"- une date sans année est en {today.year} si elle tombe au plus tard aujourd'hui, "
        f"sinon en {today.year - 1}"
    )
    return "\n".join(lines)


def _day_label(day: date) -> str:
    return f"{WEEKDAYS[day.weekday()]} {day.isoformat()}"


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

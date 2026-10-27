"""Score claim extraction against a labelled set of claims."""

import math
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from claim_extractor.extractor import ExtractionError, extract_claim
from claim_extractor.llm.base import LLMClient
from claim_extractor.models import ClaimExtraction, RequestType, Urgency

# resume and confiance are free-form, so they have no single right answer to compare with.
EVALUATED_FIELDS = ("type_demande", "montant", "date_incident", "urgence")
AMOUNT_TOLERANCE = 0.005  # amounts match to within half a cent


class ExpectedFields(BaseModel):
    """The labelled answer for the fields that can be compared objectively."""

    model_config = ConfigDict(extra="forbid")

    type_demande: RequestType
    montant: float | None
    date_incident: date | None
    urgence: Urgency


class EvalCase(BaseModel):
    """One line of the evaluation set."""

    model_config = ConfigDict(extra="forbid")

    id: str
    received_on: date  # used as "today", so relative dates ("hier") have a fixed answer
    text: str
    expected: ExpectedFields
    note: str | None = None  # why a tricky case is labelled the way it is


@dataclass(frozen=True)
class CaseResult:
    case: EvalCase
    output: ClaimExtraction | None  # None: output still invalid after the retry
    matches: dict[str, bool]
    duration_s: float


@dataclass(frozen=True)
class Summary:
    total: int
    valid_json_rate: float
    accuracy: dict[str, float]
    mean_duration_s: float


def load_cases(path: Path) -> list[EvalCase]:
    """Read and validate a JSONL evaluation set, pointing at the faulty line if any."""
    cases: list[EvalCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(EvalCase.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(f"{path}, line {line_number}: {exc}") from exc
    if not cases:
        raise ValueError(f"{path} contains no cases.")
    return cases


def compare(expected: ExpectedFields, output: ClaimExtraction | None) -> dict[str, bool]:
    """Field-by-field match. An invalid output counts as wrong on every field."""
    if output is None:
        return dict.fromkeys(EVALUATED_FIELDS, False)
    return {
        "type_demande": output.type_demande == expected.type_demande,
        "montant": _same_amount(expected.montant, output.montant),
        "date_incident": output.date_incident == expected.date_incident,
        "urgence": output.urgence == expected.urgence,
    }


def run_evaluation(cases: list[EvalCase], client: LLMClient) -> list[CaseResult]:
    """Extract every case. An LLMError is not caught: a broken backend stops the run."""
    results: list[CaseResult] = []
    for case in cases:
        start = time.perf_counter()
        output: ClaimExtraction | None
        try:
            output = extract_claim(case.text, client, today=case.received_on)
        except ExtractionError:
            output = None
        duration_s = time.perf_counter() - start
        results.append(CaseResult(case, output, compare(case.expected, output), duration_s))
    return results


def summarize(results: list[CaseResult]) -> Summary:
    if not results:
        raise ValueError("No results to summarize.")
    total = len(results)
    return Summary(
        total=total,
        valid_json_rate=sum(r.output is not None for r in results) / total,
        accuracy={f: sum(r.matches[f] for r in results) / total for f in EVALUATED_FIELDS},
        mean_duration_s=sum(r.duration_s for r in results) / total,
    )


def format_report(results: list[CaseResult], model: str) -> str:
    """A summary table followed by every mismatch, for the prompt-tuning loop."""
    summary = summarize(results)
    scores = [("Valid JSON", summary.valid_json_rate)]
    scores += [(field, summary.accuracy[field]) for field in EVALUATED_FIELDS]

    lines = [
        f"Evaluation of {summary.total} claims with model {model}",
        "",
        f"{'Metric':<15} {'Score':>7}",
        f"{'-' * 15} {'-' * 7}",
    ]
    lines += [f"{name:<15} {score:>7.1%}" for name, score in scores]
    lines.append(f"{'Mean duration':<15} {summary.mean_duration_s:>6.1f}s")

    mismatches = _format_mismatches(results)
    if mismatches:
        lines += ["", "Mismatches:", *mismatches]
    return "\n".join(lines)


def _same_amount(expected: float | None, actual: float | None) -> bool:
    if expected is None or actual is None:
        return expected is None and actual is None
    return math.isclose(expected, actual, abs_tol=AMOUNT_TOLERANCE)


def _format_mismatches(results: list[CaseResult]) -> list[str]:
    lines: list[str] = []
    for result in results:
        if result.output is None:
            lines.append(f"  {result.case.id}  invalid output after retry")
            continue
        for field in EVALUATED_FIELDS:
            if not result.matches[field]:
                expected = getattr(result.case.expected, field)
                actual = getattr(result.output, field)
                lines.append(f"  {result.case.id}  {field:<13} expected={expected}  got={actual}")
    return lines

"""Structured output of a claim extraction, validated by Pydantic."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RequestType = Literal["remboursement", "perte", "information", "reclamation_service", "autre"]
Urgency = Literal["faible", "moyenne", "haute"]


class ClaimExtraction(BaseModel):
    """Structured data extracted from one free-text customer claim.

    Every field is required, including the nullable ones: the LLM must write
    null explicitly instead of omitting a key, so a forgotten field is an error.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    type_demande: RequestType = Field(description="Category of the customer's request.")
    montant: float | None = Field(
        ge=0, description="Amount in euros mentioned in the claim, or null if none."
    )
    date_incident: date | None = Field(
        description="Date of the incident in ISO 8601 format (YYYY-MM-DD), or null if unknown."
    )
    urgence: Urgency = Field(description="How urgently the claim should be handled.")
    resume: str = Field(
        min_length=1, max_length=300, description="One-sentence summary of the claim, in French."
    )
    confiance: float = Field(
        ge=0, le=1, description="Confidence in this extraction, from 0 (none) to 1 (certain)."
    )

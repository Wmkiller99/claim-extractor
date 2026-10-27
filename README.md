# claim-extractor

Turn free-text customer claims into structured, validated data with an LLM.

> "Mon pass Navigo a été débité deux fois le 3 octobre, 86,40 €, je veux être remboursé"

becomes a JSON object (request type, amount, incident date, urgency, summary, confidence)
validated by Pydantic and ready for a customer-service workflow.

**Status:** work in progress.

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
cp .env.example .env             # then fill in your values
```

## Development

```bash
ruff check .        # lint
ruff format .       # format
mypy                # type-check
pytest              # tests
```

## Project layout

```
src/claim_extractor/   application code
tests/                 pytest test suite
.env.example           configuration template (copy to .env)
```

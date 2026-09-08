# Contributing to VeritasAI

Thanks for helping improve VeritasAI, the evidence-based misinformation analysis platform.

## Local setup

1. Create and activate a Python virtual environment.
2. Install the dependencies from `requirements.txt`.
3. Copy `.env.example` to `.env` and keep credentials local.
4. Install the spaCy model required by the NLP pipeline.
5. Run the backend and Streamlit frontend separately as described in the README.

## Contribution guidelines

- Keep changes focused on one agent, service, or user workflow.
- Add or update tests when changing claim extraction, evidence handling, verdicts, or API behavior.
- Do not commit API keys, real user claims, private source data, or generated cache files.
- Treat model output as evidence to review, not as an authority by itself.
- Explain data sources, uncertainty, and failure behavior in pull requests.
- Include a short reproduction or screenshot for user-facing changes.

## Pull request checklist

- [ ] The README or relevant documentation is updated.
- [ ] Tests or validation commands have been run.
- [ ] Secrets and local environment files are excluded.
- [ ] Claims about accuracy or confidence are backed by a reproducible check.

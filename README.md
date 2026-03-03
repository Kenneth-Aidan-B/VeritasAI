# VeritasAI — Intelligent Real-Time Misinformation Analysis & Correction Platform

## Overview
VeritasAI is a multi-agent, LangGraph-orchestrated platform that ingests text/URLs, decomposes claims into atomic units, cross-references evidence from multiple sources (Tavily, Google Fact Check API, Wikipedia), classifies veracity with calibrated confidence, and generates transparent reasoning chains + evidence-based corrections — all streamed in real-time to an interactive Streamlit dashboard.

## Project Structure
```
VeritasAI/
├── backend/
│   ├── agents/
│   │   ├── claim_extractor.py      # Claim decomposition & NER
│   │   ├── evidence_retriever.py   # Multi-source evidence gathering
│   │   ├── veracity_classifier.py  # 5-level verdict classification
│   │   ├── explanation_generator.py# Transparent reasoning chains
│   │   ├── correction_generator.py # Factual corrections & bias analysis
│   │   └── debate_agents.py        # Multi-agent debate feature
│   ├── graph/
│   │   └── orchestrator.py         # LangGraph workflow orchestrator
│   ├── services/
│   │   ├── tavily_service.py       # Tavily Search API
│   │   ├── factcheck_service.py    # Google Fact Check Tools API
│   │   ├── wikipedia_service.py    # Wikipedia API
│   │   └── cache_service.py        # ChromaDB caching layer
│   ├── models/
│   │   └── schemas.py              # Pydantic models
│   ├── config.py                   # Settings & configuration
│   ├── main.py                     # FastAPI application entry point
│   └── __init__.py
├── frontend/
│   └── app.py                      # Streamlit dashboard
├── .env.example                    # Environment variable template
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Clone & Install
```bash
cd VeritasAI
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run Backend
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Run Frontend
```bash
streamlit run frontend/app.py
```

## API Keys Required (All Free Tiers)
| API | Free Tier | Sign-up |
|-----|-----------|---------|
| OpenRouter | Free models available | https://openrouter.ai |
| Tavily | 1000 searches/month | https://tavily.com |
| Google Fact Check | Free (Google Cloud) | https://console.cloud.google.com |

## Tech Stack
- **LLM**: OpenRouter (Qwen3 235B Thinking) — Free reasoning model
- **Orchestration**: LangGraph — Stateful graph workflows
- **Backend**: FastAPI + WebSockets — Real-time streaming
- **Evidence**: Tavily + Google Fact Check + Wikipedia
- **Vector DB**: ChromaDB — Claim caching
- **NLP**: spaCy — NER & sentence segmentation
- **Frontend**: Streamlit — Interactive dashboard

## Key Features
- Multi-agent LangGraph pipeline with 5 specialized nodes
- 3 sources cross-referenced with agreement/disagreement matrix
- 5-level veracity scale with calibrated confidence %
- Transparent reasoning chain with inline citations
- Detection + structured corrections + bias analysis
- Real-time WebSocket streaming with live progress
- Multi-Agent Debate for controversial claims
- Responsible AI card with assumptions & limitations

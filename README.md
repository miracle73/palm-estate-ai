# 🌴 Palm Oil Estate Intelligence Platform

AI-powered multi-agent system for palm oil estate monitoring, analytics, and forecasting. Built with **OpenAI Agents SDK** + **OpenRouter** + **FastAPI**.

## Architecture

Five AI agents run in sequence every 30 seconds:

1. **Data Collection Agent** — Ingests and validates estate data
2. **Unification Agent** — Normalises and standardises across estates
3. **Analytics & P&L Agent** — Calculates revenue, cost, margin, breakdowns
4. **Forecasting Agent** — 30/60/90 day projections with LLM-generated summaries
5. **Alert Agent** — Threshold-based colour-coded alerts (red/amber/green)

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your OpenRouter API key
```bash
export API_TOKEN="your-openrouter-api-key"
```

### 3. Generate simulated data
```bash
python generate_data.py
```

### 4. Start the server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open the dashboard
Navigate to `http://localhost:8000` in your browser.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/pipeline-status` | GET | Agent run status and logs |
| `/api/metrics` | GET | P&L metrics from Analytics Agent |
| `/api/forecast` | GET | 30/60/90 day projections |
| `/api/alerts` | GET | Colour-coded alerts |
| `/api/estates` | GET | Per-estate performance summary |
| `/api/ingest` | POST | Manually trigger pipeline run |

## Tech Stack

- **Backend**: Python, FastAPI, OpenAI Agents SDK
- **LLM**: OpenRouter (openai/gpt-4o)
- **Frontend**: HTML, Chart.js
- **Data**: Simulated JSON (3 estates × 3 blocs × 180 days)

## Simulated Data

- 3 estates, each with 3 blocs
- 180 days of daily records
- Estate 2 Bloc 3 deliberately underperforms to demonstrate alerts
- Realistic NGN values (revenue ₦1.5M–3M/day/estate)
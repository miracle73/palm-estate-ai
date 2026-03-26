"""Palm Oil Estate Intelligence Agents — OpenAI Agents SDK + OpenRouter."""
import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from agents import Agent, Runner, function_tool, set_default_openai_client, ModelSettings
from openai import AsyncOpenAI

# ── OpenRouter client ──────────────────────────────────────────────
API_KEY = os.environ.get("API_TOKEN", "")
MODEL = "openai/gpt-4o"
DATA_FILE = "estate_data.json"

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('API_TOKEN'),
)
set_default_openai_client(client)

# ── Shared state (served by FastAPI) ──────────────────────────────
pipeline_state: dict[str, Any] = {
    "status": "idle",
    "last_run": None,
    "logs": [],
    "raw_data": [],
    "raw_columns": [],
    "unified_data": [],
    "cost_categories": [],
    "metrics": {},
    "forecast": {},
    "alerts": [],
    "estate_summary": [],
}

def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    pipeline_state["logs"].append(entry)
    # Keep last 100 log lines
    if len(pipeline_state["logs"]) > 100:
        pipeline_state["logs"] = pipeline_state["logs"][-100:]
    print(entry)


# ═══════════════════════════════════════════════════════════════════
# TOOL FUNCTIONS  (called by agents via function_tool)
# ═══════════════════════════════════════════════════════════════════

@function_tool
def ingest_estate_data() -> str:
    """Read and validate estate operations data from uploaded JSON/CSV file."""
    try:
        with open(DATA_FILE) as f:
            data = json.load(f)
    except FileNotFoundError:
        return json.dumps({"error": "No data file found. Upload a JSON or CSV file first."})

    if not data or not isinstance(data, list):
        return json.dumps({"error": "Data file is empty or invalid format."})

    # Auto-detect columns — report what we found
    sample = data[0]
    columns = list(sample.keys())

    # Basic validation: drop rows with all-None values
    clean = [r for r in data if any(v is not None and v != "" for v in r.values())]

    pipeline_state["raw_data"] = clean
    pipeline_state["raw_columns"] = columns
    _log(f"Data Collection Agent: {len(clean)} records ingested. Columns: {columns}")
    return json.dumps({
        "records_ingested": len(clean),
        "columns_found": columns,
        "sample_rows": clean[:3],
    })


@function_tool
async def unify_data() -> str:
    """Use LLM to intelligently map any column names to standard fields, then unify data."""
    raw = pipeline_state["raw_data"]
    if not raw:
        return json.dumps({"error": "No raw data available"})

    columns = list(raw[0].keys())
    sample_rows = raw[:3]

    # ── LLM-powered column mapping ──────────────────────────────────
    mapping_prompt = f"""You are a data engineer. I have a dataset with these columns:
{json.dumps(columns)}

Here are 3 sample rows:
{json.dumps(sample_rows, indent=2)}

Map each column to ONE of these standard categories. Return ONLY a JSON object, no markdown, no explanation:
{{
  "date": "column_name_or_null",
  "estate": "column_name_or_null",
  "bloc": "column_name_or_null",
  "yield": "column_name_or_null",
  "revenue": "column_name_or_null",
  "costs": ["list_of_cost_column_names"],
  "supply_chain": ["list_of_supply_chain_column_names"],
  "marketing": ["list_of_marketing_column_names"],
  "other_numeric": ["list_of_other_numeric_column_names"]
}}

Rules:
- "date" = any date/time/period column
- "estate" = farm/plantation/site/location/estate name
- "bloc" = block/section/zone/area/field within an estate
- "yield" = production output in tonnes/kg
- "revenue" = sales/income/turnover/gross receipts
- "costs" = ALL expense/cost/spend/wages/salary/overhead columns
- "supply_chain" = logistics/transport/shipping/delivery/freight/warehouse/storage columns
- "marketing" = advertising/marketing/promo/branding/campaign columns
- "other_numeric" = any remaining numeric columns not matched above
- If a column doesn't match any category, put null
- supply_chain and marketing columns should ALSO appear in costs if they represent expenses"""

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": mapping_prompt}],
            temperature=0,
        )
        mapping_text = response.choices[0].message.content.strip()
        # Clean markdown fences if present
        mapping_text = mapping_text.replace("```json", "").replace("```", "").strip()
        col_map = json.loads(mapping_text)
        _log(f"Unification Agent: LLM mapped columns → {json.dumps(col_map)}")
    except Exception as e:
        _log(f"Unification Agent: LLM mapping failed ({e}), falling back to keyword matching.")
        col_map = _fallback_column_map(columns, raw)

    # ── Apply mapping to build unified dataset ──────────────────────
    col_date = col_map.get("date")
    col_estate = col_map.get("estate")
    col_bloc = col_map.get("bloc")
    col_yield = col_map.get("yield")
    col_revenue = col_map.get("revenue")
    cost_cols = col_map.get("costs", []) or []
    supply_cols = col_map.get("supply_chain", []) or []
    marketing_cols = col_map.get("marketing", []) or []
    other_cols = col_map.get("other_numeric", []) or []

    # Merge all expense columns (deduplicated)
    all_cost_cols = list(dict.fromkeys(cost_cols + supply_cols + marketing_cols))

    def safe_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    unified = []
    for r in raw:
        cost_breakdown = {}
        total_cost = 0
        for cc in all_cost_cols:
            if cc and cc in r:
                label = cc.lower().replace("_ngn", "").replace("_cost", "").replace("cost_", "")
                label = label.replace("_usd", "").replace("_", " ").strip()
                val = safe_float(r.get(cc, 0))
                cost_breakdown[label] = val
                total_cost += val

        revenue = safe_float(r.get(col_revenue, 0)) if col_revenue else 0

        unified.append({
            "date": r.get(col_date, "unknown") if col_date else "unknown",
            "estate": r.get(col_estate, "Estate 1") if col_estate else "Estate 1",
            "bloc": r.get(col_bloc, "Bloc 1") if col_bloc else "Bloc 1",
            "yield_tonnes": safe_float(r.get(col_yield, 0)) if col_yield else 0,
            "cost_breakdown": cost_breakdown,
            "total_cost": round(total_cost, 2),
            "revenue": round(revenue, 2),
            "currency": "NGN",
        })

    pipeline_state["unified_data"] = unified
    pipeline_state["cost_categories"] = list(set(
        k for r in unified for k in r.get("cost_breakdown", {}).keys()
    ))
    estates = set(r["estate"] for r in unified)

    _log(f"Unification Agent: Data standardised across {len(estates)} estates. {len(all_cost_cols)} cost categories detected.")
    return json.dumps({"records_unified": len(unified), "estates": list(estates), "column_mapping": col_map})


def _fallback_column_map(columns, raw):
    """Keyword-based fallback if LLM mapping fails."""
    def find_col(keywords, cols):
        for kw in keywords:
            for c in cols:
                if kw in c.lower().replace(" ", "_").replace("-", "_"):
                    return c
        return None

    cost_keywords = ["cost", "expense", "spend", "labour", "labor", "logistics",
                     "transport", "fertiliser", "fertilizer", "processing",
                     "maintenance", "overhead", "wages", "salary", "freight",
                     "marketing", "advertising", "promo", "shipping", "storage"]
    cost_cols = [c for c in columns
                 if any(kw in c.lower().replace(" ", "_") for kw in cost_keywords)]

    return {
        "date": find_col(["date", "day", "period", "time"], columns),
        "estate": find_col(["estate", "farm", "plantation", "site", "location"], columns),
        "bloc": find_col(["bloc", "block", "section", "zone", "area", "field"], columns),
        "yield": find_col(["yield", "production", "output", "harvest", "tonnes"], columns),
        "revenue": find_col(["revenue", "sales", "income", "turnover"], columns),
        "costs": cost_cols,
        "supply_chain": [],
        "marketing": [],
        "other_numeric": [],
    }


@function_tool
def compute_analytics() -> str:
    """Calculate P&L metrics and cost breakdown from unified data."""
    data = pipeline_state["unified_data"]
    if not data:
        return json.dumps({"error": "No unified data available"})

    total_revenue = sum(r["revenue"] for r in data)
    total_cost = sum(r["total_cost"] for r in data)
    gross_profit = total_revenue - total_cost
    margin = round((gross_profit / total_revenue) * 100, 2) if total_revenue else 0

    # Dynamic cost breakdown from whatever categories exist
    from collections import defaultdict
    cost_totals = defaultdict(float)
    for r in data:
        for cat, val in r.get("cost_breakdown", {}).items():
            cost_totals[cat] += val

    # Monthly trend (last 6 months)
    monthly_rev = defaultdict(float)
    monthly_cost = defaultdict(float)
    for r in data:
        month_key = str(r["date"])[:7]  # YYYY-MM
        monthly_rev[month_key] += r["revenue"]
        monthly_cost[month_key] += r["total_cost"]

    months_sorted = sorted(monthly_rev.keys())
    trend = [{"month": m, "revenue": round(monthly_rev[m], 2), "cost": round(monthly_cost[m], 2)}
             for m in months_sorted]

    # Per-estate summary
    estate_data = defaultdict(lambda: {"revenue": 0, "cost": 0, "yield": 0, "records": 0})
    for r in data:
        e = estate_data[r["estate"]]
        e["revenue"] += r["revenue"]
        e["cost"] += r["total_cost"]
        e["yield"] += r["yield_tonnes"]
        e["records"] += 1

    estate_summary = []
    for name, d in estate_data.items():
        profit = d["revenue"] - d["cost"]
        est_margin = round((profit / d["revenue"]) * 100, 2) if d["revenue"] else 0
        estate_summary.append({
            "estate": name,
            "revenue": round(d["revenue"], 2),
            "cost": round(d["cost"], 2),
            "profit": round(profit, 2),
            "margin": est_margin,
            "total_yield": round(d["yield"], 2),
            "cost_efficiency": round((d["cost"] / d["revenue"]) * 100, 2) if d["revenue"] else 0,
        })

    metrics = {
        "total_revenue": round(total_revenue, 2),
        "total_cost": round(total_cost, 2),
        "gross_profit": round(gross_profit, 2),
        "margin_pct": margin,
        "cost_breakdown": {k: round(v, 2) for k, v in cost_totals.items()},
        "monthly_trend": trend,
    }

    pipeline_state["metrics"] = metrics
    pipeline_state["estate_summary"] = estate_summary
    _log(f"Analytics Agent: P&L calculated. Margin: {margin}%.")
    return json.dumps({"margin": margin, "gross_profit": round(gross_profit, 2)})


@function_tool
def compute_forecast() -> str:
    """Generate 30/60/90 day projections from monthly trend data."""
    metrics = pipeline_state.get("metrics", {})
    trend = metrics.get("monthly_trend", [])
    if len(trend) < 2:
        return json.dumps({"error": "Need at least 2 months of trend data"})

    # Simple linear extrapolation
    revenues = [m["revenue"] for m in trend]
    costs = [m["cost"] for m in trend]

    avg_monthly_rev = sum(revenues) / len(revenues)
    avg_monthly_cost = sum(costs) / len(costs)

    # Trend slope
    n = len(revenues)
    rev_slope = (revenues[-1] - revenues[0]) / max(n - 1, 1)
    cost_slope = (costs[-1] - costs[0]) / max(n - 1, 1)

    projections = {}
    for days, label in [(30, "30_day"), (60, "60_day"), (90, "90_day")]:
        months_ahead = days / 30
        proj_rev = revenues[-1] + rev_slope * months_ahead
        proj_cost = costs[-1] + cost_slope * months_ahead
        projections[label] = {
            "projected_revenue": round(proj_rev, 2),
            "projected_cost": round(proj_cost, 2),
            "projected_profit": round(proj_rev - proj_cost, 2),
        }

    # Total yield from unified data for yield projection
    data = pipeline_state["unified_data"]
    from collections import defaultdict
    monthly_yield = defaultdict(float)
    for r in data:
        monthly_yield[r["date"][:7]] += r["yield_tonnes"]
    yields = [monthly_yield[m] for m in sorted(monthly_yield.keys())]
    if len(yields) >= 2:
        yield_slope = (yields[-1] - yields[0]) / max(len(yields) - 1, 1)
        for days, label in [(30, "30_day"), (60, "60_day"), (90, "90_day")]:
            projections[label]["projected_yield"] = round(yields[-1] + yield_slope * (days / 30), 2)

    pipeline_state["forecast"] = projections
    _log("Forecasting Agent: Projections generated for 30/60/90 days.")
    return json.dumps(projections)


@function_tool
def evaluate_alerts() -> str:
    """Check metrics against thresholds and generate alerts."""
    metrics = pipeline_state.get("metrics", {})
    data = pipeline_state.get("unified_data", [])
    alerts = []

    if not metrics or not data:
        return json.dumps({"error": "No metrics or data available"})

    # Threshold 1: cost above 80% of revenue
    cost_ratio = (metrics["total_cost"] / metrics["total_revenue"] * 100) if metrics["total_revenue"] else 0
    if cost_ratio > 80:
        alerts.append({"level": "red", "message": f"Total cost is {cost_ratio:.1f}% of revenue — exceeds 80% threshold"})
    elif cost_ratio > 70:
        alerts.append({"level": "amber", "message": f"Total cost is {cost_ratio:.1f}% of revenue — approaching 80% threshold"})
    else:
        alerts.append({"level": "green", "message": f"Cost-to-revenue ratio healthy at {cost_ratio:.1f}%"})

    # Threshold 2: yield per bloc — check for underperformers (>10% below average)
    from collections import defaultdict
    bloc_yields = defaultdict(float)
    bloc_counts = defaultdict(int)
    for r in data:
        key = f"{r['estate']} - {r['bloc']}"
        bloc_yields[key] += r["yield_tonnes"]
        bloc_counts[key] += 1

    avg_yields = {k: bloc_yields[k] / bloc_counts[k] for k in bloc_yields}
    overall_avg = sum(avg_yields.values()) / len(avg_yields) if avg_yields else 0

    target_yield = overall_avg  # Using average as target
    for bloc_name, avg_y in avg_yields.items():
        pct_of_target = (avg_y / target_yield * 100) if target_yield else 0
        if pct_of_target < 85:
            alerts.append({"level": "red", "message": f"{bloc_name}: yield at {pct_of_target:.0f}% of target — below 85% threshold"})
        elif pct_of_target < 95:
            alerts.append({"level": "amber", "message": f"{bloc_name}: yield at {pct_of_target:.0f}% of target — watch closely"})

    # Threshold 3: per-estate underperformance >10%
    estate_rev = defaultdict(float)
    estate_count = defaultdict(int)
    for r in data:
        estate_rev[r["estate"]] += r["revenue"]
        estate_count[r["estate"]] += 1

    avg_estate_rev = sum(estate_rev.values()) / len(estate_rev) if estate_rev else 0
    for estate, rev in estate_rev.items():
        if rev < avg_estate_rev * 0.90:
            alerts.append({"level": "amber", "message": f"{estate} revenue {((rev/avg_estate_rev)-1)*100:.1f}% vs average"})

    if not any(a["level"] in ("red", "amber") for a in alerts):
        alerts.append({"level": "green", "message": "All estates operating within normal parameters"})

    pipeline_state["alerts"] = alerts
    _log(f"Alert Agent: {len(alerts)} alerts generated.")
    return json.dumps({"alert_count": len(alerts), "alerts": alerts})


# ═══════════════════════════════════════════════════════════════════
# AGENT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════

data_collection_agent = Agent(
    name="Data Collection Agent",
    instructions="You are the data collection agent. Call ingest_estate_data to read estate data. Report how many records were ingested. Be brief.",
    tools=[ingest_estate_data],
    model=MODEL,
)

unification_agent = Agent(
    name="Unification Agent",
    instructions="You are the data unification agent. Call unify_data to normalise the estate data. Report how many estates were standardised. Be brief.",
    tools=[unify_data],
    model=MODEL,
)

analytics_agent = Agent(
    name="Analytics & P&L Agent",
    instructions="You are the analytics agent. Call compute_analytics to calculate P&L metrics. Report the profit margin. Be brief.",
    tools=[compute_analytics],
    model=MODEL,
)

forecasting_agent = Agent(
    name="Forecasting Agent",
    instructions="You are the forecasting agent. Call compute_forecast to generate 30/60/90 day projections. Provide a 2-sentence natural language summary of the forecast outlook. Be brief.",
    tools=[compute_forecast],
    model=MODEL,
)

alert_agent = Agent(
    name="Alert Agent",
    instructions="You are the alert agent. Call evaluate_alerts to check all metrics against thresholds. Report how many alerts were generated and highlight any red alerts. Be brief.",
    tools=[evaluate_alerts],
    model=MODEL,
)


# ═══════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════

async def run_pipeline():
    """Run all 5 agents sequentially."""
    pipeline_state["status"] = "running"
    _log("Orchestrator: Pipeline started.")

    agents_sequence = [
        (data_collection_agent, "Ingest estate data now."),
        (unification_agent, "Unify and normalise the ingested data now."),
        (analytics_agent, "Compute P&L analytics now."),
        (forecasting_agent, "Generate 30/60/90 day forecasts now."),
        (alert_agent, "Evaluate all alerts now."),
    ]

    for agent, prompt in agents_sequence:
        try:
            result = await Runner.run(agent, prompt)
            _log(f"  → {agent.name} completed.")
            # Store forecast summary if it's the forecasting agent
            if agent.name == "Forecasting Agent" and result.final_output:
                pipeline_state["forecast"]["summary"] = result.final_output
        except Exception as e:
            _log(f"  ✗ {agent.name} error: {str(e)[:120]}")

    pipeline_state["status"] = "idle"
    pipeline_state["last_run"] = datetime.now().isoformat()
    _log("Orchestrator: Pipeline complete.")


async def pipeline_loop():
    """Background loop — runs pipeline every 30 seconds."""
    while True:
        try:
            await run_pipeline()
        except Exception as e:
            _log(f"Orchestrator error: {e}")
        await asyncio.sleep(30)
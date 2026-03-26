"""FastAPI backend for Palm Oil Estate Intelligence Platform."""
import asyncio
import csv
import io
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from estate_agents import pipeline_state, run_pipeline, pipeline_loop, DATA_FILE


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Only start auto-loop if data file already exists
    if os.path.exists(DATA_FILE):
        task = asyncio.create_task(pipeline_loop())
    yield


app = FastAPI(title="Palm Oil Estate Intelligence", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Endpoints ──────────────────────────────────────────────────

@app.get("/api/pipeline-status")
def get_status():
    return {
        "status": pipeline_state["status"],
        "last_run": pipeline_state["last_run"],
        "logs": pipeline_state["logs"][-30:],
    }


@app.get("/api/metrics")
def get_metrics():
    return pipeline_state.get("metrics", {})


@app.get("/api/forecast")
def get_forecast():
    return pipeline_state.get("forecast", {})


@app.get("/api/alerts")
def get_alerts():
    return pipeline_state.get("alerts", [])


@app.get("/api/estates")
def get_estates():
    return pipeline_state.get("estate_summary", [])


@app.post("/api/upload")
async def upload_data(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a JSON or CSV file with estate operations data, then run pipeline."""
    content = await file.read()
    filename = file.filename or ""

    try:
        if filename.lower().endswith(".csv"):
            # Parse CSV → list of dicts
            text = content.decode("utf-8")
            reader = csv.DictReader(io.StringIO(text))
            records = []
            for row in reader:
                # Auto-convert numeric fields
                clean = {}
                for k, v in row.items():
                    k = k.strip()
                    try:
                        clean[k] = float(v)
                    except (ValueError, TypeError):
                        clean[k] = v.strip() if isinstance(v, str) else v
                records.append(clean)
        elif filename.lower().endswith(".json"):
            records = json.loads(content)
            if isinstance(records, dict):
                records = records.get("data", records.get("records", [records]))
        else:
            return {"error": "Unsupported file type. Upload .json or .csv"}

        # Save to data file
        with open(DATA_FILE, "w") as f:
            json.dump(records, f, indent=2)

        pipeline_state["logs"].append(f"[UPLOAD] {filename}: {len(records)} records received.")

    except Exception as e:
        return {"error": f"Failed to parse file: {str(e)}"}

    # Trigger pipeline
    if pipeline_state["status"] != "running":
        background_tasks.add_task(asyncio.get_event_loop().create_task, run_pipeline())

    return {"message": f"Uploaded {len(records)} records from {filename}. Pipeline triggered."}


@app.post("/api/ingest")
async def trigger_ingest(background_tasks: BackgroundTasks):
    if pipeline_state["status"] == "running":
        return {"message": "Pipeline already running"}
    if not os.path.exists(DATA_FILE):
        return {"message": "No data file found. Upload a JSON or CSV file first."}
    background_tasks.add_task(asyncio.get_event_loop().create_task, run_pipeline())
    return {"message": "Pipeline triggered"}


# ── Serve frontend ─────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")
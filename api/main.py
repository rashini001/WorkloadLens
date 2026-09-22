"""
WorkloadLens API — wires the trained ML risk classifier (src/predict.py) to
the RAG generation pipeline (src/rag/rag_pipeline.py) behind a small FastAPI
app, and serves the demo frontend.

Run:
    uvicorn main:app --reload --port 8000
(from inside the api/ folder — see README for the full sequence)

Endpoints:
    POST /api/classify   -> ML risk classification only
    POST /api/guidance    -> ML classification + RAG-grounded guidance
    POST /api/compare     -> RAG guidance AND baseline (no-RAG) guidance,
                              side by side — used for the RQ3/RQ4 evaluation
                              study in Chapter 3.6
    GET  /                -> demo frontend
"""
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
import predict  # noqa: E402
from rag import rag_pipeline  # noqa: E402

app = FastAPI(title="WorkloadLens API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class ProfileRequest(BaseModel):
    profile: Dict[str, Any]          # raw survey-style answers, see predict.py docstring
    free_text: Optional[str] = ""    # optional student description of their situation


@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/api/classify")
def classify(req: ProfileRequest):
    try:
        return predict.predict_risk(req.profile)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/guidance")
def guidance(req: ProfileRequest):
    try:
        risk_result = predict.predict_risk(req.profile)
        rag_result = rag_pipeline.generate_rag_response(risk_result, req.free_text)
        return {
            "risk": risk_result,
            "guidance": rag_result["answer"],
            "retrieved_sources": [
                {"layer": c["metadata"]["layer"], "source": c["metadata"]["source"]}
                for c in rag_result["retrieved_chunks"]
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compare")
def compare(req: ProfileRequest):
    """RAG-grounded guidance vs. baseline (no retrieval) — for the blind
    student rating study described in Chapter 3.6."""
    try:
        risk_result = predict.predict_risk(req.profile)
        rag_result = rag_pipeline.generate_rag_response(risk_result, req.free_text)
        baseline_result = rag_pipeline.generate_baseline(risk_result, req.free_text)
        return {
            "risk": risk_result,
            "workloadlens_response": rag_result["answer"],
            "baseline_llm_response": baseline_result["answer"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

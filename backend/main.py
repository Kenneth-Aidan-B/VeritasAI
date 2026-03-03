"""
VeritasAI — FastAPI Backend Application
Provides REST API endpoints and WebSocket real-time streaming for the verification pipeline.
"""

import json
import uuid
import logging
import asyncio
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import settings
from backend.models.schemas import (
    AnalysisRequest,
    FullAnalysisResult,
    AnalysisStatus,
    WSMessage,
)
from backend.graph.orchestrator import run_analysis

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─── WebSocket Connection Manager ───────────────────────────────────────────


class ConnectionManager:
    """Manages WebSocket connections for real-time streaming."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected: {session_id}")

    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)
        logger.info(f"WebSocket disconnected: {session_id}")

    async def send_update(self, session_id: str, message: dict):
        ws = self.active_connections.get(session_id)
        if ws:
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send WS update to {session_id}: {e}")
                self.disconnect(session_id)

    async def broadcast(self, message: dict):
        disconnected = []
        for session_id, ws in self.active_connections.items():
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(session_id)
        for sid in disconnected:
            self.disconnect(sid)


manager = ConnectionManager()

# ─── Application Lifespan ───────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("🚀 VeritasAI Backend starting up...")
    missing_keys = settings.validate()
    if missing_keys:
        logger.warning(f"⚠️  Missing API keys: {', '.join(missing_keys)}")
        logger.warning("Some features may be limited. Set keys in .env file.")
    else:
        logger.info("✅ All API keys configured")
    yield
    logger.info("🛑 VeritasAI Backend shutting down...")


# ─── FastAPI Application ────────────────────────────────────────────────────

app = FastAPI(
    title="VeritasAI",
    description="Intelligent Real-Time Misinformation Analysis & Correction Platform",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory session store ────────────────────────────────────────────────

analysis_results: dict[str, FullAnalysisResult] = {}

# ─── REST API Endpoints ─────────────────────────────────────────────────────


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "name": "VeritasAI",
        "version": "1.0.0",
        "status": "running",
        "description": "Intelligent Real-Time Misinformation Analysis & Correction Platform",
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    missing = settings.validate()
    return {
        "status": "healthy" if not missing else "degraded",
        "missing_keys": missing,
        "llm_model": settings.LLM_MODEL,
    }


@app.post("/api/analyze", response_model=FullAnalysisResult)
async def analyze_text(request: AnalysisRequest):
    """
    Submit text for fact-checking analysis.
    Returns the complete analysis result.
    """
    session_id = request.session_id or str(uuid.uuid4())

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(request.text) > 10000:
        raise HTTPException(status_code=400, detail="Text exceeds maximum length of 10,000 characters")

    logger.info(f"Starting analysis for session {session_id}")

    async def progress_callback(update: dict):
        """Send progress updates via WebSocket."""
        await manager.send_update(session_id, update)

    try:
        result = await run_analysis(
            text=request.text,
            session_id=session_id,
            enable_debate=request.enable_debate,
            progress_callback=progress_callback,
        )

        # Store result
        analysis_results[session_id] = result

        return result

    except Exception as e:
        logger.error(f"Analysis failed for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/result/{session_id}", response_model=FullAnalysisResult)
async def get_result(session_id: str):
    """Retrieve a previously completed analysis result."""
    result = analysis_results.get(session_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis result not found")
    return result


@app.get("/api/sessions")
async def list_sessions():
    """List all analysis sessions."""
    return {
        "sessions": [
            {
                "session_id": sid,
                "status": r.status.value,
                "claims_count": len(r.claims),
                "started_at": r.started_at,
                "completed_at": r.completed_at,
            }
            for sid, r in analysis_results.items()
        ]
    }


@app.get("/api/responsible-ai-card")
async def get_responsible_ai_card():
    """Get the Responsible AI card template."""
    from backend.graph.orchestrator import _build_responsible_ai_card
    return _build_responsible_ai_card()


# ─── WebSocket Endpoint ─────────────────────────────────────────────────────


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time analysis streaming.
    
    Client connects before starting analysis to receive live updates.
    """
    await manager.connect(websocket, session_id)

    try:
        # Send initial connection confirmation
        await manager.send_update(session_id, {
            "type": "connected",
            "session_id": session_id,
            "message": "Connected to VeritasAI real-time stream",
            "timestamp": datetime.utcnow().isoformat(),
        })

        while True:
            # Listen for messages from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await manager.send_update(session_id, {
                    "type": "error",
                    "message": "Invalid JSON",
                })
                continue

            if message.get("type") == "analyze":
                # Start analysis from WebSocket
                text = message.get("text", "")
                enable_debate = message.get("enable_debate", True)

                if not text.strip():
                    await manager.send_update(session_id, {
                        "type": "error",
                        "message": "Text cannot be empty",
                    })
                    continue

                await manager.send_update(session_id, {
                    "type": "status_update",
                    "status": "starting",
                    "message": "Analysis pipeline starting...",
                    "progress": 0.0,
                    "timestamp": datetime.utcnow().isoformat(),
                })

                # Run analysis with progress callbacks
                async def ws_progress_callback(update: dict):
                    await manager.send_update(session_id, update)

                try:
                    result = await run_analysis(
                        text=text,
                        session_id=session_id,
                        enable_debate=enable_debate,
                        progress_callback=ws_progress_callback,
                    )

                    analysis_results[session_id] = result

                    await manager.send_update(session_id, {
                        "type": "analysis_complete",
                        "session_id": session_id,
                        "result": json.loads(result.model_dump_json()),
                        "timestamp": datetime.utcnow().isoformat(),
                    })

                except Exception as e:
                    await manager.send_update(session_id, {
                        "type": "error",
                        "message": f"Analysis failed: {str(e)}",
                        "timestamp": datetime.utcnow().isoformat(),
                    })

            elif message.get("type") == "ping":
                await manager.send_update(session_id, {
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat(),
                })

    except WebSocketDisconnect:
        manager.disconnect(session_id)
        logger.info(f"WebSocket client {session_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
        manager.disconnect(session_id)


# ─── Run with Uvicorn ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=True,
    )

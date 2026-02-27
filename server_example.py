#!/usr/bin/env python3
"""
Minimal FastAPI server for Neo - allows remote access to Neo's reasoning engine.

Run with: uvicorn server_example:app --host 0.0.0.0 --port 8000
"""

import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from neo.cli import NeoEngine, NeoInput, TaskType, ContextFile
from neo.adapters import create_adapter
from neo.config import NeoConfig

app = FastAPI(title="Neo Reasoning Server")


# Request/Response models
class NeoRequest(BaseModel):
    prompt: str
    working_directory: str | None = None
    context_files: list[dict] = []
    task_type: str = "feature"
    json_output: bool = True


class NeoResponse(BaseModel):
    plan: list[dict]
    code_suggestions: list[dict]
    confidence: float
    confidence_interpretation: str
    notes: str
    next_questions: list[str]
    metadata: dict


@app.get("/")
async def root():
    return {
        "service": "Neo Reasoning Server",
        "version": "0.10.0",
        "status": "running",
    }


@app.post("/reason", response_model=NeoResponse)
async def reason(request: NeoRequest):
    """Process a reasoning request and return structured output."""
    try:
        # Try to load config for API key
        config = NeoConfig.load()

        # Create adapter (requires API key to be configured)
        adapter = create_adapter(
            provider=config.provider,
            model=config.model,
            api_key=config.api_key,
        )

        # Create engine
        codebase_root = request.working_directory or os.getcwd()
        engine = NeoEngine(
            lm_adapter=adapter,
            codebase_root=codebase_root,
            config=config,
        )

        # Build input
        context_files = [
            ContextFile(**cf) for cf in request.context_files
        ]

        neo_input = NeoInput(
            prompt=request.prompt,
            task_type=TaskType(request.task_type),
            context_files=context_files,
            working_directory=codebase_root,
            safe_read_paths=[codebase_root] if codebase_root else [],
        )

        # Process
        output = engine.process(neo_input)

        # Build response
        result = {
            "plan": [
                {
                    "description": step.description,
                    "rationale": step.rationale,
                    "dependencies": step.dependencies,
                }
                for step in output.plan
            ],
            "code_suggestions": [
                {
                    "file_path": sugg.file_path,
                    "unified_diff": sugg.unified_diff,
                    "description": sugg.description,
                    "confidence": sugg.confidence,
                    "tradeoffs": sugg.tradeoffs,
                }
                for sugg in output.code_suggestions
            ],
            "confidence": output.confidence,
            "confidence_interpretation": _interpret_confidence(
                output.confidence,
                output.next_questions,
                output.plan,
                output.code_suggestions,
            ),
            "notes": output.notes,
            "next_questions": output.next_questions,
            "metadata": output.metadata,
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _interpret_confidence(confidence, next_questions, plan, code_suggestions):
    """Copy of the confidence interpreter function."""
    if confidence >= 0.8:
        interpretation = (
            "High confidence! The solution appears solid with strong alignment to "
            "project patterns and requirements."
        )
    elif confidence >= 0.6:
        interpretation = (
            "Moderate confidence. The solution is reasonable but consider reviewing "
            "the suggested Next Questions for potential areas to clarify."
        )
    elif confidence >= 0.4:
        interpretation = (
            "Lower confidence. The solution may need careful review. See Next Questions "
            "for key areas to verify before implementation."
        )
    else:
        interpretation = (
            "Very low confidence. Proceed with caution. The issue may require additional "
            "context or a different approach altogether."
        )
    return interpretation


# Health check endpoint
@app.get("/health")
async def health():
    """Check if server is healthy and configured."""
    try:
        # Try to load config
        config = NeoConfig.load()
        return {
            "status": "healthy",
            "configured": bool(config.api_key),
            "provider": config.provider,
            "model": config.model,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "configured": False,
        }


if __name__ == "__main__":
    import uvicorn

    print("Starting Neo Reasoning Server...")
    print("Configure Neo first: neo --config set --config-key provider openai")
    uvicorn.run(app, host="0.0.0.0", port=8000)

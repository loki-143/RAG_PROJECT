#!/bin/bash
cd rag_agent
gunicorn fastapi_app:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

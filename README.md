# generative_ai_project

Two API endpoints:

- POST /api/upload — form-data: file (xlsx/csv), project_name
- POST /api/query — JSON body: {"project_name": "...", "query": "..."}

Environment variables required for full functionality:
- DATABASE_URL (postgres connection string)
- AWS_S3_BUCKET (bucket name)
- AWS_REGION
- LITE_LLM_API_KEY (optional for real model calls)

Run locally (after setting env vars):
```bash
uvicorn src.main:app --reload --port 8000
```

Notes:
- The project provides a Lite LLM stub. Replace src/llm/lite_client.py with your provider SDK when ready.
- S3 and Postgres must be reachable from the runtime environment.

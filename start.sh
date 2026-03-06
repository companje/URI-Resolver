# DATA_DIR=app/data uv run uvicorn app.main:app --reload
DATA_DIR=app/data uv run uvicorn app.main:app --reload --workers 1 --host 127.0.0.1 --port 8000 
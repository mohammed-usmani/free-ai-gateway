test:
	uv run pytest -q

run:
	uv run uvicorn gateway.app:build --factory --port 8000

smoke:
	uv run python scripts/smoke.py

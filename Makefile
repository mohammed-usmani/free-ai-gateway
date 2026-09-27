test:
	uv run pytest -q

run:
	uv run uvicorn gateway.app:build --factory --port 8000

smoke:
	uv run python scripts/smoke.py --mode free

smoke-paid:
	uv run python scripts/smoke.py --mode paid

usage:
	@curl -s -H "Authorization: Bearer $$GATEWAY_API_KEY" 'http://127.0.0.1:8000/v1/usage?group_by=model' | python3 -m json.tool

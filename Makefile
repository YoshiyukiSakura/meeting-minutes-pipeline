.PHONY: test test-light lint doctor

test:
	uv run --with pytest pytest -q

test-light:
	PYTHONPATH=. uv run --no-project --with pytest --with numpy --with pillow pytest -q

lint:
	uv run --no-project --with ruff ruff check .

doctor:
	uv run meeting-minutes doctor

.PHONY: install test lint run-api run-ui eval docker-up

install:
	pip install -e .[ui,dev]

test:
	pytest

lint:
	ruff check src tests app eval

run-api:
	uvicorn rag_assistant.api:app --reload --port 8000

run-ui:
	streamlit run app/streamlit_app.py

eval:
	python eval/run_eval.py

docker-up:
	docker compose up --build

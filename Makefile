.PHONY: build up down logs test eval optimize clean help run-api run-frontend dashboard

help:
	@echo "Available commands:"
	@echo "  make build         - Build the docker images"
	@echo "  make up            - Start local services (e.g., Ollama server) in the background"
	@echo "  make down          - Stop local services"
	@echo "  make logs          - View logs of running services"
	@echo "  make test          - Run the full pytest suite in the docker container"
	@echo "  make eval          - Run the baseline and calibrated hybrid evaluation script"
	@echo "  make optimize      - Run the threshold sweep and calibration optimization script"
	@echo "  make clean         - Clean python cache files and local execution logs"
	@echo "  make run-api       - Start the FastAPI backend"
	@echo "  make run-frontend  - Start the React frontend dashboard"
	@echo "  make dashboard     - Start both backend and frontend dashboard concurrently"

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

test:
	docker compose run --rm app python -m pytest

eval:
	docker compose run --rm app python run_eval.py

optimize:
	docker compose run --rm app python sweep_optimizer.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f routing_execution.jsonl eval_results_log.json

run-api:
	docker compose run --rm app uvicorn api:app --host 0.0.0.0 --port 8000

run-frontend:
	cd frontend && npm run dev -- --host 0.0.0.0

dashboard:
	@echo "Starting backend and frontend..."
	make -j2 run-api run-frontend


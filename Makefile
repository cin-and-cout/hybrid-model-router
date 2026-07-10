.PHONY: build up down logs test eval optimize clean help

help:
	@echo "Available commands:"
	@echo "  make build    - Build the docker images"
	@echo "  make up       - Start local services (e.g., Ollama server) in the background"
	@echo "  make down     - Stop local services"
	@echo "  make logs     - View logs of running services"
	@echo "  make test     - Run the full pytest suite in the docker container"
	@echo "  make eval     - Run the baseline and calibrated hybrid evaluation script"
	@echo "  make optimize - Run the threshold sweep and calibration optimization script"
	@echo "  make dashboard - Run the Streamlit interactive dashboard"
	@echo "  make clean    - Clean python cache files and local execution logs"

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

dashboard:
	docker compose run --rm app streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f routing_execution.jsonl eval_results_log.json

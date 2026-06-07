.PHONY: install dev train serve test docker-up docker-down clean

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

train:        ## run a single retrain/gate/export cycle
	python -m forge.pipeline --once

loop:         ## run the daily retrain loop in the foreground
	python -m forge.pipeline --loop

serve:        ## run the inference server locally
	uvicorn forge.server:app --host 0.0.0.0 --port 8000

test:
	pytest -q

docker-up:    ## build and start trainer + inference
	docker compose up -d --build

docker-down:
	docker compose down

clean:
	rm -rf data models predict.pkl __pycache__ forge/__pycache__ .pytest_cache

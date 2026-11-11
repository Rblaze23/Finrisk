.PHONY: up down ingest train validate test

up:
	docker compose up -d

down:
	docker compose down

ingest:
	python -m finrisk.data.ingest --sample

train:
	python -m finrisk.models.train

validate:
	python -m finrisk.validation.run_validation

test:
	pytest -q

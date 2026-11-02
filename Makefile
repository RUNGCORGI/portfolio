.PHONY: help up down build test lint sample logs clean size

help:
	@echo "make build   - build docker image"
	@echo "make up      - chay app tai http://localhost:8501"
	@echo "make test    - chay pytest trong container"
	@echo "make sample  - sinh lai du lieu mau offline"
	@echo "make size    - so sanh kich thuoc image (P1 demo)"
	@echo "make clean   - xoa cache"

build:
	docker compose build

up:
	docker compose up -d && echo "-> http://localhost:8501"

down:
	docker compose down

test:
	docker compose run --rm --profile ci tests

test-local:
	python -m pytest -v

sample:
	python -m core.fixtures

size:
	docker images portfolio-terminal:nhom4 --format "{{.Repository}}:{{.Tag}}\t{{.Size}}"

logs:
	docker compose logs -f terminal

clean:
	rm -rf data/cache __pycache__ .pytest_cache core/__pycache__ tabs/__pycache__

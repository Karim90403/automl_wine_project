install:
	uv pip install -r requirements.txt

download:
	python -m src.download_data

train:
	python -m src.train

evaluate:
	python -m src.evaluate

test:
	pytest

monitor:
	python -m src.monitor

build:
	docker compose build

up:
	docker compose up

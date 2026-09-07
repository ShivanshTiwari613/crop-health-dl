PY      ?= .venv/bin/python
CONFIG  ?= configs/resnet50_coarse.yaml
RUN     ?= results/resnet50_coarse

.PHONY: setup data train eval table test lint smoke clean

setup:            ## create venv + install deps
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements.txt

data:             ## download PlantVillage via tensorflow_datasets (~0.8 GB)
	$(PY) data/download.py

train:            ## train one config, e.g. make train CONFIG=configs/inceptionv3_coarse.yaml
	$(PY) -m src.train --config $(CONFIG)

eval:             ## re-evaluate a finished run on the test split
	$(PY) -m src.evaluate --run $(RUN)

table:            ## rebuild results/RESULTS.md from all results/*/metrics.json
	$(PY) scripts/results_table.py

smoke:            ## 2-batch end-to-end run (needs data)
	$(PY) -m src.train --config configs/smoke.yaml

test:             ## unit/smoke tests (no dataset needed)
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests scripts data
	$(PY) -m black --check src tests scripts data

clean:
	rm -rf .pytest_cache .ruff_cache src/__pycache__ tests/__pycache__

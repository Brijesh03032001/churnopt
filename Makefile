.PHONY: help setup pipeline sql train decide report dashboard notebooks test lint format clean

PYTHON := python3
VENV := venv
BIN := $(VENV)/bin

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Create a virtual environment and install everything (including dev tools)
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt

pipeline:  ## Run every stage: sql, train, decide, experiment, explain, report
	$(BIN)/python run_pipeline.py

sql:  ## Run only the SQL analysis
	$(BIN)/python run_pipeline.py --stage sql

train:  ## Run only model comparison and evaluation
	$(BIN)/python run_pipeline.py --stage train

decide:  ## Re-run the decision analysis (after editing configs/economics.yaml)
	$(BIN)/python run_pipeline.py --stage decide
	$(BIN)/python run_pipeline.py --stage experiment
	$(BIN)/python run_pipeline.py --stage report

report:  ## Rebuild figures and the results summary
	$(BIN)/python run_pipeline.py --stage report

dashboard:  ## Launch the retention simulator (needs outputs/ from `make pipeline`)
	$(BIN)/streamlit run app/dashboard.py

notebooks:  ## Re-execute both notebooks in place (needs outputs/ from `make pipeline`)
	$(BIN)/jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

test:  ## Run the test suite with coverage
	$(BIN)/python -m pytest --cov=src --cov-report=term-missing

lint:  ## Check formatting and style
	$(BIN)/black --check src tests app run_pipeline.py
	$(BIN)/isort --check-only src tests app run_pipeline.py
	$(BIN)/flake8 src tests app run_pipeline.py --max-line-length=100 --ignore=E203,W503,E501,E226

format:  ## Format with black and isort
	$(BIN)/isort src tests app run_pipeline.py
	$(BIN)/black src tests app run_pipeline.py

clean:  ## Remove caches (keeps outputs/)
	rm -rf .pytest_cache .coverage htmlcov

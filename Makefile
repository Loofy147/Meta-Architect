.PHONY: install test clean lint format docs

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=hamha --cov=lma --cov-report=html

test-fast:
	pytest tests/ -v -x

lint:
	flake8 hamha/ lma/ tests/
	mypy hamha/ lma/

format:
	black hamha/ lma/ tests/ examples/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

docs:
	cd docs && make html

demo:
	python main.py

train:
	python examples/training_integration.py

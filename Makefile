.PHONY: lint format

lint: ## Run all Python linters
	pylint --rcfile=.pylintrc .
	flake8 --config=setup.cfg .
	mypy .
	ruff .

format: ## Format Python code
	black .

requirements: ## Generate requirements.txt
	pip freeze > requirements.txt

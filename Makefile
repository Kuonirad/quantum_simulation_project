.PHONY: lint:backend format:backend

lint\:backend: ## Run all Python linters
	pylint --rcfile=.pylintrc .
	flake8 --config=setup.cfg .
	mypy .
	ruff .

format\:backend: ## Format Python code
	black .

requirements.txt: ## Generate requirements.txt
	pip freeze > requirements.txt

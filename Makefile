check:
	flake8 --config .flake8-code .
	flake8 --config .flake8-doc .
	pylint lookout

.PHONY: check

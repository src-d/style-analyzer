check:
	flake8 --config .flake8-code .
	pylint lookout
	flake8 --config .flake8-doc .

.PHONY: check

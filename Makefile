.PHONY: default
default: check ;

check:
	ruff check

test:
	python3 -m pytest

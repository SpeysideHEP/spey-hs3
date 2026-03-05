.PHONY: all
all:
	make install
	python -m pip install pytest
	make test

.PHONY: install
install:
	pip install -e .

.PHONY: uninstall
uninstall:
	pip uninstall spey-hs3

.PHONY: test
test:
	pytest --cov=spey-hs3 tests/*py #--cov-fail-under 99

.PHONY: build
build:
	python -m build

.PHONY: check_dist
check_dist:
	twine check dist/*

.PHONY: testpypi
testpypi:
	python3 -m twine upload -r testpypi --verbose dist/*

.PHONY: pypi
pypi:
	python3 -m twine upload -r pypi --verbose dist/*

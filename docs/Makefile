all: docs

.PHONY: all docs clean-docs clean

docs: clean-docs
	sphinx-build -M html . _build

live:
	sphinx-autobuild . _build/html

clean-docs:
	rm -fr ./_build/*

clean: clean-docs

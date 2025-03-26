all: docs

.PHONY: all docs clean-docs clean

docs: clean
	mkdir -p build/html build/doctrees
	sphinx-build -WT --keep-going -b html -d build/doctrees source build/html

clean:
	rm -fr build/* source/reference/generated

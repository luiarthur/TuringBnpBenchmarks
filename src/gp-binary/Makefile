.PHONY: data html

all: script 

# docs-nb-html-dir = ../../docs/_posts/nb-html
docs-nb-html-dir = html

execute-all-notebooks:
	for fname in `ls notebooks/*.ipynb`; do \
		jupyter nbconvert \
		--to=notebook \
		--execute $$fname \
		--ExecutePreprocessor.timeout=None \
		--inplace; \
		done

# Provide argument `NB=<path-to-nb>`
execute-notebook:
		jupyter nbconvert \
			--to=notebook \
			--execute $(NB) \
			--ExecutePreprocessor.timeout=None \
			--inplace \

script:
	mkdir -p scripts
	for fname in `ls notebooks/*.ipynb`; do \
		jupyter-nbconvert $$fname \
		--to=script --output-dir=scripts/; \
		done

html:
	for fname in `ls notebooks/*.ipynb`; do \
		jupyter-nbconvert $$fname \
		--to=html --template=basic \
		--output-dir=$(docs-nb-html-dir) --template=basic \
		--HTMLExporter.anchor_link_text=' ' \
		--TemplateExporter.exclude_input_prompt=True; \
		done

parse_times:
	cd util && python3 parse_timings.py

bundle_to_docs: parse_times
	cp timings/timings.csv ../../docs/_data/timings/gpbinary/gpbinary_ppl_timings.csv

data:
	. ../../venv/bin/activate && cd util && python3 generate_data.py

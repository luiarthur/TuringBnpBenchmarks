.PHONY: clean serve clean_serve

all: clean_serve

clean:
	rm -rf _site

serve:
	bundle exec jekyll serve --incremental

serve_draft:
	bundle exec jekyll serve --incremental --drafts

clean_serve: clean serve

clean_serve_draft: clean serve_draft

update-gems:
	rm Gemfile.lock
	bundle install

install:
	bundle install

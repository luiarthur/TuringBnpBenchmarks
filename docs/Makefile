.PHONY: clean serve clean-serve

all: clean-serve

clean:
	rm -rf _site

serve:
	bundle exec jekyll serve --incremental --livereload

serve_draft:
	bundle exec jekyll serve --incremental --drafts

clean-serve: clean serve

clean-serve-draft: clean serve_draft

update-gems:
	rm Gemfile.lock
	bundle install

install:
	bundle install

python=python3
runtests_file=runtests.py
setup_file=setup.py

test: tests
tests: ${runtests_file}
	${python} $<

vtest: vtests
vtests: ${runtests_file}
	${python} $< -v

install: ${setup_file}
	${python} $< $@

develop: ${setup_file}
	${python} $< $@

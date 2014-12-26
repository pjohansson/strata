python=python3

test: runtests.py setup.py
	${python} setup.py test

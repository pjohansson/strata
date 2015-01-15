from setuptools import setup, Command, find_packages

class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import sys,subprocess
        coverage = [
                '--cov', 'flow',
                '--cov', 'dataformats',
                '--cov-report', 'term-missing'
                ]
        errno = subprocess.call([sys.executable, 'runtests.py', '-v'] + coverage)
        raise SystemExit(errno)

setup(
        name='flowfield',
        version='0.1',
        description='Tools for studying droplet and fluid dynamics data.',
        url='https://github.com/pjohansson/flowtools-rewrite',
        author='Petter Johansson',
        author_email='pettjoha@kth.se',
        license='None',
        packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
        cmdclass={'test': PyTest},
        install_requires=['setuptools'],
        tests_require=['pytest', 'pytest-cov', 'coverage'],
        zip_safe=True
        )

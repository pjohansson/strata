import inspect
import os
from setuptools import setup, Command, find_packages

def get_version():
    version_filename = 'VERSION'
    root_dir, _ = os.path.split(os.path.realpath(__file__))
    with open(os.path.join(root_dir, version_filename)) as fp:
        version = fp.readline().strip()
    return version

class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import sys,subprocess
        coverage = [
                '--cov', 'droplets',
                '--cov', 'strata',
                '--assert=plain',
                '--cov-report', 'term-missing'
                ]
        errno = subprocess.call([sys.executable, 'runtests.py'] + coverage)
        raise SystemExit(errno)

setup(
        name='strata',
        version=get_version(),
        description='Tools for studying droplet and fluid dynamics data.',
        url='https://github.com/pjohansson/flowtools-rewrite',
        author='Petter Johansson',
        author_email='pettjoha@kth.se',
        license='LGPLv3',
        packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
        cmdclass={'test': PyTest},
        install_requires=['setuptools', 'Click'],
        include_package_data=True,
        entry_points='''
            [console_scripts]
            strata=strata.strata:strata
        ''',
        tests_require=['pytest', 'pytest-cov', 'coverage'],
        zip_safe=False
        )

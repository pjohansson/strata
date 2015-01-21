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
                '--cov', 'droplets',
                '--cov', 'strata',
                '--cov-report', 'term-missing'
                ]
        errno = subprocess.call([sys.executable, 'runtests.py'] + coverage)
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
        install_requires=['setuptools', 'Click'],
        include_package_data=True,
        entry_points='''
            [console_scripts]
            avg_strata=strata.strata:average_cli
            conv_strata=strata.strata:convert_cli
            strata=strata.strata:strata
        ''',
        tests_require=['pytest', 'pytest-cov', 'coverage'],
        zip_safe=False
        )

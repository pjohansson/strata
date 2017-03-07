# Flow Field Tools

This package contains tools for analysing 2D data from flow field experiments,
mainly droplet wetting experiments.

These tools are split into two modules, separating an interaction layer
(`strata`) which contains command line utilities and reads data from file,
from a functional layer (`droplets`) which works with and analyses the data.
All of the modules are written targeting Python 3.4.


## Installation

The installation uses `setuptools` . Ensure that the requirements listed
in `requirements.txt` and `setup.py` are installed and run:

    python setup.py install

where `python` should be your Python 3 binary.


## Usage

All command line functionality is called from the installed utility `strata`.

    $ strata --help
    Usage: strata [OPTIONS] COMMAND [ARGS]...

      Tools for reading and analysing files of flow data.

    Options:
      -v, --version  Print version number and exit.
      -h, --help     Show this message and exit.

    Commands:
      average       Average multiple input data files.
      contact_line  Analyze the contact line bins.
      convert       Convert data files to another format.
      interface     Work with interface data of droplets.
      sample        Sample data maps.
      spreading     View or collect spreading data of droplets.
      view          Visualise the data of binned map files.


## Planned features

- [x] Reading 2D flow data from a basic file format.
- [x] Calculating the spreading radius per time of a droplet wetting
    experiment.
- [x] Implementing a command line utility for interacting with data.
- [x] Calculating the dynamic contact angle of a droplet per time.
- [x] Drawing flow fields of input data as quiver graphs.
- [x] Drawing an interface contour of a droplet.
- [x] Analysing the flow around a droplet's wetting contact line.


## To-do

- [ ] Document file formats.
- [ ] Implement a *better* file format.


## Development

These tools were written with the intent of fully utilizing test-driven
development and to build some experience with the methodology. Some tools,
mainly in the `strata` layer, still have very lacking coverage.

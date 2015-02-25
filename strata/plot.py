#!/usr/bin/env python
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""Module for plotting the spreading of droplets."""


@click.command(name='plot', short_help='Plot input spreading files.')
@click.argument('files', type=click.Path(exists=True), nargs=-1)
def plot(files):
    """Plot the spreading curves of input files.

    Input files are plaintext files of Grace format, which has the
    first column as time in ps and all following columns as spreading
    radius in nm.

    Args:
        files (paths): List of input files to plot.

    """

    data = read_spreading_data(*files)

    for d in data:
        d.plot()

    plt.show()

    return None


def read_spreading_data(*files):
    """Return spreading data read from input files.

    Args:
        files (paths): List of input files.

    Returns:
        pd.Series: List of pandas Series with data.

    """

    def read_file(filename):
        data = np.loadtxt(filename, unpack=True)
        times = data[0]

        return [pd.Series(rs, index=times, name='%s.%d' % (filename, i+1))
                for i, rs in enumerate(data[1:])]

    data = []
    for filename in files:
        data.extend(read_file(filename))

    return data


if __name__ == '__main__':
    plot()

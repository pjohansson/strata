#!/usr/bin/env python
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""Module for plotting the spreading of droplets."""


@click.command(name='plot', short_help='Plot input spreading files.')
@click.argument('files', type=click.Path(exists=True), nargs=-1)
@click.option('-rs', '--sync_radius', type=float, default=None)
def spreading_plot(files, **kwargs):
    """Plot the spreading curves of input files.

    Input files are plaintext files of Grace format, which has the
    first column as time in ps and all following columns as spreading
    radius in nm.

    Optionally synchronises the spreading data times to a common spreading
    radius if one is input.

    Args:
        files (paths): List of input files to plot.

    Keyword Args:
        sync_radius (float): Synchronise times at this radius.

    """

    data = read_spreading_data(*files)
    df = combine_spreading_data(data, kwargs.pop('sync_radius'))

    df.plot(legend=False, grid=False)
    plt.show()

    return None


def combine_spreading_data(data, sync_radius=None):
    """Return a DataFrame of combined data.

    Optionally synchronises the data at an input common spreading radius.

    Input:
        data (pd.Series): List of pandas Series objects to combine.

        sync_radius (float, optional): Spreading radius to synchronise
            times at.

    Returns:
        pd.DataFrame: Combined DataFrame.

    """

    if sync_radius != None:
        data = sync_time_at_radius(data, sync_radius)

    return pd.DataFrame(data).T


def sync_time_at_radius(data, radius):
    """Synchronise times to a common spreading radius.

    Args:
        data (pd.Series): List of pandas Series objects to combine.

        radius (float): Radius to synchronise at.

    Returns:
        pd.Series: List of pandas Series with synchronised data.

    """

    def calc_closest_time(s, radius):
        return (s - radius).abs().argmin()

    def get_adjusted_series(s, radius, sync_time):
        times = s.index - (calc_closest_time(s, radius) - sync_time)
        return pd.Series(s.values, index=times)

    sync_time = np.min([calc_closest_time(s, radius) for s in data])
    return [get_adjusted_series(s, radius, sync_time) for s in data]


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
    spreading_plot()

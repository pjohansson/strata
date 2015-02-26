#!/usr/bin/env python3
import click
import numpy as np
import os
import pandas as pd

#from strata.utils import prepare_path

"""Module for plotting the spreading of droplets."""


@click.command(name='view', short_help='Plot input spreading files.')
@click.argument('files', type=click.Path(exists=True), nargs=-1)
@click.option('-rs', '--sync_radius', type=float, default=None)
@click.option('-x', '--save_xvg', type=click.Path(), default=None)
@click.option('--plot/--noplot', default=True)
@click.option('--loglog', is_flag=True)
def spreading_view(files, **kwargs):
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

        save_xvg (path): Save combined data to path in .xvg format.

    """

    data = read_spreading_data(*files)
    df = combine_spreading_data(data, kwargs.pop('sync_radius'))

    save_path = kwargs.pop('save_xvg')
    if save_path:
        try:
            write_spreading_data(save_path, df)
        except Exception:
            print("[ERROR] Could not save data to '%s'" % save_path)

    if kwargs.pop('plot'):
        plot_spreading_data(df, **kwargs)

    return None


def decorate_graph(func):
    """Wrapper for decorating a figure.

    Creates a new figure window and sets options described below.

    Keyword Args:
        title (str, default=''): Title of graph.

        xlabel, ylabel (str, default=''): Axis labels.

        xlim, ylim (2-tuples, default=None): Limits of axes.

        loglog (bool, default=False): Set both axes to logarithmic scale.

        show (bool, default=True): Show the graph.

    Raises:
        StandardError: If something went wrong.

    """

    import matplotlib.pyplot as plt

    def graph_wrapper(*args, **kwargs):
        def pop_figure_kwargs(kwargs):
            fargs = {}

            key_defaults = (
                    (['title', 'xlabel', 'ylabel'], ''),
                    (['xlim', 'ylim'], None),
                    (['show'], True),
                    (['loglog'], False)
            )

            for keys, default in key_defaults:
                for k in keys:
                    fargs[k] = kwargs.pop(k, default)

            return fargs

        fargs = pop_figure_kwargs(kwargs)
        func(*args, **kwargs)

        plt.title(fargs['title'])
        plt.xlabel(fargs['xlabel'])
        plt.ylabel(fargs['ylabel'])

        plt.xlim(fargs['xlim'])
        plt.ylim(fargs['ylim'])

        if fargs['loglog']:
            plt.xscale('log')
            plt.yscale('log')

        if fargs['show']:
            plt.show()

        return func(*args, **kwargs)

    return graph_wrapper


@decorate_graph
def plot_spreading_data(df, **kwargs):
    df.plot(legend=False, grid=False)


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
        return pd.Series(s.values, index=times, name=s.name)

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
        try:
            data.extend(read_file(filename))
        except Exception:
            print("[WARNING] Could not read file at '%s'." % filename)
            raise SyntaxError

    return data


#@prepare_path
def write_spreading_data(path, df):
    """Write spreading data to file at path.

    Data is output in whitespace separated xmgrace format. NaN values
    are converted to 0 for compliance with column based plotting tools.

    Args:
        path (str): Path to output file.

        df (pd.DataFrame): Spreading data.

    """

    def write_header(path, df):
        import pkg_resources
        import time

        with open(path, 'w') as fp:
            time_str = time.strftime('%c', time.localtime())
            version_str = pkg_resources.require("flowfield")[0].version

            header = (
                    "# Spreading radius of a droplet impacting a substrate\n"
                    "# \n"
                    "# Created by module: %s\n"
                    "# Creation date: %s\n"
                    "# Using module version: %s\n"
                    "# \n"
                    % (__name__, time_str, version_str))

            inputs = (
                    "# Working directory: '%s'\n"
                    "# Input files:\n"
                    ) % os.path.abspath(os.curdir)

            names = [k.rsplit('.', 1)[0] for k in df.keys()]
            for name in names:
                inputs += "#   '%s'\n" % name

            inputs += (
                    "# \n"
                    "# Time (ps) Radius (nm)\n"
                    )

            fp.write(header + inputs)

    write_header(path, df)
    df.to_csv(path, sep=' ', mode='a', na_rep=0., header=False,
        float_format='%.3f')


if __name__ == '__main__':
    spreading_view()

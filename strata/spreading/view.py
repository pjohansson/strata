import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings

from strata.utils import decorate_graph, prepare_path

"""Module for plotting the spreading of droplets."""


def view_spreading(files, **kwargs):
    """Plot the spreading curves of input files.

    Input files are plaintext files of Grace format, which has the
    first column as time in ps and all following columns as spreading
    radius in nm.

    Optionally synchronises the spreading data times to a common spreading
    radius if one is input.

    Spreading times and radii can be scaled by supplying scaling factors.
    Input scaling factors are broadcasted to the data. This means that
    the input factors must be either a single factor scaling all data,
    or a list with separate factors for the entire data set.

    Args:
        files (paths): List of input files to plot.

    Keyword Args:
        sync_radius (float): Synchronise times at this radius.

        tau (float): Time scaling factors.

        R (float): Radius scaling factors.

        loglog (bool, default=False): Set graph axes to logarithmic scale.

        save_fig (path): Save figure to path.

        save_xvg (path): Save combined data to path in .xvg format.

    See `strata.utils.decorate_graph` for more figure drawing options.

    """

    data = read_spreading_data(*files)
    synced_data = combine_spreading_data(data,
            sync_radius=kwargs.pop('sync_radius', None),
            tau=kwargs.pop('tau', 1.), R=kwargs.pop('R', 1.)
            )

    save_path = kwargs.pop('save_xvg')
    if save_path:
        try:
            write_spreading_data(save_path, synced_data)
        except Exception as err:
            print(err)

    if (kwargs.get('show', True) or kwargs.get('save_fig', None) != None
            and len(data) > 0):
        kwargs['axis'] = 'tight'
        plot_spreading_data(synced_data, **kwargs)

    return None


def combine_spreading_data(data, sync_radius=None, tau=1., R=1.):
    """Return a DataFrame of combined data.

    Optionally synchronises the data at an input common spreading radius
    and scales time and radius data by factors.

    See `scale_spreading_data` for information on scaling factors.
    Scaling is performed before the optional radius synchronisation.

    Input:
        data (pd.Series): List of pandas Series objects to combine.

        sync_radius (float, optional): Spreading radius to synchronise
            times at.

        tau (float, optional): Time scaling factors.

        R (float, optional): Radius scaling factors.

    Returns:
        pd.Series: List of pandas Series with synchronised data.

    """

    data = scale_spreading_data(data, tau, R)

    if sync_radius != None:
        data = sync_time_at_radius(data, sync_radius)

    return data


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


def scale_spreading_data(data, tau=1., R=1.):
    """Scale spreading times and radii.

    The data is scaled by dividing with the input factors, ie. t* = t/tau
    where t is the non-scaled times, tau the scaling factor and t* the
    scaled times of the returned data.

    Input scaling factors are broadcasted to the data. This means that
    the input factors must be either a single factor scaling all data,
    or a list with separate factors for the entire data set.

    Args:
        data (pd.Series): List of data to scale.

        tau (float): Time scaling factors.

        R (float): Radius scaling factors.

    Returns:
        pd.Series: List of scaled data.

    Raises:
        TypeError: If scaling factors can not be broadcast to data.

    """

    def scale_series(view, tau, R):
        new = view().copy()
        new.index /= tau
        new /= R

        return new

    # Create list of views to broadcast against the list and not its data
    # Either numpy or pandas is trying to be too smart otherwise
    view = [d.view for d in data]

    try:
        bc = np.broadcast(view, tau, R)
    except ValueError:
        raise TypeError("Could not broadcast scaling factors to data.")
    else:
        scaled_data = [scale_series(v, t, r) for v, t, r in bc]

    return scaled_data

def read_spreading_data(*files):
    """Return spreading data read from input files.

    Args:
        files (paths): List of input files.

    Returns:
        pd.Series: List of pandas Series with data.

    """

    def read_file(filename):
        """Read Grace formatted file, comments starting with # or @."""

        def read_plain_file(filename):
            """Read a simple file, comments starting with '#'."""

            data = np.loadtxt(filename, unpack=True, comments='#')
            times = data[0]
            all_series = [get_series(radii, times, filename, i+1)
                    for i, radii in enumerate(data[1:])]

            return all_series

        def read_xvg_file(filename):
            """Read a Grace formatted file properly."""

            def split_graph_data(data):
                split_data = data.split('@target ')
                header = split_data[0]
                graphs = [g.splitlines()[1:] for g in split_data[1:]]

                return header, graphs

            def get_series_data(graph_data):
                """Parse individual graph line sets for data."""

                def get_values(line):
                    """Get the values from a single line."""

                    try:
                        vals = [float(v) for v in line.split()]
                    except Exception:
                        raise ValueError("Could not parse line %r" % line)

                    return vals

                num_graph = 0
                all_series = []
                for graph in graph_data:
                    set_data = np.array([get_values(line) for line in graph
                            if (not line.startswith('@') and line != '&')])

                    times = set_data[:,0]

                    # Might get multiple values per time, these are separate series
                    num_sets = set_data.shape[1] - 1
                    for i in range(num_sets):
                        s = get_series(set_data[:,i+1], times, filename, num_graph+1)
                        all_series.append(s)
                        num_graph += 1

                return all_series

            with open(filename) as fp:
                read_data = fp.read()

            # Header might be used later to read comments and legends?
            header, graph_series = split_graph_data(read_data)
            series_data = get_series_data(graph_series)

            return series_data

        def get_series(radii, times, filename, num):
            """Return a pd.Series object of input data."""

            get_name = lambda filename, num: '%s.%d' % (filename, num)
            s = pd.Series(radii, index=times, name=get_name(filename, num))

            return s.dropna()

        try:
            series = read_plain_file(filename)
        except Exception:
            series = read_xvg_file(filename)

        return series

    data = []
    for filename in files:
        try:
            data.extend(read_file(filename))
        except Exception:
            print("[WARNING] Could not read file at '%s'." % filename)

    return data


@prepare_path
def write_spreading_data(path, all_series):
    """Write spreading data to file at path.

    Data is output in whitespace separated xmgrace format. NaN values
    are converted to 0 for compliance with column based plotting tools.

    Args:
        path (str): Path to output file.

        all_series (pd.Series): List of spreading data as pandas Series.

    """

    def write_header(path, all_series):
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

            files = [s.name for s in all_series]
            for filename in files:
                inputs += "#   '%s'\n" % filename

            inputs += (
                    "# \n"
                    "# Time (ps) Radius (nm)\n"
                    )

            fp.write(header + inputs)


    def write_data(path, all_series):
        """Output spreading data to file."""

        with open(path, 'a') as fp:
            fp.write("@with g0\n")
            for i, filename in enumerate(s.name for s in all_series):
                fp.write("@    s%d comment \"%s\"\n" % (i, filename))

            for i, s in enumerate(all_series):
                fp.write("@target G0.S%d\n" % i)
                fp.write("@type xy\n")
                fp.write(s.to_string())
                fp.write('\n')

                if i + 1 < len(all_series):
                    fp.write("&\n")

    write_header(path, all_series)
    write_data(path, all_series)


@decorate_graph
def plot_spreading_data(all_series, **kwargs):
    """Plot input spreading data.

    Args:
        all_series (list): List of pd.Series with spreading data.

    See `strata.utils.decorate_graph` for input options.

    """

    plt.hold(True)
    for s in all_series:
        plt.plot(s.index, s.values, **kwargs)

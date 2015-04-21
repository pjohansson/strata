import numpy as np
import os
import pandas as pd

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
    df = combine_spreading_data(data,
            sync_radius=kwargs.pop('sync_radius', None),
            tau=kwargs.pop('tau', 1.), R=kwargs.pop('R', 1.)
            )

    save_path = kwargs.pop('save_xvg')
    if save_path:
        try:
            write_spreading_data(save_path, df)
        except Exception as err:
            print(err)

    if (kwargs.get('show', True) or kwargs.get('save_fig', None) != None
            and len(data) > 0):
        kwargs['axis'] = 'tight'
        plot_spreading_data(df, **kwargs)

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
        pd.DataFrame: Combined DataFrame.

    """

    data = scale_spreading_data(data, tau, R)

    if sync_radius != None:
        data = sync_time_at_radius(data, sync_radius)

    df = pd.DataFrame(data).T

    return df


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

        try:
            data = np.loadtxt(filename, unpack=True, comments='#')
        except ValueError:
            data = np.loadtxt(filename, unpack=True, comments='@')

        times = data[0]

        return [pd.Series(rs, index=times, name='%s.%d' % (filename, i+1))
                for i, rs in enumerate(data[1:])]

    data = []
    for filename in files:
        try:
            data.extend(read_file(filename))
        except Exception:
            print("[WARNING] Could not read file at '%s'." % filename)

    return data


@prepare_path
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


@decorate_graph
def plot_spreading_data(df, **kwargs):
    """Plot input spreading data.

    See `strata.utils.decorate_graph` for input options.

    """

    df.plot(legend=False, **kwargs)

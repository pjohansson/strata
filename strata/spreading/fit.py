import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.optimize as opt

from strata.spreading.view import read_spreading_data
from strata.utils import decorate_graph, prepare_path


def fit_spreading_data(files, lims=(None, None), save_xvg=None,
            out_lims=(None, None), **kwargs):
    """Return parameters for a power law fit of spreading data in input files.

    The fit is done for the data to the function r = a*t**k where r
    is the spreading radius at time t and a, k respectively are the
    amplitude and index of the function. If several files are input the
    fit is performed for all of them and returned as separate tuples in
    a list.

    By default draws a graph of the data with the fit. This can be turned
    by supplying a `show` keyword argument.

    Args:
        files (path): Paths of files to read spreading data from.

    Keyword Args:
        lims (floats, optional): 2-tuple with set limits on time data to
            fit for.

        save_xvg (path, optional): Save a coordinate representation of
            the fit to this file.

        out_lims (floats, optional): 2-tuple with set limits on the time
            data shown for the fitted data.

    Return:
        (a, k): List of 2-tuples with amplitudes a and indices k of the
            fit, one for each input spreading data.

    See `strata.utils.decorate_graph` for figure drawing keyword arguments.

    """

    data = read_spreading_data(*files)
    fit_params = get_fitting_parameters(data, lims)
    fit_data = get_fitted_data(fit_params, data, out_lims)

    if save_xvg != None:
        write_fitting_data(save_xvg, fit_data, lims, out_lims)

    if kwargs.get('show', True):
        plot_fitting_data(fit_data, data, **kwargs)

    return fit_params


def get_fitting_parameters(data, lims):
    """Return parameters from a power law fit against input data.

    Args:
        data (pd.Series): List of spreading data objects.

        lims (float, float): Fit the data for this time interval.

    Returns:
        (a, k): List of 2-tuples with amplitudes a and indices k of the
            fit, one for each input series.

    """

    def fit_series(s, lims):
        """Fit an input spreading data object to a power law function."""

        def func_minimise(guess_init, xslog, yslog):
            """Function to minimise when matching data."""

            fit_linear = lambda a, b, x: a + b*x
            a, b = guess_init

            return yslog - fit_linear(a, b, xslog)

        inds = get_inds_in_limits(s, lims)
        inds = (s.index > 0) & inds

        xslog = np.log10(s[inds].index.values)
        yslog = np.log10(s[inds].values)

        # Set initial guess, index for water usually ~1/2
        guess_init = [0.5, 0.5]
        opt_params = opt.leastsq(func_minimise, guess_init,
                args=(xslog, yslog), full_output=True)

        loga, k = opt_params[0]
        a = 10.0**loga

        return a, k

    fit_params = [fit_series(s, lims) for s in data]

    return fit_params


def get_fitted_data(fit_params, spreading_data, out_lims):
    """Return function data for input fitting parameters.

    Args:
        fit_params (floats): List of 2-tuples with fitted power law
            parameters (a, k).

        spreading_data (pd.Series): List of original spreading data.

        out_lims (floats): 2-tuple with set limits of returned data.

    Returns:
        pd.Series: List of fitted spreading data, one for each input
            parameter set.

    """

    def get_series(series, a, k):
        """Return a series with fitted data."""

        powfunc = lambda a, k, x: a*x**k
        inds = get_inds_in_limits(series, out_lims)

        xs = series[inds].index.values
        ys = powfunc(a, k, xs)
        name = series.name + ' a=%.3f, k=%.3f' % (a, k)

        return pd.Series(ys, index=xs, name=name)

    data = [get_series(s, a, k)
            for (a, k), s in zip(fit_params, spreading_data)]

    return data


def get_inds_in_limits(series, lims):
    """Return a boolean mask for input series within limits."""

    try:
        xmin, xmax = lims
        if xmin == None: xmin = np.min(series.index.values)
        if xmax == None: xmax = np.max(series.index.values)

        inds = (series.index >= xmin) & (series.index <= xmax)
        assert(inds.any())

    except AssertionError:
        print("Series data %r not in limit range %s, fitting entire set"
                % (series.name, repr(lims)))
        inds = series.notnull()

    except Exception:
        print("Could not parse limits %s" % repr(lims))
        inds = series.notnull()

    return inds


@decorate_graph
def plot_fitting_data(fit_data, data, **kwargs):
    """Plot fitting data alongside spreading data.

    Args:
        fit_data, data (pd.Series): Lists of corresponding data.

    See `strata.utils.decorate_graph` for input options.

    """

    plt.hold(True)
    for s_fit, s_data in zip(fit_data, data):
        fit_label = s_fit.name.strip(s_data.name).strip()
        plt.plot(s_data.index, s_data.values, label=s_data.name, **kwargs)
        plt.plot(s_fit.index, s_fit.values, label=fit_label, **kwargs)


@prepare_path
def write_fitting_data(path, all_series, lims, out_lims):
    """Write fitted spreading data to file at path.

    Data is output in whitespace separated xmgrace format. NaN values
    are converted to 0 for compliance with column based plotting tools.

    Args:
        path (str): Path to output file.

        all_series (pd.Series): List of spreading data as pandas Series.

    """

    def write_header(path, all_series):
        import strata
        import time

        with open(path, 'w') as fp:
            time_str = time.strftime('%c', time.localtime())

            header = (
                    "# Fitted spreading radii of a droplet impacting a substrate\n"
                    "# \n"
                    "# Created by module: %s\n"
                    "# Creation date: %s\n"
                    "# Using module version: %s\n"
                    "# \n"
                    % (__name__, time_str, strata.strata.version))

            inputs = (
                    "# Input:\n"
                    "#   Fit limits: %r, %r\n"
                    "#   Draw limits: %r, %r\n"
                    "# \n"
                    "# Working directory: '%s'\n"
                    "# Input files and fitting parameter values (r = a*t^k):\n"
                    ) % (lims[0], lims[1], out_lims[0], out_lims[1],
                            os.path.abspath(os.curdir))

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

import numpy as np
import os
import pandas as pd
import progressbar as pbar

from droplets.flow import FlowData
from droplets.interface import get_interface
from strata.dataformats.read import read_data_file
from strata.utils import find_singles_to_singles, pop_fileopts, prepare_path


def collect_interfaces(base, output, recenter=None, **kwargs):
    """Get the interfaces from files at input base and save to output.

    Args:
        base (str): Base path to input files.

        output (str): Base path to output interface files.

    Keyword Args:
        recenter (str, optional): Recenter the interface around 'zero',
            the center of mass 'com'.

        begin (int, default=1): First data map number.

        end (int, default=inf): Final data map number.

        ext (str, default='.dat'): File extension.

        outext (str, default='.xvg.'): Output file extension.

    Returns:
        xs, ys: 2-tuple with lists of all interface coordinates along x and y.

    See `droplets.interface.get_interface` for more keyword arguments.

    """

    # Set some default options
    kwargs.setdefault('cutoff', None)
    kwargs.setdefault('cutoff_radius', None)
    kwargs.setdefault('cutoff_bins', 1)
    kwargs.setdefault('outext', '.xvg')

    fopts = pop_fileopts(kwargs)

    label = 'M'
    quiet = kwargs.pop('quiet', False)

    files = list(find_singles_to_singles(base, output, **fopts))
    xs, ys = [], []

    if not quiet:
        widgets = ['Collecting from files: ',
                pbar.Bar(), ' (', pbar.SimpleProgress(), ') ', pbar.ETA()]
        progress = pbar.ProgressBar(widgets=widgets, maxval=len(files))
        progress.start()

    for i, (fn, fnout) in enumerate(files):
        data, _, _ = read_data_file(fn)
        flow = FlowData(data)
        x, y = get_interface_coordinates(flow, label, recenter, **kwargs)
        interface = pd.Series(x, index=y)

        write_interface_data(fnout, interface, [fnout], kwargs, recenter)

        xs.append(interface.values)
        ys.append(interface.index)

        if not quiet:
            progress.update(i+1)

    if not quiet:
        progress.finish()

    return xs, ys


@prepare_path
def write_interface_data(path, interface, fngroup, kwargs, recenter=None):
    """Write spreading data to file at path.

    Data is output in whitespace separated xmgrace format. NaN values
    are converted to 0 for compliance with column based plotting tools.

    Args:
        path (str): Path to output file.

        interface (Series): Interface data.

        fngroup (list): List of data files read from.

    """

    def get_header(path, fngroup):
        import strata
        import time

        time_str = time.strftime('%c', time.localtime())

        header = (
                "# Interface coordinates of a droplet\n"
                "# \n"
                "# Created by module: %s\n"
                "# Creation date: %s\n"
                "# Using module version: %s\n"
                "# \n"
                % (__name__, time_str, strata.strata.version))

        inputs = (
                "# Working directory: '%s'\n"
                "# Input files:\n"
                ) % os.path.abspath(os.curdir)
        for name in fngroup:
            inputs += "#   '%s'\n" % name
        inputs += "# \n"

        try:
            inputs += (
                    "# Input options:\n"
                    "#   Recenter: %r\n"
                    "#   Mass cut-off: %r\n"
                    "#   Radius cut-off: %r\n"
                    "#   Required # of bins: %r\n"
                    "# \n"
                    ) % (recenter, kwargs['cutoff'],
                            kwargs['cutoff_radius'], kwargs['cutoff_bins'])
        except KeyError:
            inputs += (
                    "# Input options: See original files.\n"
                    "# \n"
                    )

        inputs += "# x (nm) y (nm)"

        return header + inputs

    header = get_header(path, fngroup)
    data = np.array([interface, interface.index]).T
    np.savetxt(path, data, fmt='%.3f', delimiter=' ',
            header=header, comments='')


def get_interface_coordinates(flow, label, recenter=None, **kwargs):
    """Get the interface from a FlowData object using input label.

    Args:
        flow (FlowData): A FlowData object. Must contain a data record
            with coordinates and selected height map label.

        label (str): Record label used as base for the interface height map.

    Keyword Args:
        recenter (str, optional): Recenter the interface around 'zero' or
            the center of mass 'com'.

        coord_labels (2-tuple, default=('X', 'Y'): Record labels for coordinates.

    Returns:
        ndarray, ndarray: 2-tuple with interfaces coordinates along x and y.

    See `get_interface` for additional keyword arguments.

    """

    xl, yl = kwargs.get('coord_labels', ('X', 'Y'))

    left, right = np.array(list(get_interface(flow, label, **kwargs))).T
    inds = np.array(left.tolist() + right[::-1].tolist())

    xs, ys = (flow.data[l][inds] for l in (xl, yl))
    ys -= ys[0]

    if recenter == 'zero':
        xs -= np.mean([xs[0], xs[-1]])
    elif recenter == 'com':
        xs -= np.average(flow.data[xl], weights=flow.data[label])

    return xs, ys

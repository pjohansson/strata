import numpy as np
import os
import pandas as pd
import progressbar as pbar

from droplets.flow import FlowData
from droplets.interface import get_interface
from strata.dataformats.read import read_data_file
from strata.utils import find_singles_to_singles, pop_fileopts, prepare_path


def collect_interfaces(base, output, adjust_com=True, **kwargs):
    """Get the interfaces from files at input base and save to output.

    Args:
        base (str): Base path to input files.

        output (str): Base path to output interface files.

    Keyword Args:
        adjust_com (bool, default=True): Center spreading coordinates around
            the center of mass.

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
        x, y = get_interface_coordinates(flow, label, adjust_com, **kwargs)
        interface = pd.Series(x, index=y)

        write_interface_data(fnout, interface, [fnout], kwargs)

        xs.append(interface.values)
        ys.append(interface.index)

        if not quiet:
            progress.update(i+1)

    if not quiet:
        progress.finish()

    return xs, ys


@prepare_path
def write_interface_data(path, interface, fngroup, kwargs):
    """Write spreading data to file at path.

    Data is output in whitespace separated xmgrace format. NaN values
    are converted to 0 for compliance with column based plotting tools.

    Args:
        path (str): Path to output file.

        interface (Series): Interface data.

        fngroup (list): List of data files read from.

    """

    def get_header(path, fngroup):
        import pkg_resources
        import time

        time_str = time.strftime('%c', time.localtime())
        version_str = pkg_resources.require("flowfield")[0].version

        header = (
                "# Interface coordinates of a droplet\n"
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
        for name in fngroup:
            inputs += "#   '%s'\n" % name
        inputs += "# \n"

        try:
            inputs += (
                    "# Input options:\n"
                    "#   Adjust COM: %r\n"
                    "#   Mass cut-off: %r\n"
                    "#   Radius cut-off: %r\n"
                    "#   Required # of bins: %r\n"
                    "# \n"
                    ) % (kwargs['adjust_com'], kwargs['cutoff'],
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


def get_interface_coordinates(flow, label, adjust_com=False, **kwargs):
    """Get the interface from a FlowData object using input label.

    Args:
        flow (FlowData): A FlowData object. Must contain a data record
            with coordinates and selected height map label.

        label (str): Record label used as base for the interface height map.

    Keyword Args:
        adjust_com (bool, default=False): Center spreading coordinates around
            the center of mass.

        coord_labels (2-tuple, default=('X', 'Y'): Record labels for coordinates.

    Returns:
        ndarray, ndarray: 2-tuple with interfaces coordinates along x and y.

    See `get_interface` for additional keyword arguments.

    """

    xl, yl = kwargs.get('coord_labels', ('X', 'Y'))

    left, right = np.array(list(get_interface(flow, label, **kwargs))).T
    inds = np.array(left.tolist() + right[::-1].tolist())

    xs, ys = (flow.data[l][inds] for l in (xl, yl))
    if adjust_com == True:
        xs -= np.average(flow.data[xl], weights=flow.data[label])

    return xs, ys

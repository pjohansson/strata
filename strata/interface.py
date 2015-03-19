import numpy as np
import os
import pandas as pd

from droplets.flow import FlowData
from droplets.interface import get_interface
from strata.dataformats.read import read_from_files
from strata.utils import find_groups_to_singles, pop_fileopts, prepare_path


def read_interfaces(base, average=1, save_xvg='', **kwargs):
    """Get the interfaces from files at input base.

    Args:
        base (str): Base path to input files.

    Keyword Args:
        adjust_com (bool, default=True): Center spreading coordinates around
            the center of mass.

        average (int, default=1): Group and average interfaces from files
            of this bundle size.

        save_xvg (str, optional): Output interface coordinates to files
            at this base path.

        begin (int, default=1): First data map number.

        end (int, default=inf): Final data map number.

        ext (str, default='.dat'): File extension.

    Returns:
        xs, ys: 2-tuple with lists of all interface coordinates along x and y.

    See `get_interface` for more keyword arguments.

    """

    kwargs.setdefault('adjust_com', True)
    kwargs.setdefault('outext', '.xvg')
    fopts = pop_fileopts(kwargs)

    groups_singles = list(find_groups_to_singles(base, save_xvg, average,
            **fopts))
    xs, ys = [], []
    label = 'M'

    for i, (fn_group, fn_out) in enumerate(groups_singles):
        xs_group, ys_group = [], []
        for j, (data, _, _) in enumerate(read_from_files(*fn_group)):
            flow = FlowData(data)

            x, y = get_interface_coordinates(flow, label, **kwargs)
            xs_group.append(x)
            ys_group.append(y)

        left, right = combine_interfaces(xs_group, ys_group)
        interface = stitch_edge_series(left.mean(axis=1), right.mean(axis=1))

        if save_xvg != '':
            write_interface_data(fn_out, interface, fn_group, kwargs)

        xs.append(interface.values)
        ys.append(interface.index)

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

        inputs += (
                "# Input options:\n"
                "#   Adjust COM: %r\n"
                "#   Mass cut-off: %r\n"
                "#   Radius cut-off: %r\n"
                "#   Required # of bins: %r\n"
                "# \n"
                "# x (nm) y (nm)"
                ) % (kwargs['adjust_com'], kwargs['cutoff'],
                        kwargs['cutoff_radius'], kwargs['cutoff_bins'])

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


def combine_interfaces(xs, ys):
    """Return DataFrame objects of collected left and right interfaces.

    Args:
        xs, ys (ndarrays): Lists of input coordinate arrays.

    Returns:
        DataFrames: 2-tuple of left and right edges.

    """

    left, right = [pd.DataFrame(edge_coords).T
            for edge_coords in all_coords_to_edges(xs, ys)]

    return left, right


def all_coords_to_edges(xs, ys):
    """Return lists of coordinates as Series objects for both interface edges.

    Input lists of several interfaces to return as edges. Returned Series's
    will have y as indices and x as values.

    Args:
        xs, ys (ndarrays): Lists of input coordinate arrays.

    Returns:
        pd.Series, pd.Series: 2-tuple for left and right interface boundaries,
            with list of Pandas Series's objects for each edge.

    """

    split_at = lambda a: int(len(a)/2)

    left = [pd.Series(x[:split_at(x)], index=y[:split_at(x)])
            for x, y in zip(xs, ys)]
    right = [pd.Series(x[split_at(x):], index=y[split_at(x):])
            for x, y in zip(xs, ys)]

    return left, right


def stitch_edge_series(left, right):
    """Return coordinates from input left and right edges as single arrays.

    Args:
        left, right (Series): Pandas Series objects of edges.

    Returns:
        Series: Pandas Series of combined edges.

    """

    return left.append(right)

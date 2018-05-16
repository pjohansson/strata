import numpy as np
import os
import pandas as pd
import progressbar as pbar

from collections import namedtuple

from droplets.flow import FlowData
from droplets.interface import get_interface
from strata.dataformats.read import read_data_file
from strata.spreading.collect import init_periodic_info, \
    check_and_update_periodic_info, add_pbc_multipliers_to_edges, PeriodicInfo
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

    pbc_info_per_y = None
    yindex_data = None

    if not quiet:
        widgets = ['Collecting from files: ',
                pbar.Bar(), ' (', pbar.SimpleProgress(), ') ', pbar.ETA()]
        progress = pbar.ProgressBar(widgets=widgets, maxval=len(files))
        progress.start()

    for i, (fn, fnout) in enumerate(files):
        data, info, _ = read_data_file(fn)
        flow = FlowData(data, info=info)
        interface, pbc_info_per_y, yindex_data = get_interface_coordinates(
            flow, label, pbc_info_per_y, yindex_data, recenter, **kwargs
        )

        write_interface_data(fnout, interface, [fnout], kwargs, recenter)

        if not quiet:
            progress.update(i + 1)

    if not quiet:
        progress.finish()


def get_interface_coordinates(flow, label, pbc_info_per_y, yindex_data,
        recenter=None, **kwargs):
    """Get the interface from a FlowData object using input label.

    Args:
        flow (FlowData): A FlowData object. Must contain a data record
            with coordinates and selected height map label.

        label (str): Record label used as base for the interface height map.

        pbc_info_per_y (PeriodicInfo): Array of current pbc information for
            every y-value in the previous interface.

        yindex_data (YIndexData): Information about the indexing for the
            `pbc_info_per_y` array per y-value.

    Keyword Args:
        recenter (str, optional): Recenter the interface around 'zero' or
            the center of mass 'com'.

        coord_labels (2-tuple, default=('X', 'Y'): Record labels for coordinates.

    Returns:
        ndarray, ndarray, [PeriodicInfo], YIndexData: 2-tuple with interfaces coordinates along x and y, the updated periodic information and indexing.

    See `get_interface` for additional keyword arguments.

    """

    xl, yl = kwargs.get('coord_labels', ('X', 'Y'))

    left, right = np.array(list(
        get_interface(flow, label, search_longest_connected=True, **kwargs)
    )).T

    ys = flow.data[yl][left]
    xs_left = flow.data[xl][left]
    xs_right = flow.data[xl][right]

    box_x, _ = flow.size()

    interface, pbc_info_per_y, yindex_data = update_interface_with_pbc_info(
        ys, xs_left, xs_right, box_x, pbc_info_per_y, yindex_data
    )

    xs = interface['X']
    ys = interface['Y']

    ys -= ys[0]

    if recenter == 'zero':
        xs -= np.mean([xs[0], xs[-1]])
    elif recenter == 'com':
        xs -= np.average(flow.data[xl], weights=flow.data[label])

    return interface, pbc_info_per_y, yindex_data


# Data for indexing a vector for floating point y-values.
YIndexData = namedtuple("YIndexData", ["ymin", "ymax", "dy", "num"])

def get_yindex_data(ys):
    """Return the `YIndexData` corresponding to the input valuesself.

    Note that this assumes that all the values in the input array are
    evenly spaced.

    """

    dy = np.unique(np.diff(ys))[0]

    return YIndexData(
        ymin=np.min(ys),
        ymax=np.max(ys),
        dy=dy,
        num=ys.size,
    )

def get_yindex_for_value(y, yindex_data):
    """For a given `YIndexData`, return the index corresponding to a value.

    Values outside of the minimum and maximum values are clipped to the edges.

    """

    y = min(max(y, yindex_data.ymin), yindex_data.ymax)
    return int((y - yindex_data.ymin) / yindex_data.dy)

def transfer_values(xs, yindex_data, yindex_data_new):
    """Transfer data from an array to one with a new indexing.

    If `yindex_data` is `None`, an array matching `yindex_data_new` is
    returned filled with `None` values. In this case, `xs` can also be
    `None`, since it is not used.

    The idea is that we can have interface data for different y-values
    in every frame, but want to keep track of their PBC information to
    adjust the final coordinates. Thus we need some array that can merge
    these different sets.

    The indices which did not have a data in the input array are set to `None`.

    Args:
        xs (ndarray): Values corresponding to the `yindex_data` indexing.

        yindex_data (YIndexData): Indexing for the input data.

        yindex_data_new (YIndexData): Indexing for the output data.

    Returns:
        ndarray: An array matching the `yindex_data_new` object with the
            values from the input array set in their corresponding indices.

    Excepts:
        ValueError: If the input `YIndexData` objects do not have identical
            spacing `dy`.

    """

    xs_new = np.array([None for _ in range(yindex_data_new.num)])

    if yindex_data == None:
        return xs_new

    # Just return the array if the metadata is identical
    if np.isclose(yindex_data.ymin, yindex_data_new.ymin) \
            and np.isclose(yindex_data.ymax, yindex_data_new.ymax) \
            and np.isclose(yindex_data.dy, yindex_data_new.dy) \
            and yindex_data.num == yindex_data_new.num:
        return xs

    if not np.isclose(yindex_data.dy, yindex_data_new.dy):
        raise ValueError(
                "transfer of values is not implemented for different dy"
            )

    imin = get_yindex_for_value(yindex_data.ymin, yindex_data_new)
    imax = get_yindex_for_value(yindex_data.ymax, yindex_data_new)

    jmin = get_yindex_for_value(yindex_data_new.ymin, yindex_data)
    jmax = get_yindex_for_value(yindex_data_new.ymax, yindex_data)

    xs_new[imin:imax + 1] = xs[jmin:jmax + 1]

    return xs_new

def create_interface_array(ys, xs_left, xs_right):
    """Merge the left and right interface edges at the top."""

    interface = np.zeros(
        (2 * ys.size, ), dtype=[('Y', np.float), ('X', np.float)]
    )

    interface['Y'][:ys.size] = ys
    interface['Y'][ys.size:] = ys[::-1]
    interface['X'][:ys.size] = xs_left
    interface['X'][ys.size:] = xs_right[::-1]

    return interface

def fill_pbc_info_for_empty_y_values(pbc_info_per_y):
    """Return the input array with `None` values replaced by copies of others.

    The copies are taken towards the inner part, ie. from the bottom and from
    the top depending on where the `None` value is. The center is assumed
    to have no gaps.

    If no `PeriodicInfo` objects exist in the array, all values are set to
    the default `PeriodicInfo` object.

    """

    try:
        first = 0
        while (pbc_info_per_y[first] == None):
            first += 1
    except:
        pbc_info = init_periodic_info()
        pbc_info_per_y = [pbc_info] * len(pbc_info_per_y)
    else:
        for i in range(first):
            pbc_info_per_y[i] = pbc_info_per_y[first]

        last = -1
        while (pbc_info_per_y[last] == None):
            last -= 1

        for i in range(-1, last, -1):
            pbc_info_per_y[i] = pbc_info_per_y[last]

    return pbc_info_per_y

def update_interface_with_pbc_info(ys, xs_left, xs_right, box_x,
        old_pbc_info_per_y, old_yindex_data):
    """Check whether the interface jumps for any y-value and update pbc info."""

    new_yindex_data = get_yindex_data(ys)
    new_pbc_info_per_y = transfer_values(
        old_pbc_info_per_y, old_yindex_data, new_yindex_data
    )
    new_pbc_info_per_y = fill_pbc_info_for_empty_y_values(new_pbc_info_per_y)

    for i, (y, xl, xr) in enumerate(zip(ys, xs_left, xs_right)):
        pbc_info = new_pbc_info_per_y[i]

        if pbc_info != None:
            new_pbc_info = check_and_update_periodic_info(
                pbc_info, box_x, xl, xr
            )
        else:
            new_pbc_info = init_periodic_info(xleft=xl, xright=xr)

        new_pbc_info_per_y[i] = new_pbc_info

    final_xs_left = np.zeros(xs_left.shape)
    final_xs_right = np.zeros(xs_right.shape)

    for i, (xl, xr, pbc_info) in enumerate(zip(
                xs_left, xs_right, new_pbc_info_per_y
            )):
        xleft, xright = add_pbc_multipliers_to_edges(pbc_info, box_x, xl, xr)
        final_xs_left[i] = xleft
        final_xs_right[i] = xright

    interface = create_interface_array(ys, final_xs_left, final_xs_right)

    return interface, new_pbc_info_per_y, new_yindex_data


@prepare_path
def write_interface_data(path, interface, fngroup, kwargs, recenter=None):
    """Write spreading data to file at path.

    Data is output in whitespace separated xmgrace format. NaN values
    are converted to 0 for compliance with column based plotting tools.

    Args:
        path (str): Path to output file.

        interface (ndarray): Interface data.

        fngroup (list): List of data files read from.

    """

    def get_header(path, fngroup):
        from strata.strata import version
        import time

        time_str = time.strftime('%c', time.localtime())

        header = (
                "# Interface coordinates of a droplet\n"
                "# \n"
                "# Created by module: %s\n"
                "# Creation date: %s\n"
                "# Using module version: %s\n"
                "# \n"
                % (__name__, time_str, version))

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
    data = np.array([interface['X'], interface['Y']]).T
    np.savetxt(path, data, fmt='%.3f', delimiter=' ',
            header=header, comments='')

import os
import numpy as np

from droplets.flow import FlowData
from droplets.droplet import get_interface
from strata.dataformats.read import read_from_files
from strata.utils import *

"""Module for finding the spreading of a droplet.

Analyses groups of data files and outputs wetting radius as a
function of time.

"""


def collect(base, **kwargs):
    """Return the spreading radius of a droplet as a function of time.

    Droplets are contained in data map files at input base. The spreading
    radius is found as the outermost filled bins of the
    bottom interface layer as found by 'droplets.droplet.get_interface'.
    The accumulated mass inside bins is used as the parameter determining
    whether a system bin is full or not.

    The time of spreading is zeroed at impact, which is taken as the
    first data map for which a spreading is found. Set a 'floor' to only
    accept spreading at a certain system height, ignoring files before
    the droplet reaches the impacted substrate.

    Other options are detailed below.

    Args:
        base (str): Base path to input files.

    Keyword Args:
        output (str): Write spreading data to an output file.

        dt (float): Time difference between input maps.

        floor (float): Height at which spreading occurs. Defaults to the
            bottom interface bins found in the data maps.

        cutoff (float): Which mass value to cut the boundary at.
            Defaults to the midpoint mass.

        include_radius (float, default=1): Radius to include bins within.

        num_bins (int, default=1): Number of bins inside the set radius
            which must pass the cut-off criteria.

        begin (int, default=1): First data map number.

        end (int, default=inf): Final data map number.

        ext (str, default='.dat'): File extension.

    Returns:
        ndarray: 2D array with time and radius as elements.

    """

    def prepare_output(output, header_opts, fopts):
        header_opts.update(fopts)

        try:
            write_header(output, base, header_opts)
        except PermissionError:
            print("[WARNING] Output disabled: could not open '%s' for writing."
                    % output)
            output = None

        return output

    def get_spreading_ndarray(times, radii):
        dtype=[('t', 'float32'), ('r', 'float32')]
        data = np.zeros(len(times), dtype=dtype)

        data['t'] = times
        data['r'] = radii

        return data

    fopts = pop_fileopts(kwargs)

    output = kwargs.pop('output', None)
    if output != None:
        output = prepare_output(output, kwargs.copy(), fopts)

    time = 0
    dt = kwargs.pop('dt', 1)
    include_radius = kwargs.pop('include_radius', 1)

    times = []
    radii = []

    files = list(find_datamap_files(base, **fopts))
    for data, _, meta in read_from_files(*files):
        flow = FlowData(data)
        left, right = get_spreading_edges(flow, 'M', include_radius, **kwargs)

        if left != None and right != None:
            radius = 0.5*(right - left)

            radii.append(radius)
            times.append(time)

            if output != None:
                write_spreading(output, time, radius, meta.pop('path'))

            time += dt

    return get_spreading_ndarray(times, radii)


def get_spreading_edges(flow, label, include_radius, **kwargs):
    """Return the left and right edges of wetting.

    Args:
        flow (FlowData): A FlowData object. Must contain a data record
            with coordinates and selected height map label, as well
            as an input shape of the system.

        label (str): Record label used as base for the interface height map.

        include_radius (float): Radius to include bins within.

    Keyword Args:
        floor (float): Height at which spreading occurs. Defaults to the
            bottom interface bins found in the data map.

        cutoff (float): Which interface height to cut the boundary at.
            Defaults to the midpoint height.

        num_bins (int, default=1): Number of bins inside the set radius
            which must pass the cut-off criteria.

        coord_labels (2-tuple, default=('X', 'Y'): Record labels for coordinates.

    Returns:
        float, float: 2-tuple with left and right edges.

    """

    def get_floor_height(floor, ys):
        if floor != None:
            return ys[np.abs(np.ceil(ys - floor) - 1).argmin()]

    xs, ys = (flow.data[l] for l in kwargs.get('coord_labels', ('X', 'Y')))

    floor = kwargs.pop('floor', None)
    yfloor = get_floor_height(floor, ys)
    kwargs['ylims'] = (yfloor, yfloor)

    interface = get_interface(flow, label, include_radius, **kwargs)

    while True:
        try:
            ileft, iright = next(interface)
            ylayer = ys[ileft]
        except StopIteration:
            return None, None
        else:
            if floor == None or ylayer == yfloor:
                break

    return xs[ileft], xs[iright]


# Output helper functions
@prepare_path
def write_header(output_path, input_base, kwargs):
    """Verify that output path is writable and write header."""

    import pkg_resources
    import time

    with open(output_path, 'w') as fp:
        time_str = time.strftime('%c', time.localtime())
        version_str = pkg_resources.require("flowfield")[0].version

        header = (
                "# Spreading radius of a droplet impacting a substrate\n"
                "# \n"
                "# Creation date: %s\n"
                "# Using module version: %s\n"
                "# \n"
                % (time_str, version_str))

        inputs = (
                "# Inputs:\n"
                "#   File base path: %r\n"
                "#   Begin, end: %r, %r\n"
                "#   Floor: %r\n"
                "#   Delta-t: %r\n"
                "#   Mass cut-off: %r\n"
                "#   Radius cut-off: %r\n"
                "#   Required # of bins: %r\n"
                "# \n"
                % (os.path.realpath(input_base),
                    kwargs.get('begin', None), kwargs.get('end', None),
                    kwargs.get('floor', None), kwargs.get('dt', 1.),
                    kwargs.get('cutoff', None), kwargs.get('include_radius', 1.),
                    kwargs.get('num_bins', 1)))

        fp.write(header + inputs)


@static_variable('impact', False)
def write_spreading(output_path, time, radius, cur_filename):
    """Write time and spreading radius to output file."""

    def output_impact_time(fp, time, impact_path):
        """Write impact time and column header."""

        _, filename = os.path.split(impact_path)
        impact_comment = (
                "# Droplet impact at t = %.3f\n"
                "#                file = '%s'\n"
                "# \n"
                "# Time (ps) Radius (nm)\n" % (time, filename)
                )
        fp.write(impact_comment)

    with open(output_path, 'a') as fp:
        if not write_spreading.impact:
            output_impact_time(fp, time, cur_filename)
            write_spreading.impact = True

        fp.write('%.3f %.3f\n' % (time, radius))

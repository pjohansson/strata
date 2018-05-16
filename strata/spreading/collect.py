import os
import numpy as np
import progressbar as pbar

from collections import namedtuple

from droplets.flow import FlowData
from droplets.interface import get_interface
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
    bottom interface layer as found by 'droplets.interface.get_interface'.
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
        save (str): Write spreading data to an output file.

        dt (float): Time difference between input maps.

        floor (float): Height at which spreading occurs. Defaults to the
            bottom interface bins found in the data maps.

        cutoff (float): Which mass value to cut the boundary at.
            Defaults to the midpoint mass.

        cutoff_radius (float, default=1): Radius to include bins within.

        cutoff_bins (int, default=1): Number of bins inside the set radius
            which must pass the cut-off criteria.

        begin (int, default=1): First data map number.

        end (int, default=inf): Final data map number.

        ext (str, default='.dat'): File extension.

        quiet (bool, default=False): Do not print progress.

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

    save = kwargs.pop('save', None)
    if save != None:
        save = prepare_output(save, kwargs.copy(), fopts)

    quiet = kwargs.pop('quiet', False)

    dt = kwargs.pop('dt', 1.)
    time = kwargs.pop('t0', 0.)
    cutoff_radius = kwargs.pop('cutoff_radius', 1.)

    times = []
    values = []

    files = list(find_datamap_files(base, **fopts))

    if not quiet:
        widgets = ['Reading files: ',
                pbar.Bar(), ' (', pbar.SimpleProgress(), ') ', pbar.ETA()]
        progress = pbar.ProgressBar(widgets=widgets, maxval=len(files))
        progress.start()

    pbc_info = init_periodic_info()

    for i, (data, info, meta) in enumerate(read_from_files(*files)):
        flow = FlowData(data, info=info)
        left, right = get_spreading_edges(flow, 'M', cutoff_radius,
            search_longest_connected=True, **kwargs)

        if left != None and right != None:
            box_x, _ = flow.size()

            pbc_info = check_and_update_periodic_info(
                pbc_info, box_x, left, right
            )

            left_absolute, right_absolute = add_pbc_multipliers_to_edges(
                pbc_info, box_x, left, right
            )

            radius = abs(0.5*(right_absolute - left_absolute))

            values.append(radius)
            times.append(time)

            if save != None:
                cur_path = meta.pop('path')

                if not write_spreading.impact:
                    output_impact_time(save, i*dt, cur_path)
                    write_spreading.impact = True

                write_spreading(
                    save, time, radius, left_absolute, right_absolute, cur_path
                )

            time += dt

        if not quiet:
            progress.update(i+1)

    if not quiet:
        progress.finish()

    return get_spreading_ndarray(times, values)


def get_spreading_edges(flow, label, cutoff_radius, **kwargs):
    """Return the left and right edges of wetting.

    Args:
        flow (FlowData): A FlowData object. Must contain a data record
            with coordinates and selected height map label, as well
            as an input shape of the system.

        label (str): Record label used as base for the interface height map.

        cutoff_radius (float): Radius to include bins within.

    Keyword Args:
        floor (float): Height at which spreading occurs. Defaults to the
            bottom interface bins found in the data map.

        cutoff (float): Which interface height to cut the boundary at.
            Defaults to the midpoint height.

        cutoff_bins (int, default=1): Number of bins inside the set radius
            which must pass the cut-off criteria.

        search_longest_connected (bool, default=False): Instead of searching
            for the interface from out and in, look for the longest stretch
            of filled cells and take the edges as the edges of those.
            See `get_interface` for a description of this method.

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

    interface = get_interface(flow, label, cutoff_radius=cutoff_radius, **kwargs)

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


PeriodicInfo = namedtuple(
    "PeriodicInfo",
    ["xleft_prev", "xright_prev", "pbc_multiplier_left", "pbc_multiplier_right"]
)

def init_periodic_info(xleft=None, xright=None):
    """Initialize a `PeriodicInfo` object with no previous positions
    and zeroed multipliers.

    """
    return PeriodicInfo(
            xleft_prev=xleft,
            xright_prev=xright,
            pbc_multiplier_left=0,
            pbc_multiplier_right=0
        )

def check_and_update_periodic_info(pbc_info, box_x,
        xleft_current, xright_current):
    """Detect whether an edge has crossed a boundary and update the multipliers.

    If a PBC boundary has been crossed by one edge (assuming that the time
    difference between maps is small enough to not allow both edges to cross
    at once), the edge that crossed should be the one which is furthest from
    its previous position.

    The boundary that it crossed is that which is furthest away from its new
    position, by which we can infer how to adjust box-relative position to
    the system-absolute.

    Note that this function assumes that at all times, the system-absolute
    position of the left edge is smaller (along x) than the system-absolute
    position of the right edge. The exception is for the first step, when
    no previous positions have been set. If that is the case and the left
    position is ahead of the right edge, it is assumed that the right edge
    has alread crossed the edge once: that multiplier is increased by 1.

    Args:
        pbc_info (PeriodicInfo): The previous state of the periodic multiplers
            and edge positions.

        box_x (float): The system size along x.

        xleft_current, xright_current (float): The box-relative position of
            the left and right edges.

    Returns:
        PeriodicInfo: The new state of the periodic multipliers
            and edge positions.

    """

    def get_pbc_edge_multiplier_addition(x_current, box_x):
        """Return -1 if the left edge was crossed, +1 if the right edge."""

        if x_current % box_x > abs(box_x - x_current % box_x):
            return -1
        else:
            return +1

    def detect_crossing_event(xleft_current, xright_current,
            xleft_prev, xright_prev):
        """Detect whether an edge was crossed since the last state."""

        left_ahead_current = xleft_current > xright_current
        left_ahead_prev = xleft_prev > xright_prev

        return (left_ahead_current and (not left_ahead_prev)) \
            or ((not left_ahead_current) and left_ahead_prev)

    pbc_multiplier_left = pbc_info.pbc_multiplier_left
    pbc_multiplier_right = pbc_info.pbc_multiplier_right

    xleft = xleft_current + pbc_multiplier_left * box_x
    xright = xright_current + pbc_multiplier_right * box_x

    try:
        crossing_event = detect_crossing_event(xleft_current, xright_current,
            pbc_info.xleft_prev, pbc_info.xright_prev)
    # Catch if we are in the state where no previous positions are set:
    # If so, ignore the regular PBC check and only check whether the left
    # edge is ahead of the right. If that is true, add to the right multiplier.
    except TypeError:
        if xleft_current > xright_current:
            pbc_multiplier_right += 1
    else:
        if crossing_event:
            dx_left = abs(xleft_current - pbc_info.xleft_prev)
            dx_right = abs(xright_current - pbc_info.xright_prev)

            if dx_left > dx_right:
                pbc_multiplier_left += get_pbc_edge_multiplier_addition(
                    xleft, box_x
                )
            else:
                pbc_multiplier_right += get_pbc_edge_multiplier_addition(
                    xright, box_x
                )

    pbc_info = PeriodicInfo(
        xleft_prev=xleft_current,
        xright_prev=xright_current,
        pbc_multiplier_left=pbc_multiplier_left,
        pbc_multiplier_right=pbc_multiplier_right
    )

    return pbc_info

def add_pbc_multipliers_to_edges(pbc_info, box_x, xleft, xright):
    """Adjust the input edge coordinates to their system absolute positions."""

    return (
            xleft + pbc_info.pbc_multiplier_left * box_x,
            xright + pbc_info.pbc_multiplier_right * box_x
        )

# Output helper functions
@prepare_path
def write_header(output_path, input_base, kwargs):
    """Verify that output path is writable and write header."""

    import strata
    import time

    with open(output_path, 'w') as fp:
        time_str = time.strftime('%c', time.localtime())

        header = (
                "# Spreading radius of a droplet impacting a substrate\n"
                "# \n"
                "# Creation date: %s\n"
                "# Using module version: %s\n"
                "# \n"
                % (time_str, strata.strata.version))

        inputs = (
                "# Input:\n"
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
                    kwargs.get('cutoff_bins', 1)))

        fp.write(header + inputs)


def output_impact_time(output_path, time, impact_path):
    """Write impact time and column header."""

    _, filename = os.path.split(impact_path)
    impact_comment = (
            "# Droplet impact:\n"
            "#   Time: %.3f ps\n"
            "#   File: '%s'\n"
            "# \n" % (time, filename)
            )
    legend = "# Time (ps) Radius (nm) Left (nm) Right (nm)\n"

    with open(output_path, 'a') as fp:
        _, filename = os.path.split(impact_path)
        fp.write(impact_comment)
        fp.write(legend)


@static_variable('impact', False)
def write_spreading(output_path, time, radius, left, right, cur_filename):
    """Write time and spreading radius to output file."""

    with open(output_path, 'a') as fp:
        fp.write('%.3f %.3f %.3f %.3f\n' % (time, radius, left, right))

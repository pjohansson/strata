import numpy as np
from droplets.flow import FlowData
from droplets.droplet import get_interface
from strata.utils import find_datamap_files, pop_fileopts, prepare_path
from strata.dataformats.read import read_from_files

"""Module for finding the spreading of a droplet.

Analyses groups of data files and outputs wetting radius as a
function of time.

"""


def spreading(base, **kwargs):
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

    fopts = pop_fileopts(kwargs)

    output = kwargs.pop('output', None)
    if output != None:
        output = prepare_output(output)
        impact = False

    time = 0
    dt = kwargs.pop('dt', 1)
    include_radius = kwargs.pop('include_radius', 1)

    times = []
    spreading_radius = []
    files = list(find_datamap_files(base, **fopts))
    for i, (data, _, meta) in enumerate(read_from_files(*files)):
        flow = FlowData(data)
        left, right = get_spreading_edges(flow, 'M', include_radius, **kwargs)

        if left != None and right != None:
            radius = 0.5*(right - left)
            spreading_radius.append(radius)
            times.append(time)

            if output != None:
                if not impact:
                    output_impact_time(output, i*dt, meta.pop('path'))
                    impact = True
                output_spreading(output, time, radius)

            time += dt

    dtype=[('t', 'float32'), ('r', 'float32')]
    spreading_data = np.zeros(len(times), dtype=dtype)
    spreading_data['t'] = times
    spreading_data['r'] = spreading_radius

    return spreading_data


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
def prepare_output(output):
    """Verify that output path is writable and write header."""
    try:
        prepare_path(output)
        fp = open(output, 'w')
    except PermissionError:
        warnings.warn("Output disabled: could not open '%s' for writing"
                % output, UserWarning)
        output = None
    else:
        header = "# Spreading radius of a droplet impacting a substrate\n"

        fp.write(header)
        fp.close()

    return output


def output_impact_time(path, time, fn):
    """Write impact time and column header."""
    with open(path, 'a') as fp:
        impact_comment = (
                "# Droplet impact at t = %.3f\n"
                "# ('%s')\n"
                "#\n"
                "# Time (ps) Radius (nm)\n" % (time, fn)
                )
        fp.write(impact_comment)


def output_spreading(path, time, radius):
    """Write time and spreading radius."""
    with open(path, 'a') as fp:
        fp.write('%.3f %.3f\n' % (time, radius))

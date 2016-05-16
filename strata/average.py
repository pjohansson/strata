import numpy as np
import progressbar as pbar

from droplets.flow import FlowData

from strata.dataformats.read import read_from_files
from strata.dataformats.write import write
from strata.spreading.collect import get_spreading_edges
from strata.utils import find_groups_to_singles, pop_fileopts


def average(base, output, group=1, **kwargs):
    """Average and output data maps from input base.

    Joins the file bases and extensions with a five-digit integer
    signifying the file number ('%s%05d%s').

    Output numbering of files will largely be compensated by the input
    frame numbering as K = ceil(N/M) where N is the first number of
    the current bundle and M is the bundle size.

    By supplying the keyword argument `recenter` the data maps can be
    recentered to the left or right contact lines before calculating
    the average. This will partly compensate for the movement of the
    contact line if quantities around it is to be analysed. Keep in
    mind that the geometry of the contact line will not be affected
    by this recentering.

    The determination of the contact line is by the same method as
    in `strata spreading collect`. Options for the spreading floor
    and cutoffs are input as per that.

    Args:
        base (str): Base path to input files.

        output (str): Base path to output files.

    Keyword Args:
        recenter (str, optional): Recenter data at the input contact line,
            'left' or 'right'.

        floor (float): Height at which spreading occurs. Defaults to the
            bottom interface bins found in the data map.

        cutoff (float): Which interface height to cut the boundary at.
            Defaults to the midpoint height.

        cutoff_bins (int, default=1): Number of bins inside the set radius
            which must pass the cut-off criteria.

        cutoff_radius (float, default=1.): Radius to include bins within.

        begin (int, default=1): First data map number.

        end (int, default=inf): Final data map number.

        group (int, default=1): Group input files in bundles of this length.
            Can be input as the third positional argument.

        ext (str, default='.dat'): File extension.

        quiet (bool, default=False): Do not print progress.

    """

    fopts = pop_fileopts(kwargs)
    quiet = kwargs.pop('quiet', False)
    recenter = kwargs.pop('recenter', False)
    cutoff_radius = kwargs.pop('cutoff_radius', 1.)

    groups_singles = list(find_groups_to_singles(base, output, group, **fopts))

    if not quiet:
        widgets = ['Averaging files: ',
                pbar.Bar(), ' (', pbar.SimpleProgress(), ') ', pbar.ETA()]
        progress = pbar.ProgressBar(widgets=widgets, maxval=len(groups_singles))
        progress.start()

    for i, (fn_group, fn_out) in enumerate(groups_singles):
        group_data = []
        used_modules = set([])

        for data, _, meta in read_from_files(*fn_group):
            group_data.append(data)
            used_modules.add(meta.pop('module'))

        # Assert that a single module was used and retrieve it
        assert (len(used_modules) == 1)
        module = used_modules.pop()

        # Optionally recenter the data maps at the contact line
        if recenter:
            flow_data = [FlowData(data) for data in group_data]

            if recenter == 'right':
                xs_edges = [edge for _, edge in (
                        get_spreading_edges(data, 'M', cutoff_radius, **kwargs)
                        for data in flow_data
                    )
                ]
            elif recenter == 'left':
                xs_edges = [edge for edge, _ in (
                        get_spreading_edges(data, 'M', cutoff_radius, **kwargs)
                        for data in flow_data
                    )
                ]
            else:
                raise ValueError("Invalid edge to recenter around ('%s'). Must be 'left' or 'right'" % recenter)


            group_data = recenter_maps(group_data, xs_edges)

        avg_data = module.average_data(*group_data)
        write(fn_out, avg_data)

        if not quiet:
            progress.update(i+1)

    progress.finish()


def recenter_maps(data_maps, recenter_values):
    """Recenter input data around x values and return the intersection."""

    try:
        assert(len(data_maps) == len(recenter_values))
    except AssertionError:
        raise IndexError("The number of input maps (%d) is not equal to the number of input x values (%d)"
            % (len(data_maps), len(recenter_values)))

    xs_trans = np.zeros((len(data_maps), data_maps[0]['X'].size))

    # Collect recentered data in array
    try:
        for i, (data, recenter) in enumerate(zip(data_maps, recenter_values)):
            xs_trans[i] = data['X'] - recenter
    except TypeError:
        raise TypeError("Bad input values for recentering")

    # Find xmin/xmax of the intersection of all sets
    xmin = np.max(np.min(xs_trans, axis=1))
    xmax = np.min(np.max(xs_trans, axis=1))

    recentered_data = []

    for xs, data in zip(xs_trans, data_maps):
        # Get indices of subset for current data map
        inds = np.greater_equal(xs, xmin) & np.less_equal(xs, xmax)

        # Get the subset data and add translated x array
        data_subset = {l: data[l][inds] for l in data.keys() - 'X'}
        data_subset['X'] = xs[inds]

        recentered_data.append(data_subset)

    return recentered_data


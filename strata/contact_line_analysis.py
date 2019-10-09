import os
import progressbar as pbar
import sys

from droplets.average import average_flow_data, get_combined_grid, transfer_data
from droplets.contact_line import *
from droplets.flow import FlowData
from strata.dataformats.read import read_data_file
from strata.dataformats.write import write
from strata.sample_average import sample_value
from strata.utils import gen_filenames, find_datamap_files, pop_fileopts, prepare_path, write_module_header


def extract_contact_line_bins(base, output, average=1,
                              fuse=False, rolling=False,
                              recenter=False, **kwargs):
    """Extract bins from the contact line of a wetting system.

    Can average the extracted data by supplying a positive number of
    files to perform the average over.

    Args:
        base (str): Base path to input files.

        output (str): Save extracted bins to files with
            this base path.

    Keyword Args:
        average (int, optional): Number of files to average data over.

        rolling (bool, optional): Perform a rolling average over the data.

        recenter (bool, optional): Recenter the contact line edges around zero.

        fuse (bool, optional): Fuse the contact lines.

        begin (int, default=1): First data map number.

        end (int, default=inf): Final data map number.

        ext (str, default='.dat'): File extension.

        quiet (bool, default=False): Do not print progress.

    See `droplets.contact_line` for additional keyword arguments.

    """

    fopts = pop_fileopts(kwargs)

    kwargs['cutoff_label'] = 'M'
    weights = [('U', 'M'), ('V', 'M'), ('T', 'N')]

    filenames = list(find_datamap_files(base, **fopts))
    fnout = gen_filenames(output, **fopts)

    # An input floor should be used which is set by the `ylims` which
    # is in the end passed to the interface search function
    floor = kwargs.pop('floor', None)
    kwargs['ylims'] = (floor, None)

    averaged_data = get_averaged_contact_line_edges(filenames, average,
        rolling, recenter, weights, **kwargs)

    for avg_flow_per_edge in averaged_data:
        spacing = avg_flow_per_edge[0].spacing

        if fuse:
            fuse_edges(avg_flow_per_edge, spacing)

        recombined_flow_data = combine_flow_data(avg_flow_per_edge, spacing)
        write(next(fnout), recombined_flow_data.data)


"""Fuse the left and right edges in the center."""
def fuse_edges(flow_edges, spacing):
    dx, _ = spacing
    left, right = flow_edges

    left_xmax = np.max(left.data['X'])
    right_xmin = np.min(right.data['X'])

    left_xshift = left_xmax + 0.5 * dx
    right_xshift = right_xmin - 0.5 * dx

    left.data['X'] -= left_xshift
    right.data['X'] -= right_xshift


def sample_contact_line_edges(base, labels, save=None,
                              average=1, rolling=False, recenter=False,
                              dt=1., sum=False, viscosity=8.44e-4,
                              **kwargs):
    """Sample data of bins from the contact line of a wetting system.

    Can average the extracted data by supplying a positive number of
    files to perform the average over.

    See `strata.sample_average.sample_value` for details on how the sampling
    is performed. For the case of sampling the velocities along X are radial,
    meaning that the left edge has its velocity along X reversed before sampling.

    Args:
        base (str): Base path to input files.

        labels (str's): Labels to sample data of. Can be a set.

    Keyword Args:
        average (int, optional): Number of files to average data over.

        rolling (bool, optional): Perform a rolling average over the data.

        recenter (bool, optional): Recenter the contact line edges around zero.

        dt (float, optional): Time difference between data files.

        save (str, optional): Save the data as an xmgrace formatted file.

        begin (int, default=1): First data map number.

        end (int, default=inf): Final data map number.

        ext (str, default='.dat'): File extension.

        quiet (bool, default=False): Do not print progress.

    See `droplets.contact_line` for additional keyword arguments.

    """

    def prepare_output(output, labels, cutoff, cutoff_label, average, rolling, header_opts, fopts):
        header_opts.update(fopts)
        write_header(output, base, labels, cutoff, cutoff_label, average, rolling, header_opts)

    fopts = pop_fileopts(kwargs)

    cutoff_label = 'M'
    cutoff = kwargs.get('cutoff', None)

    kwargs['cutoff_label'] = cutoff_label
    weights = [('U', 'M'), ('V', 'M'), ('T', 'N')]

    if save:
        try:
            prepare_output(save, labels, cutoff, cutoff_label, average, rolling, kwargs.copy(), fopts)
        except PermissionError:
            print("[WARNING] Output disabled: could not open '%s' for writing."
                % save)
            save = None

    filenames = list(find_datamap_files(base, **fopts))

    # An input floor should be used which is set by the `ylims` which
    # is in the end passed to the interface search function
    floor = kwargs.pop('floor', None)
    kwargs['ylims'] = (floor, None)

    averaged_data = get_averaged_contact_line_edges(filenames, average,
        rolling, recenter, weights, **kwargs)

    samples = [[] for _ in labels]
    times = []

    # TODO: Sometime fix this.
    if not sum:
        print("warning: the output value is the simple (unweighed) average "
              "of the sampled value for each contact line edge. If "
              "the sampled data is of a type that should be weighed when "
              "averaged (eg. temperature, mass flow) the returned value will "
              "not be correct, unless the two edges are very similar.")

    for i, (left_edge, right_edge) in enumerate(averaged_data):
        # Since we are sampling radial flow the left edge velocities
        # are mirrored along X
        left_edge.data['U'] *= -1

        for j, label in enumerate(labels):
            try:
                means_and_stds = [
                        sample_value(flow, label, cutoff, cutoff_label,
                            sum, viscosity)
                        for flow in (left_edge, right_edge)
                    ]

                means = [t[0] for t in means_and_stds]
                # stds = [t[1] for t in means_and_stds]

                if sum:
                    value = np.sum(means)
                else:
                    value = np.mean(means)

                # try:
                    # std = np.mean([t[1] for t in means_and_stds])
                # except:
                    # std = None

                samples[j].append([value, std])
            except Exception as exc:
                print(
                        "error: could not add value for averaged map no. {}, "
                        "data label '{}' ({!s})".format(i + 1, label, exc)
                    )

                sys.exit(1)

        if rolling:
            time = (np.floor(0.5 * average) + i) * dt
        else:
            time = (np.floor(0.5 * average) + i * average) * dt

        times.append(time)

        if save:
            with open(save, 'a') as fp:
                fp.write("{:.3f} ".format(times[-1]))
                try:
                    sampled_values = []
                    for label_samples in samples:
                        value, std = label_samples[-1]
                        sampled_values.append(value)

                    fp.write(' '.join(
                            ['{:g}'.format(v) for v in sampled_values]
                        ))

                except TypeError:
                    sys.stderr.write("\nError: Some sampled value was not calculated correctly, ie. a 'None'\n")
                    sys.stderr.write("or similar value was encountered. Aborting.\n")
                    sys.exit(1)
                fp.write('\n')

    return times, samples


def get_averaged_contact_line_edges(filenames, average, rolling,
        recenter, weights, **kwargs):
    """Read data from filebanes and yield averaged edges as `EdgeData` objects.

    """

    def adjust_coordinates(avg_flow, xadj_per_edge, dx, yadj, recenter):
        """Adjust the extracted cell coordinates."""

        if recenter:
            xdiff = 0.5 * (xadj_per_edge[1] - xadj_per_edge[0])
            xadj_per_edge[0] = -xdiff
            xadj_per_edge[1] = xdiff

        for i, xadj in enumerate(xadj_per_edge):
            xadj_on_grid = get_coord_on_grid(xadj, dx)
            avg_flow[i].data['X'] += xadj_on_grid
            avg_flow[i].data['Y'] += yadj

        return avg_flow

    def get_coord_on_grid(x, dx):
        """Return input coordinate x on grid of difference dx."""

        return (np.floor(x / dx) + 0.5) * dx

    def estimate_shape(data):
        """Estimate the shape of the contact line grid."""

        nx = len(np.unique(data['X']))
        ny = len(np.unique(data['Y']))

        return (nx, ny)

    quiet = kwargs.pop('quiet', False)
    if not quiet:
        if rolling:
            length = len(filenames) - average + 1
        else:
            length = int(len(filenames) / average)

        widgets = ['Extracting/sampling contact line: ',
                pbar.Bar(), ' (', pbar.SimpleProgress(), ') ', pbar.ETA()]
        progress = pbar.ProgressBar(widgets=widgets, max_value=length)
        progress.start()

    # Get a generator which reads data to average over.
    #
    # The data is yielded as a somewhat complex object:
    # grouped_data = (spacing, left, right) where
    # spacing is the read spacing along x and y for the data
    # and left/right are lists of the averaging length, where
    # each element is in itself a tuple.
    #
    # These tuples contain the flow data for each map, which has
    # been moved so that the innermost edge of the extracted contact
    # line area (from the interface) is placed at x, y = 0, 0.
    # The other part of the tuple is the original coordinates so that
    # we can later on move the data to the original positions after
    # the averaging is finished.
    #
    # This structure is a bit too complex and should be fixed.

    grouped_data = get_grouped_data(filenames, average, rolling,
            progress, quiet, **kwargs)

    for spacing, left, right in grouped_data:
        avg_flow_per_edge = []
        xadj_per_edge = []

        for data_list in [left, right]:
            coord_adjs, flow_data = np.array(data_list).T.tolist()

            # Get mean adjusting x coordinate of edge
            xadj_mean = np.mean(coord_adjs, axis=0)[0]
            xadj_per_edge.append(xadj_mean)

            avg_flow_per_edge.append(
                average_flow_data(
                    flow_data, weights=weights, exclude_empty_sets=True
                )
            )

        yadj = get_coord_on_grid(0, spacing[1])
        avg_flow_per_edge = adjust_coordinates(avg_flow_per_edge,
                xadj_per_edge, spacing[0], yadj, recenter)

        yield avg_flow_per_edge

    if not quiet:
        progress.finish()


def get_grouped_data(fns, average, rolling, progress, quiet, **kwargs):
    """Generate contact line data from list of input files."""

    label = kwargs.pop('cutoff_label')
    left, right = [], []

    j = 0
    for fn in fns:
        data, info, _ = read_data_file(fn)

        left_bins, right_bins = get_contact_line_cells(
            FlowData(data), label, **kwargs
        )

        left.append(add_adjusted_flow(left_bins, 'left', info))
        right.append(add_adjusted_flow(right_bins, 'right', info))

        if len(left) == average:
            if not quiet:
                progress.update(j + 1)
                j += 1

            yield info['spacing'], left, right

            if rolling == True:
                left.pop(0)
                right.pop(0)
            else:
                left, right = [], []


def combine_flow_data(avg_flow, spacing):
    """Return a combined FlowData object of the left and right edges."""

    def merge_data(grid, left, right, spacing):
        """Merge the data from averaged edges onto a grid."""

        weights = [('U', 'M'), ('V', 'M'), ('T', 'N')]

        weighted_labels = [l for l, _ in weights]
        nonweighted_labels = set(grid.dtype.names) - set(weighted_labels)

        # The grid and input left, right edges are sorted
        # in y-major, x-minor order. Average each bin:
        for bin, left_bin, right_bin in zip(
            grid.ravel(), left.ravel(), right.ravel()
        ):
            for (l, w) in weights:
                if (left_bin[l] != 0.0) and (right_bin[l] != 0.0):
                    total_weight = left_bin[w] + right_bin[w]

                    if total_weight != 0.0:
                        bin[l] = (
                            left_bin[w] * left_bin[l]
                            + right_bin[w] * right_bin[l]
                            ) / total_weight

                elif left_bin[l] != 0.0:
                    bin[l] = left_bin[l]
                elif right_bin[l] != 0.0:
                    bin[l] = right_bin[l]

            for l in nonweighted_labels:
                if (left_bin[l] != 0.0) and (right_bin[l] != 0.0):
                    bin[l] = 0.5 * (left_bin[l] + right_bin[l])

                elif left_bin[l] != 0.0:
                    bin[l] = left_bin[l]
                elif right_bin[l] != 0.0:
                    bin[l] = right_bin[l]

        dx, dy = spacing

        grid['X'] -= 0.5 * dx
        grid['Y'] -= 0.5 * dy

        grid.sort(order=['X', 'Y'])

        return [(l, grid[l]) for l in grid.dtype.names]

    left, right = avg_flow

    data = [flow.data for flow in avg_flow]
    grid = get_combined_grid(data, spacing)

    xs = grid.ravel()['X']
    ys = grid.ravel()['Y']

    left, right = (
        transfer_data(grid, flow.data, flow.shape, spacing)
        for flow in avg_flow
    )

    combined_data = merge_data(grid, left, right, spacing)

    ny, nx = grid.shape

    return FlowData(*combined_data, info={'spacing': spacing, 'shape': (nx, ny)})


def add_adjusted_flow(bins, direction, info):
    """Return adjusted FlowData object.

    The `info` object is adjusted to the final shape of the contact line bins.

    """

    def get_cut_shape(data, dx, dy):
        # After sorting, the data is in y-major, x-minor order
        ys = data['Y']
        y0 = ys[0]
        nx = 1

        while (ys[nx] >= y0 - 0.5 * dy) and (ys[nx] < y0 + 0.5 * dx):
            nx += 1

        ny = data.size // nx

        return nx, ny

    dx, dy = info['spacing']

    xadj, data = adjust_cell_coordinates(bins, direction)

    adj_data = [(l, data[l]) for l in bins.dtype.names]

    flow = FlowData(*adj_data, info=info)
    flow.sort()

    nx, ny = get_cut_shape(flow.data, dx, dy)

    flow.origin = xadj
    flow.shape = (nx, ny)
    flow.num_bins = nx * ny

    return xadj, flow


@prepare_path
def write_header(output_path, input_base, labels, cutoff, cutoff_label, average, rolling, kwargs):
    """Verify that output path is writable and write header."""

    title = "Sample average of contact line data from a simulation"
    write_module_header(output_path, __name__, title)

    with open(output_path, 'a') as fp:
        inputs = (
                "# Input:\n"
                "#   Sample data labels: %r\n"
                "#   Average over # files: %r\n"
                "#   Rolling average: %s\n"
                "#   File base path: %r\n"
                "#   Begin, end: %r, %r\n"
                "#   Delta-t: %r\n"
                "#   Cut-off: %r\n"
                "#   Cut-off label: %r\n"
                "# \n"
                "# Time (ps) %s\n"
                % (labels, average, rolling, os.path.realpath(input_base),
                    kwargs.get('begin', None), kwargs.get('end', None),
                    kwargs.get('dt', 1.), cutoff,
                    cutoff_label, ' '.join(labels)
                    ))

        fp.write(inputs)

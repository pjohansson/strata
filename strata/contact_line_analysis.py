import os
import progressbar as pbar

from droplets.average import average_flow_data, get_combined_grid, transfer_data
from droplets.contact_line import *
from droplets.flow import FlowData
from strata.dataformats.read import read_data_file
from strata.dataformats.write import write
from strata.utils import gen_filenames, find_datamap_files, pop_fileopts


def extract_contact_line_bins(base, output, average=1, rolling=False,
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

        begin (int, default=1): First data map number.

        end (int, default=inf): Final data map number.

        ext (str, default='.dat'): File extension.

        quiet (bool, default=False): Do not print progress.

    See `droplets.contact_line` for additional keyword arguments.

    """

    def adjust_coordinates(avg_flow, xadj, recenter):
        """Adjust the extracted cell coordinates."""

        if recenter:
            xdiff = 0.5*(xadj[1] - xadj[0])
            xadj[0] = -xdiff
            xadj[1] = xdiff

        avg_flow[0].data['X'] += xadj[0]
        avg_flow[1].data['X'] += xadj[1]

        return avg_flow

    def get_closest_adjusting_coord(xs, dx):
        """Find the adjusting coordinate on input grid with size dx."""

        return np.floor(np.mean(xs)/dx)*dx

    fopts = pop_fileopts(kwargs)
    quiet = kwargs.pop('quiet', False)

    kwargs.setdefault('extract_area', (1., 1.))
    kwargs['cutoff_label'] = 'M'
    weights = [('U', 'M'), ('V', 'M'), ('T', 'N')]

    fns = list(find_datamap_files(base, **fopts))
    fnout = gen_filenames(output, **fopts)

    if not quiet:
        if rolling: length = len(fns) - average + 1
        else: length = int(len(fns)/average)
        length = len(fns)

        widgets = ['Extracting contact line: ',
                pbar.Bar(), ' (', pbar.SimpleProgress(), ') ', pbar.ETA()]
        progress = pbar.ProgressBar(widgets=widgets, maxval=length)
        progress.start()

    # Get a generator which reads data to average over
    grouped_data = get_grouped_data(fns, average, rolling,
            progress, quiet, **kwargs)

    for bin_size, left, right in grouped_data:
        avg_flow = []
        xadj = []

        for data_list in [left, right]:
            xadjs, flow_data = np.array(data_list).T.tolist()
            xadj.append(get_closest_adjusting_coord(xadjs, bin_size[0]))
            avg_flow.append(average_flow_data(flow_data,
                    weights=weights, exclude_empty_sets=True))

        avg_flow = adjust_coordinates(avg_flow, xadj, recenter)
        write(next(fnout), combine_flow_data(avg_flow, bin_size).data)

    if not quiet:
        progress.finish()


def get_grouped_data(fns, average, rolling, progress, quiet, **kwargs):
    """Generate contact line data from list of input files."""

    label = kwargs.pop('cutoff_label')
    left, right = [], []

    for i, fn in enumerate(fns):
        data, info, _ = read_data_file(fn)
        left_cells, right_cells = get_contact_line_cells(
                FlowData(data), label, **kwargs)

        left.append(add_adjusted_flow(left_cells, 'left', info))
        right.append(add_adjusted_flow(right_cells, 'right', info))

        if not quiet:
            progress.update(i+1)
            i += 1

        if len(left) == average:
            yield info['bin_size'], left, right

            if rolling == True:
                left.pop(0)
                right.pop(0)
            else:
                left, right = [], []


def combine_flow_data(avg_flow, bin_size):
    """Return a combined FlowData object of the left and right edges."""

    def merge_data(grid, left, right):
        """Merge the data from averaged edges onto a grid."""

        coord_labels = ('X', 'Y')
        value_labels = set(grid.dtype.names).difference(coord_labels)

        data = np.empty_like(grid)

        for d, l, r in zip(data, left, right):
            for k in coord_labels:
                d[k] = l[k]
            for k in value_labels:
                d[k] = l[k] + r[k]

        return [(k, data[k]) for k in data.dtype.names]

    flowdata = [flow.data for flow in avg_flow]
    grid = get_combined_grid(flowdata, bin_size)
    left, right = (transfer_data(grid, flow.data) for flow in avg_flow)

    data = merge_data(grid, left, right)

    return FlowData(*data, info={'bin_size': bin_size})


def add_adjusted_flow(cells, direction, info):
    """Return adjusted FlowData object."""

    xadj, data = adjust_cell_coordinates(cells, direction)
    adj_data = [(l, data[l]) for l in cells.dtype.names]
    flow = FlowData(*adj_data, info=info)

    return xadj, flow

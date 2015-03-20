import os
import progressbar as pbar

from droplets.average import average_flow_data, get_combined_grid, transfer_data
from droplets.contact_line import *
from droplets.flow import FlowData
from strata.dataformats.read import read_from_files
from strata.dataformats.write import write
from strata.utils import find_groups_to_singles, pop_fileopts


def extract_contact_line_bins(base, output, **kwargs):
    """Extract bins from the contact line of a wetting system.

    Can average the extracted data by supplying a positive number of
    files to perform the average over.

    Args:
        base (str): Base path to input files.

        output (str): Save extracted bins to files with
            this base path.

    Keyword Args:
        average (int, default=1): Number of files to average data over.

        extract_area (float, default=0.): 2-tuple of area around contact
            line to extract.

        cutoff (float): Which mass value to cut the boundary at.
            Defaults to the midpoint mass of each data map.

        include_radius (float, default=1): Radius to include bins within.

        num_bins (int, default=1): Number of bins inside the set radius
            which must pass the cut-off criteria.

        begin (int, default=1): First data map number.

        end (int, default=inf): Final data map number.

        ext (str, default='.dat'): File extension.

        quiet (bool, default=False): Do not print progress.

    """

    def get_closest_adjusting_coord(xs, dx):
        """Find the adjusting coordinate on input grid with size dx."""

        return np.floor(np.mean(xs)/dx)*dx

    fopts = pop_fileopts(kwargs)
    quiet = kwargs.pop('quiet', False)

    average = kwargs.pop('average', 1)
    extract_area = kwargs.pop('extract_area', (0., 0.))
    include_radius = kwargs.pop('include_radius', 1.)

    label = 'M'
    weights = [('U', 'M'), ('V', 'M'), ('T', 'N')]
    groups_singles = list(find_groups_to_singles(base, output, average,
            **fopts))

    if not quiet:
        length = len(groups_singles)*average
        widgets = ['Extracting contact line: ',
                pbar.Bar(), ' (', pbar.SimpleProgress(), ') ', pbar.ETA()]
        progress = pbar.ProgressBar(widgets=widgets, maxval=length)
        progress.start()

    i = 0
    for fn_group, fn_out in groups_singles:
        left, right = [], []

        dxs = []

        for data, info, meta in read_from_files(*fn_group):
            left_cells, right_cells = get_contact_line_cells(
                    FlowData(data), label,
                    size=extract_area, radius=include_radius,
                    cutoff=kwargs['cutoff'], num_bins=kwargs['num_bins']
                    )

            dxs.append(info['bin_size'][0])

            left.append(add_adjusted_flow(left_cells, 'left', info))
            right.append(add_adjusted_flow(right_cells, 'right', info))

            if not quiet:
                progress.update(i+1)
                i += 1

        avg_flow = []
        dx = np.mean(dxs)

        for data_list in [left, right]:
            xadjs, flow_data = np.array(data_list).T.tolist()
            avg_flow.append(average_flow_data(flow_data, weights=weights))

            xadj = get_closest_adjusting_coord(xadjs, dx)
            avg_flow[-1].data['X'] += xadj

        write(fn_out, combine_flow_data(avg_flow, info).data)

    if not quiet:
        progress.finish()


def combine_flow_data(avg_flow, info):
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
    grid = get_combined_grid(flowdata, info['bin_size'])
    left, right = (transfer_data(grid, flow.data) for flow in avg_flow)

    data = merge_data(grid, left, right)

    return FlowData(*data, info=info)


def add_adjusted_flow(cells, direction, info):
    """Return adjusted FlowData object."""

    xadj, data = adjust_cell_coordinates(cells, direction)
    adj_data = [(l, data[l]) for l in cells.dtype.names]
    flow = FlowData(*adj_data, info=info)

    return xadj, flow

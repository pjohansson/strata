import progressbar as pbar

from droplets.average import average_flow_data
from droplets.contact_line import get_contact_line_cells
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

    fopts = pop_fileopts(kwargs)
    quiet = kwargs.pop('quiet', False)

    average = kwargs.pop('average', 1)
    extract_area = kwargs.pop('extract_area', (0., 0.))
    include_radius = kwargs.pop('include_radius', 1.)

    weights = [('U', 'M'), ('V', 'M'), ('T', 'N')]
    groups_singles = list(find_groups_to_singles(base, output, average,
            **fopts))

    if not quiet:
        widgets = ['Extracting contact line: ',
                pbar.Bar(), ' (', pbar.SimpleProgress(), ') ', pbar.ETA()]
        progress = pbar.ProgressBar(widgets=widgets, maxval=len(groups_singles))
        progress.start()

    for i, (fn_group, fn_out) in enumerate(groups_singles):
        group_data = []

        for data, info, meta in read_from_files(*fn_group):
            flow = FlowData(data, info=info)
            contact_line_bins = get_contact_line_cells(flow, 'M',
                    size=extract_area, radius=include_radius)
            group_data.append(bins_to_flowdata(contact_line_bins, info))

        avg_flow = average_flow_data(group_data, weights=weights)
        print('je')
        avg_flow.data.sort(order='X')
        write(fn_out, avg_flow.data)
        print('je')

        if not quiet:
            progress.update(i+1)

    if not quiet:
        progress.finish()


def bins_to_flowdata(bins, info):
    """Return a FlowData object from input cell data."""

    data = [(l, bins[0][l]) for l in bins[0].dtype.names]

    return FlowData(*data, info=info)

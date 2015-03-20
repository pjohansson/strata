import numpy as np
from droplets.flow import FlowData
from strata.dataformats.write import write


"""Module for averaging FlowData objects."""


def average_flow_data(input_flow_maps, weights=[], coord_labels=('X', 'Y')):
    """Average input FlowData objects.

    The input data is projected onto a common coordinate grid before
    being averaged. All input data hence must have bin_size's set
    and identical.

    By default the data is averaged using an arithmetic mean. By inputting
    a list of (label, weight) tuples some data can be averaged instead
    using a weighted arithmetic mean. Here 'weight' refers to the data label
    to use for these weights.

    The coordinate vectors of all input data must be identical.

    Args:
        flow_data (FlowData): List of objects to average.

    Keyword Args:
        weights (label, weight): A list of 2-tuples with labels of data
            and weights to calculate a weighted mean for.

        coord_labels (2-tuple, default=('X', 'Y'): Record labels for coordinates.

    Returns:
        FlowData: Averaged data.

    Raises:
        ValueError: If data with non-matching coordinates are input.

        KeyError: If non-existant labels are input.

    """

    def get_bin_size(flow_maps):
        try:
            for flow in flow_maps[1:]:
                assert (np.isclose(flow.bin_size, flow_maps[0].bin_size).all())
        except AssertionError:
            raise ValueError("Input bin sizes of FlowData objects not identical.")

        return flow_maps[0].bin_size

    def get_flowdata(data, grid, bin_size):
        info = {
                'shape': tuple(len(np.unique(grid[l])) for l in (xl, yl)),
                'num_bins': grid.size,
                'size': [[grid[l][i] for i in (0, -1)] for l in (xl, yl)],
                'bin_size': bin_size
                }

        return FlowData(*[(l, data[l]) for l in data.dtype.names], info=info)

    try:
        bin_size = get_bin_size(input_flow_maps)
        assert bin_size != (None, None)
    except IndexError:
        raise ValueError("No FlowData input.")
    except AssertionError:
        raise ValueError("No bin sizes set in FlowData input.")

    xl, yl = coord_labels
    data_list = [flow.data for flow in input_flow_maps]

    grid = get_combined_grid(data_list, bin_size, coord_labels)
    data_on_grid = [transfer_data(grid, data, coord_labels)
            for data in data_list]

    avg_data = average_data(data_on_grid, weights, coord_labels)

    return get_flowdata(avg_data, grid, bin_size)


def average_data(data_records, weights=[], coord_labels=('X', 'Y')):
    """Return average of input data records.

    By default the data is averaged using an arithmetic mean. By inputting
    a list of (label, weight) tuples some data can be averaged instead
    using a weighted arithmetic mean. Here 'weight' refers to the data label
    to use for these weights.

    The coordinate vectors of all input data must be identical.

    Args:
        data_records (ndarray): List of data records with coordinates
            and values.

    Keyword Args:
        weights (label, weight): A list of 2-tuples with labels of data
            and weights to calculate a weighted mean for.

        coord_labels (2-tuple, default=('X', 'Y'): Record labels for coordinates.

    Returns:
        ndarray: Averaged data, empty if no data is input.

    Raises:
        ValueError: If data with non-matching coordinates are input.

        KeyError: If non-existant labels are input.

    """

    def assert_grids_equal(data_records):
        control = data_records[0]
        for data in data_records[1:]:
            assert np.allclose(data[xl], control[xl], atol=1e-4)
            assert np.allclose(data[yl], control[yl], atol=1e-4)

    def calc_arithmetic_mean(label, data_records):
        data = get_container(label)
        for i, d in enumerate(data_records):
            data[i,:] = d[label]

        return data.mean(axis=0)

    def calc_weighted_mean(label, weight, avg_data, data_records):
        try:
            data = get_container(label)
            for i, d in enumerate(data_records):
                data[i,:] = d[label]*d[weight]
        except ValueError:
            raise KeyError("Input labels of 'weights' not in data: %r %r"
                    % (label, weight))
        else:
            total_weight = len(data_records)*avg_data[weight]

        return np.nan_to_num(data.sum(axis=0)/(total_weight))

    xl, yl = coord_labels

    try:
        assert_grids_equal(data_records)
    except AssertionError:
        raise ValueError("Input grids of data not identical.")
    except IndexError:
        avg_data = np.array([])
        data_labels = set()
    else:
        avg_data = data_records[0].copy()
        data_labels = set(avg_data.dtype.names).difference(set([xl, yl]))

    weighted_labels = [l for l, _ in weights]
    data_labels.difference_update(weighted_labels)
    get_container = lambda l: np.empty((len(data_records), avg_data.size),
            dtype=avg_data[l].dtype)

    for l in data_labels:
        avg_data[l] = calc_arithmetic_mean(l, data_records)

    for l, w in weights:
        avg_data[l] = calc_weighted_mean(l, w, avg_data, data_records)

    return avg_data


def transfer_data(grid, data, coord_labels=('X', 'Y')):
    """Return a projection of data onto an input grid.

    The input grid must be a superset of the input data for the projection
    to succeed. Their dtype must be identical. A copy of the input grid
    is returned.

    Args:
        grid (ndarray): Data record with final grid coordinates.

        data (ndarray): Data record to project onto new grid.

        coord_labels (2-tuple, default=('X', 'Y'): Record labels for coordinates.

    Returns:
        ndarray: Data record with projected data.

    Raises:
        ValueError: If input data or grid has coordinate duplicates.

    """

    xl, yl = coord_labels
    full_data = grid.copy()

    for d in data:
        x, y = np.array([d[l] for l in (xl, yl)])
        ind = np.isclose(full_data[xl], x, atol=1e-4) & np.isclose(full_data[yl], y, atol=1e-4)

        try:
            ind_input = (data[xl] == x) & (data[yl] == y)
            assert (len(data[ind_input]) == 1)
            assert (len(full_data[ind]) == 1)
        except AssertionError:
            raise ValueError("Input data or grid has duplicate coordinates: "
                    "Input %r, grid %r" % (data[ind_input], full_data[ind]))

        full_data[ind] = d.copy()

    return full_data


def get_combined_grid(data, bin_size, coord_labels=('X', 'Y')):
    """Return a grid with input bin size that contains all input data.

    Args:
        data (ndarray): List of numpy records with coordinate labels.

        bin_size (int's): 2-tuple with bin sizes along the coordinate axes.

        coord_labels (2-tuple, default=('X', 'Y'): Record labels for coordinates.

    Returns:
        ndarray: A container of the same dtype as input data.

    """

    if data == []:
        raise ValueError("No data to get a combined grid from.")

    xl, yl = coord_labels

    xmin, xmax = (f([f(d[xl]) for d in data]) for f in (np.min, np.max))
    ymin, ymax = (f([f(d[yl]) for d in data]) for f in (np.min, np.max))
    dx, dy = bin_size

    xs = np.arange(xmin, xmax+dx, dx)
    ys = np.arange(ymin, ymax+dy, dy)

    x, y = np.meshgrid(xs, ys)

    combined_grid = np.zeros(x.size, dtype=data[-1].dtype)
    combined_grid[xl] = x.ravel()
    combined_grid[yl] = y.ravel()

    return combined_grid

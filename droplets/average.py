import numpy as np

"""Module for averaging FlowData objects."""


def average_data(data_records, weights=[]):
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

    Returns:
        ndarray: Averaged data, empty if no data is input.

    Raises:
        ValueError: If data with non-matching coordinates are input.

        KeyError: If non-existant labels are input.

    """

    def assert_grids_equal(data_records):
        control = data_records[0]
        for data in data_records[1:]:
            assert np.array_equal(data['X'], control['X'])
            assert np.array_equal(data['Y'], control['Y'])

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

    try:
        assert_grids_equal(data_records)
    except AssertionError:
        raise ValueError("Input grids of data not identical.")
    except IndexError:
        avg_data = np.array([])
        data_labels = set()
    else:
        avg_data = data_records[0].copy()
        data_labels = set(avg_data.dtype.names).difference(set(['X', 'Y']))

    weighted_labels = [l for l, _ in weights]
    data_labels.difference_update(weighted_labels)
    get_container = lambda l: np.empty((len(data_records), avg_data.size),
            dtype=avg_data[l].dtype)

    for l in data_labels:
        avg_data[l] = calc_arithmetic_mean(l, data_records)

    for l, w in weights:
        avg_data[l] = calc_weighted_mean(l, w, avg_data, data_records)

    return avg_data


def transfer_data(grid, data):
    """Return a projection of data onto an input grid.

    The input grid must be a superset of the input data for the projection
    to succeed. Their dtype must be identical. A copy of the input grid
    is returned.

    Args:
        grid (ndarray): Data record with final grid coordinates.

        data (ndarray): Data record to project onto new grid.

    Returns:
        ndarray: Data record with projected data.

    Raises:
        ValueError: If input data or grid has coordinate duplicates.

    """

    full_data = grid.copy()

    for d in data:
        x, y = np.array([d[l] for l in ('X', 'Y')])
        ind = np.isclose(full_data['X'], x) & np.isclose(full_data['Y'], y)

        try:
            ind_input = (data['X'] == x) & (data['Y'] == y)
            assert (len(data[ind_input]) == 1)
            assert (len(full_data[ind]) == 1)
        except AssertionError:
            raise ValueError("Input data or grid has duplicate coordinates: "
                    "Input %r, grid %r" % (data[ind_input], full_data[ind]))

        full_data[ind] = d.copy()

    return full_data


def get_combined_grid(data, bin_size):
    """Return a grid with input bin size that contains all input data.

    Args:
        data (ndarray): List of numpy records with coordinate labels.

        bin_size (int's): 2-tuple with bin sizes along the coordinate axes.

    Returns:
        ndarray: A container of the same dtype as input data.

    """

    if data == []:
        raise ValueError("No data to get a combined grid from.")

    xmin, xmax = (f([f(d['X']) for d in data]) for f in (np.min, np.max))
    ymin, ymax = (f([f(d['Y']) for d in data]) for f in (np.min, np.max))
    dx, dy = bin_size

    xs = np.arange(xmin, xmax+dx, dx)
    ys = np.arange(ymin, ymax+dy, dy)

    x, y = np.meshgrid(xs, ys)

    combined_grid = np.zeros(x.size, dtype=data[-1].dtype)
    combined_grid['X'] = x.ravel()
    combined_grid['Y'] = y.ravel()

    return combined_grid

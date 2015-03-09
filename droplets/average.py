import numpy as np

"""Module for averaging FlowData objects."""


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

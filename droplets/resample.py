import numpy as np

from droplets.average import get_combined_grid, transfer_data
from droplets.flow import FlowData


def downsample_flow_data(flow, num_combine,
        coord_labels=('X', 'Y'), weights=[]):
    """Downsample input data by combining bins.

    The input data is downsampled by summing data values when
    combining the bins. Data types can be averaged by weighing
    against another value by supplying a list of tuples in
    (label, weight) format. If the weighting sums to zero in
    an area the result is returned as zero.

    The bins are combined in positive directions starting at
    index (0, 0). If the final number of bins does not evenly
    divide the starting number, some bins will be cut from the
    system in the direction of positive x and y.

    Args:
        flow (FlowData): Object to downsample. Must have a `shape`
                and `bin_size` attached as metadata.

        num_combine (2-tuple): Number of bins to combine along x and y.

    Keyword args:
        coord_labels (2-tuple, default=('X', 'Y'): Record labels for coordinates.

        weights (label, weight): A list of 2-tuples with labels of data
            and weights to calculate a weighted mean for.

    Returns:
        FlowData: A new object with downsampled grid.

    """

    # Create grid to average onto
    info = get_downscaled_grid_info(flow, num_combine, coord_labels)
    coords = get_downscaled_grid_coords(info)

    # Resample onto grid
    resampled_flow = average_data(coords, flow, num_combine, info,
            coord_labels, weights)

    return FlowData(*resampled_flow, info=info)


def average_data(coords, flow, num_combine, info, coord_labels, weights):
    # Create container for result and add coords data
    data = np.zeros(coords[0].shape, dtype=flow.data.dtype)

    for l, cs in zip(coord_labels, coords):
        data[l] = cs

    # Sort input data in same order as the new grid
    # and reshape to 2D array
    coords_order = [coord_labels[i] for i in (1, 0)]
    shape = [flow.shape[i] for i in (1, 0)]
    reshaped_input = np.reshape(np.sort(flow.data, order=coords_order),
                                shape)

    # Get data labels, keep labels to be weighed separate
    weighted_labels = [l for l, _ in weights]
    data_labels = [l for l in data.dtype.names
                   if l not in list(coord_labels) + weighted_labels]

    nx, ny = num_combine

    for i, row in enumerate(data):
        for j, col in enumerate(row):
            inds_x, inds_y = (slice(n*k, n*(k+1))
                              for n, k in [(nx, j), (ny, i)])

            for l, w in weights:
                # The result should be zero if no weighting data is found
                try:
                    data[l][i,j] = np.average(reshaped_input[l][inds_y, inds_x],
                                              weights=reshaped_input[w][inds_y, inds_x])
                except ZeroDivisionError:
                    data[l][i,j] = 0.

            for l in data_labels:
                data[l][i,j] = np.sum(reshaped_input[l][inds_y, inds_x])

    return [(l, data[l].ravel()) for l in data.dtype.names]


def get_downscaled_grid_info(flow, num_combine, coord_labels):
    bin_size = flow.bin_size
    origin = [np.min(flow.data[l]) for l in coord_labels]

    new_bin_size = [v*n for v, n in zip(flow.bin_size, num_combine)]
    new_shape = [v//n for v, n in zip(flow.shape, num_combine)]
    new_origin = [o + 0.5*(new_s - s)
              for o, s, new_s in zip(origin, bin_size, new_bin_size)]
    new_end = [o + (n-1)*s for o, n, s in zip(new_origin, new_shape, new_bin_size)]

    new_size = [(o, e) for o, e in zip(new_origin, new_end)]

    info = {
        'bin_size': new_bin_size,
        'shape': new_shape,
        'size': new_size,
        'num_bins': new_shape[0]*new_shape[1]
    }

    return info


def get_downscaled_grid_coords(info):
    origin = (v for v, _ in info['size'])
    bin_size = info['bin_size']
    shape = info['shape']

    x, y = (o + dx*np.arange(n) for o, dx, n in zip(origin, bin_size, shape))

    return np.meshgrid(x, y)
    

import numpy as np

def get_lim_indices(data, limits):
    """Return indices of a data record fulfilling input limits.

    The returned array of indices are an intersection of all input limits.
    If no limits are input all indices are returned.

    Args:
        data (record): Field data in record format. See FlowData.data
            for information.

        limits (dict): Limits of data fields, specified as {label: (min, max)}
            where the label corresponds to a field in the data record.
            None can be given instead of either limit, this ignores a limit
            in that direction.

    Returns:
        ndarray: A Numpy array tuple of indices along each axis fulfilling
            the input limits. See numpy.where for information on return type.

    """

    def get_indices(array, vmin, vmax):
        vmin = vmin if vmin != None else -np.inf
        vmax = vmax if vmax != None else np.inf
        return np.where((array >= vmin) & (array <= vmax))

    # If no limits are input, return all indices
    if limits == {}:
        return np.arange(data.size)

    for i, (label, (vmin, vmax)) in enumerate(limits.items()):
        if i == 0:
            indices = get_indices(data[label], vmin, vmax)
        else:
            indices_new = get_indices(data[label], vmin, vmax)
            indices = np.intersect1d(indices, indices_new)

    return indices

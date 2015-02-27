import numpy as np
import warnings


def get_interface(flow, label, radius, **kwargs):
    """Yield droplet interface indices of a FlowData object.

    Indices are yielded from bottom to the top and correspond to found
    left and right indices of the boundary.

    The simple for bins to be a part of the liquid is that the cell
    and a given number of bins within a radius of it must have a given
    parameter value (the input label) larger than or equal to a cut-off
    value.

    Periodic boundary conditions are not applied for the radius search.
    This means that bins on the outermost edges of the system are searching
    for neighbouring bins in a smaller area around themselves and might
    not be detected given identical settings.

    Args:
        flow (FlowData): A FlowData object. Must contain a data record
            with coordinates and selected height map label, as well
            as an input shape of the system.

        label (str): Record label used as base for the interface height map.

        radius (float): Radius to include bins within.

    Keyword Args:
        cutoff (float, default=None): Which interface height to cut the
            boundary at. Defaults to the midpoint height.

        num_bins (int, default=1): Number of bins inside the set radius
            which must pass the cut-off criteria.

        ylims (2-tuple, default=(None, None)): Only return interface boundary
            within these height limits.

        coord_labels (2-tuple, default=('X', 'Y'): Record labels for coordinates.

    Yields:
        int, int: Indices for the left and right boundary of an
            interface layer.

    Warnings:
        UserWarning: If a cutoff is not specified and the system is homogenous
            for the input parameter label no interface can be found.
            (None, None) is returned.

    """

    def get_cutoff(cutoff, data):
        if cutoff == None:
            vmin = np.min(data[label])
            vmax = np.max(data[label])
            assert (vmin != vmax)

            cutoff = 0.5*(vmin + vmax)

        return cutoff

    def get_yvalues(Y, ymin, ymax):
        ymin = -np.inf if ymin == None else ymin
        ymax =  np.inf if ymax == None else ymax
        ys = np.unique(Y)

        return ys[(ys >= ymin) & (ys <= ymax)]

    def traverse_layer(sorted_layer):
        for cell in sorted_layer:
            index = np.where(flow.data == cell)[0][0]

            if cell_is_droplet(index, flow.data, label, radius, cutoff, **kwargs):
                return index

        else:
            return None

    try:
        cutoff = get_cutoff(kwargs.pop('cutoff', None), flow.data)
    except AssertionError:
        warnings.warn("system is homogenous: no interface can be found", UserWarning)
        return None, None

    ylims = kwargs.pop('ylims', (None, None))

    xlabel, ylabel = kwargs.get('coord_labels', ('X', 'Y'))
    Y = flow.data[ylabel]
    ys = get_yvalues(Y, *ylims)

    for i, y in enumerate(ys):
        layer = flow.data[Y == y]
        sorted_layer = np.sort(layer, order=xlabel)

        left = traverse_layer(sorted_layer)
        right = traverse_layer(reversed(sorted_layer))

        if left != None and right != None:
            yield left, right


def cell_is_droplet(cell, system, label, radius, cutoff, **kwargs):
    """Determine whether a cell is a connected part of a liquid system.

    See 'get_interface' for details on the cell search.

    Args:
        cell (int): Index of origin cell in system record array.

        system (record): Field data in record format.

        label (str): Record label used as base for the boundary determination.

        radius (float): Radius to include bins within.

        cutoff (float): Which interface height to cut the boundary at.

    Keyword Args:
        num_bins (int, default=1): Number of bins inside the set radius
            which must pass the cut-off criteria.

        coord_labels (2-tuple, default=('X', 'Y'): Record labels for coordinates.

    Returns:
        bool: Whether or not the cell can be considered part of the liquid.

    Raises:
        IndexError: If coordinate labels could not be found in data record.

    """

    num_bins = kwargs.pop('num_bins', 1)

    try:
        assert (system[cell][label] >= cutoff)
        indices = get_indices_in_radius(cell, system, radius, **kwargs)
        assert (len(system[system[label][indices] >= cutoff]) >= num_bins)
    except AssertionError:
        return False
    except IndexError:
        raise IndexError("could not find set labels in system")
    else:
        return True


def get_indices_in_radius(cell, system, radius, **kwargs):
    """Return indices of bins within a given radius of input cell.

    Args:
        cell (int): Index of origin cell in system record array.

        system (record): Field data in record format.

        radius (float): Radius to include bins within.

    Keyword Args:
        coord_labels (2-tuple, default=('X', 'Y'): Record labels for coordinates.

    Returns:
        ndarray: Numpy array of indices of bins within set radius of origin.

    """

    coord_labels = kwargs.pop('coord_labels', ('X', 'Y'))

    x, y = (system[cell][coord_labels[i]] for i in (0, 1))
    xs, ys = (system[coord_labels[i]] for i in (0, 1))

    indices = np.where((np.sqrt((xs - x)**2 + (ys - y)**2) <= radius))[0]

    return indices[indices != cell]

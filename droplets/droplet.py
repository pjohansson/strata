import matplotlib.pyplot as plt
import numpy as np


def get_interface(flow, label, radius, **kwargs):
    """Yield droplet interface indices of a FlowData object.

    Indices are yielded from bottom to the top and correspond to found
    left and right indices of the boundary.

    Args:
        flow (FlowData): A FlowData object. Must contain a data record
            with coordinates and selected height map label, as well
            as an input shape of the system.

        label (str): Record label used as base for the interface height map.

        radius (float): Radius to include cells within.

    Keyword Args:
        cutoff (float, default=None): Which interface height to cut the
            boundary at. Defaults to the midpoint height.

        num_cells (int, default=1): Number of cells inside the set radius
            which must pass the cut-off criteria.

        coord_labels (2-tuple, default=('X', 'Y'): Record labels for coordinates.

    Yields:
        int, int: Indices for the left and right boundary of an
            interface layer.

    """

    def traverse_layer(sorted_layer):
        for cell in sorted_layer:
            index = np.where(flow.data == cell)[0][0]

            if cell_is_droplet(index, flow.data, label, radius, **kwargs):
                return index

        else:
            return None

    Y = flow.data['Y']
    ys = np.unique(Y)

    for i, y in enumerate(ys):
        layer = flow.data[Y == y]
        sorted_layer = np.sort(layer)

        left = traverse_layer(sorted_layer)
        right = traverse_layer(reversed(sorted_layer))

        if left != None and right != None:
            yield left, right


def cell_is_droplet(cell, system, label, radius, **kwargs):
    """Determine whether a cell is a connected part of a liquid system.

    The simple rule is that the cell and a given number of cells within
    a radius of it must have a parameter value larger than or equal to
    a cut-off value.

    Periodic boundary conditions are not applied for the radius search.

    Args:
        cell (int): Index of origin cell in system record array.

        system (record): Field data in record format.

        label (str): Record label used as base for the boundary determination.

        radius (float): Radius to include cells within.

    Keyword Args:
        cutoff (float, default=None): Which interface height to cut the
            boundary at. Defaults to the midpoint height.

        num_cells (int, default=1): Number of cells inside the set radius
            which must pass the cut-off criteria.

        coord_labels (2-tuple, default=('X', 'Y'): Record labels for coordinates.

    Returns:
        bool: Whether or not the cell can be considered part of the liquid.

    Raises:
        IndexError: If coordinate labels could not be found in data record.

    """

    cutoff = kwargs.pop('cutoff', None)
    num_cells = kwargs.pop('num_cells', 1)

    try:
        if cutoff == None:
            cutoff = 0.5*(np.max(system[label]) + np.min(system[label]))

        try:
            assert (system[cell][label] >= cutoff)
            indices = get_indices_in_radius(cell, system, radius, **kwargs)
            assert (len(system[system[label][indices] >= cutoff]) >= num_cells)
        except AssertionError:
            return False
        else:
            return True

    except IndexError:
        raise IndexError("could not find set labels in system")


def get_indices_in_radius(cell, system, radius, **kwargs):
    """Return indices of cells within a given radius of input cell.

    Args:
        cell (int): Index of origin cell in system record array.

        system (record): Field data in record format.

        radius (float): Radius to include cells within.

    Keyword Args:
        coord_labels (2-tuple, default=('X', 'Y'): Record labels for coordinates.

    Returns:
        ndarray: Numpy array of indices of cells within set radius of origin.

    """

    coord_labels = kwargs.pop('coord_labels', ('X', 'Y'))

    x, y = (system[cell][coord_labels[i]] for i in (0, 1))
    xs, ys = (system[coord_labels[i]] for i in (0, 1))

    indices = np.where((np.sqrt((xs - x)**2 + (ys - y)**2) <= radius))[0]

    return indices[indices != cell]

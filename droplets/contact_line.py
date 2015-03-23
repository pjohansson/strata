import numpy as np

from droplets.interface import get_interface


def get_contact_line_cells(flow, label, size=(0., 0.), **kwargs):
    """Return cells from the left and right contact line of FlowData map.

    Args:
        flow (FlowData): A FlowData object. Must contain a data record
            with coordinates and selected height map label.

        label (str): Record label used as a base for the interface height map.

    Keyword Args:
        size (float, default=0.): 2-tuple with size of area to include from
            the contact line edge.

        coord_labels (2-tuple, default=('X', 'Y'): Record labels for coordinates.

    Returns:
        (ndarray, ndarray): Numpy record arrays of left and right contact
            line cells.

    For additional keyword arguments see `droplets.interface.get_interface`.

    """

    def get_interface_of_height(height):
        """Return interface of input height with axes swapped."""

        interface = []
        interface_iter = get_interface(flow, label, **kwargs)
        try:
            interface.append(next(interface_iter))
        except StopIteration:
            return [], []

        index, _ = interface[0]
        ymax = y(index) + height

        for ind in interface_iter:
            try:
                assert (y(ind[0]) <= ymax)
            except AssertionError:
                break
            else:
                interface.append(ind)

        return np.array(interface).swapaxes(0, 1)

    def get_cells_in_direction(inds, dir_mod):
        """Get all the cells for the edge inds in the inwards direction."""

        xs = flow.data[xlabel]
        ys = flow.data[ylabel]

        try:
            xinner = x(inds[0]) + size[0]*dir_mod
        except IndexError:
            icells = []
        else:
            xmin, xmax = (f([f(x(inds)), xinner]) for f in (np.min, np.max))
            ymin, ymax = (y(inds)[i] for i in (0, -1))

            icells = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)

        return flow.data[icells].copy()

    xlabel, ylabel = kwargs.get('coord_labels', ('X', 'Y'))
    x = lambda index: flow.data[index][xlabel]
    y = lambda index: flow.data[index][ylabel]

    interface = get_interface_of_height(size[1])
    left, right = [get_cells_in_direction(inds, dir_mod)
            for inds, dir_mod in zip(interface, (1, -1))]

    return left, right


def adjust_cell_coordinates(cells, direction, coord_labels=('X', 'Y')):
    """Adjust cell coordinates to origin based on input direction.

    The input direction is either 'left' or 'right' to adjust the cells
    to that direction around the origin.

    Args:
        cells (ndarray): Record object to adjust coordinates for.

        direction ('left' or 'right'): Adjustment of cells around origin.

    Keyword Args:
        coord_labels (2-tuple, default=('X', 'Y'): Record labels for coordinates.

    Returns:
        (float, ndarray): A 2-tuple of the adjustment vector and the
            adjusted cells.

    """

    def get_x0(cells, direction, xlabel):
        xs = np.unique(cells[xlabel])

        if direction == 'right':
            x0 = xs.min()
        elif direction == 'left':
            x0 = xs.max()
        else:
            raise KeyError("Input direction must be 'left' or 'right'.")

        return x0

    def adjust_cells(cells, adj, xlabel, ylabel):
        adj_cells = cells.copy()
        adj_cells[xlabel] -= adj[0]
        adj_cells[ylabel] -= adj[1]

        return adj_cells

    try:
        xlabel, ylabel = coord_labels
        x0 = get_x0(cells, direction, xlabel)
        y0 = cells[ylabel].min()
    except ValueError:
        adj = (0, 0)
    else:
        adj = (x0, y0)

    return adj, adjust_cells(cells, adj, xlabel, ylabel)

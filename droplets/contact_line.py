import numpy as np

from droplets.interface import get_interface


def get_contact_line_cells(flow, label, extract_area=(0., 0.),
        extract_height=0., **kwargs):
    """Return cells from the left and right contact line of FlowData map.

    Extracts all cells from the interface up to the contact line edge
    and an area determined by an input box of additional cells from this
    point: Place a box of this size at the contact line edge, facing
    inwards towards the droplet. All cells within this box and the area
    covered from this box up to and including the interface are returned
    for the left and right edges.

    The box can be placed with its outermost edge in contact with the
    interface at a different height using `extract_height`. If so the
    box extends to the interface in all layers, including the bottom-most,
    as above. This is useful if for instance the bottom-most layer is
    unstable and a higher reference point should be used for contact line
    extraction, while still extracting the entire contact line tip.

    Args:
        flow (FlowData): A FlowData object. Must contain a data record
            with coordinates and selected height map label.

        label (str): Record label used as a base for the interface height map.

    Keyword Args:
        extract_area (float, optional): 2-tuple with size of area to
            extract from the contact line edge.

        extract_height (float, optional): Extract cells with area synchronised
            at this height.

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

        # Get the bottom-most interface indices
        try:
            interface.append(next(interface_iter))
        except StopIteration:
            return [], []

        # Use the left edge index for height calculation
        index, _ = interface[0]
        ymax = y(index) + height

        # Add the remaining interface indices to the list
        for ind in interface_iter:
            try:
                assert (y(ind[0]) <= ymax)
            except AssertionError:
                break
            else:
                interface.append(ind)

        return np.array(interface).swapaxes(0, 1)

    def get_cells_of_edge(inds, edge, dx, extract_height):
        """Get all the cells for the edge inds in the inwards direction."""

        xs = flow.data[xlabel]
        ys = flow.data[ylabel]

        # Find the interface index at which height to place the extraction box
        i = 0
        try:
            while y(inds[i+1]) <= y(inds[0])+extract_height:
                i += 1
        except IndexError:
            pass

        try:
            if edge == 'left':
                xinner = x(inds[i]) + dx
            else:
                xinner = x(inds[i]) - dx

            xmin = np.min([x(inds).min(), xinner])
            xmax = np.max([x(inds).max(), xinner])
            ymin, ymax = (y(inds)[i] for i in (0, -1))

            icells = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)

        except IndexError:
            icells = []

        return flow.data[icells].copy()

    xlabel, ylabel = kwargs.get('coord_labels', ('X', 'Y'))
    x = lambda index: flow.data[index][xlabel]
    y = lambda index: flow.data[index][ylabel]

    width, height = extract_area

    interface_inds = get_interface_of_height(height)
    left, right = [get_cells_of_edge(edge_inds, edge, width, extract_height)
            for edge_inds, edge in zip(interface_inds, ('left', 'right'))]

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

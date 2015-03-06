import numpy as np

from droplets.droplet import get_interface

def get_contact_line_cells(flow, label, size=(0., 0.), radius=1., **kwargs):
    """Return cells from the left and right contact line of FlowData map.

    Args:
        flow (FlowData): A FlowData object. Must contain a data record
            with coordinates and selected height map label.

        label (str): Record label used as a base for the interface height map.

    Keyword Args:
        size (float, default=0.): 2-tuple with size of area to include from
            the contact line edge.

        radius (float, default=1.): See `droplets.droplet.get_interface`.

    Returns:
        (ndarray, ndarray): Numpy record arrays of left and right contact
            line cells.

    For additional keyword arguments see `droplets.droplet.get_interface`.

    """

    def get_interface_of_height(height):
        """Return interface of input height with axes swapped."""

        y = lambda index: flow.data[index]['Y']

        interface = []
        interface_iter = get_interface(flow, label, radius, **kwargs)
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

        x = lambda index: flow.data[index]['X']
        y = lambda index: flow.data[index]['Y']

        xs = flow.data['X']
        ys = flow.data['Y']

        try:
            xinner = x(inds[0]) + size[0]*dir_mod
        except IndexError:
            icells = []
        else:
            xmin, xmax = (f([f(x(inds)), xinner]) for f in (np.min, np.max))
            ymin, ymax = (y(inds)[i] for i in (0, -1))

            icells = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)

        return flow.data[icells].copy()

    interface = get_interface_of_height(size[1])
    left, right = [get_cells_in_direction(inds, dir_mod)
            for inds, dir_mod in zip(interface, (1, -1))]

    return left, right

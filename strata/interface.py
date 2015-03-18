import numpy as np

from droplets.interface import get_interface


def get_interface_coordinates(flow, label, adjust_com=False, **kwargs):
    """Get the interface from a FlowData object using input label.

    Args:
        flow (FlowData): A FlowData object. Must contain a data record
            with coordinates and selected height map label.

        label (str): Record label used as base for the interface height map.

    Keyword Args:
        adjust_com (bool, default=False): Center spreading coordinates around
            the center of mass.

        coord_labels (2-tuple, default=('X', 'Y'): Record labels for coordinates.

    See `get_interface` for additional keyword arguments.

    """

    xl, yl = kwargs.get('coord_labels', ('X', 'Y'))

    left, right = np.array(list(get_interface(flow, label, **kwargs))).T
    inds = np.array(left.tolist() + right[::-1].tolist())

    xs, ys = (flow.data[l][inds] for l in (xl, yl))
    if adjust_com == True:
        xs -= np.average(flow.data[xl], weights=flow.data[label])

    return xs, ys

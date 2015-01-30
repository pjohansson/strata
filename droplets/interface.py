import numpy as np
from droplets.droplet import get_interface

"""Module for analysing a droplet interface."""


def get_contact_angle(flow, height, label, radius, **kwargs):
    """Return the dynamic contact angles of a droplet state.

    This is calculated for the left- and rightmost contact line edges
    by simple trigonometrics, using the interface boundary as calculated
    by `get_interface`.

    The calculation uses the bottom layer most close to the input `floor`
    height, which defaults to 0, and matches against the uppermost layer
    whose height `y` fulfils `y` <= `height` - `floor`, where `height`
    is the input height difference over which to calculate the angles.

    Args:
        flow (FlowData): A FlowData object.

        height (float): The height difference at which to calculate
            the angle.

        label (str): Record label used as base for the interface height map.

        radius (float): Radius to include bins within.

    Keyword Args:
        cutoff (float, default=None): Which interface height to cut the
            boundary at. Defaults to the midpoint height.

        num_bins (int, default=1): Number of bins inside the set radius
            which must pass the cut-off criteria.

        floor (float, default=None): Height at which the interface forms.
            Defaults to the bottom interface bins found in the data map.

        coord_labels (2-tuple, default=('X', 'Y'): Record labels for coordinates.

    Returns:
        float, float: 2-tuple with left- and rightmost dynamic contact angles
            (degrees).
        None, None: If no angle was found.

    """

    def get_floor_height(floor, ys):
        if floor != None:
            return ys[np.abs(ys - floor).argmin()]

    def get_coords(flow, indices):
        y = flow.data[ylabel][indices[0]]
        xs = np.array([flow.data[xlabel][i] for i in indices])
        return y, xs

    def get_xdeltas(xs, xedges):
        xdeltas = xs - xedges
        xdeltas[1] *= -1
        return xdeltas

    def get_angles(xs, dy):
        xdeltas = get_xdeltas(xs, xedges)
        return np.degrees(np.arctan2(dy, xdeltas))

    xlabel, ylabel = kwargs.get('coord_labels', ('X', 'Y'))

    # Set ylims from an input floor
    floor = kwargs.pop('floor', None)
    yfloor = get_floor_height(floor, flow.data[ylabel])
    kwargs['ylims'] = (yfloor, None)

    interface = get_interface(flow, label, radius, **kwargs)
    indices = next(interface)
    ymin, xedges = get_coords(flow, indices)

    # Break if a floor was specified and no cells are found in that layer
    if floor != None and ymin != yfloor:
        return None, None

    for indices in interface:
        y, xs = get_coords(flow, indices)
        dy = y - ymin

        if dy <= height:
            angles = get_angles(xs, dy)
        else:
            break

    return angles

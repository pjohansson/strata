import numpy as np

"""Module for analysing a droplet interface.

Contains functions for generating the droplet boundary and analysing
it, for example extracting the dynamic contact angle.

"""


def get_contact_angle(flow, height, label, **kwargs):
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

    Keyword Args:
        floor (float, default=None): Height at which the interface forms.
            Defaults to the bottom interface bins found in the data map.

        cutoff (float, default=None): Which interface height to cut the
            boundary at. Defaults to the midpoint height.

        cutoff_radius (float, default=None): Radius to include bins within.
            Defaults to the minimum distance found between cells in the system.

        cutoff_bins (int, default=1): Number of bins inside the set radius
            which must pass the cut-off criteria.

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

    interface = get_interface(flow, label, **kwargs)
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


def get_interface(flow, label, **kwargs):
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
            with coordinates and selected height map label.

        label (str): Record label used as base for the interface height map.

    Keyword Args:
        cutoff (float, default=None): Which interface height to cut the
            boundary at. Defaults to the midpoint height.

        cutoff_radius (float, default=None): Radius to include bins within.
            Defaults to the minimum distance found between cells in the system.

        cutoff_bins (int, default=1): Number of bins inside the set radius
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

    def get_radius(data, xlabel, ylabel, **kwargs):
        radius = kwargs.pop('cutoff_radius', None)
        if radius == None:
            xdiff = np.diff(np.unique(data[xlabel]))
            ydiff = np.diff(np.unique(data[ylabel]))
            radius = next(np.min(map(np.min, (xdiff, ydiff))))

        return radius

    def traverse_layer(sorted_layer, data):
        for cell in sorted_layer:
            index = np.where(data == cell)[0][0]

            if _cell_is_droplet(index, data, label,
                    cutoff_radius, cutoff, **kwargs):
                return index
        else:
            return None

    try:
        cutoff = get_cutoff(kwargs.pop('cutoff', None), flow.data)
    except AssertionError:
        print("[WARNING]: System is homogenous: no interface can be found.")
        return None, None

    coord_labels = kwargs.get('coord_labels', ('X', 'Y'))
    cutoff_radius = get_radius(flow.data, *coord_labels, **kwargs)

    ylims = kwargs.pop('ylims', (None, None))

    xlabel, ylabel = kwargs.get('coord_labels', ('X', 'Y'))

    # Work with non-zero part of dataset
    indices = np.where(flow.data[label] >= cutoff)[0]
    data = flow.data[indices]

    Y = data[ylabel]
    ys = get_yvalues(Y, *ylims)

    for i, y in enumerate(ys):
        layer = data[Y == y]
        sorted_layer = np.sort(layer, order=xlabel)

        left = traverse_layer(sorted_layer, data)

        if left != None:
            right = traverse_layer(reversed(sorted_layer), data)
            yield [indices[edge] for edge in (left, right)]


def _cell_is_droplet(cell, system, label, radius, cutoff, **kwargs):
    """Determine whether a cell is a connected part of a liquid system.

    See 'get_interface' for details on the cell search.

    Args:
        cell (int): Index of origin cell in system record array.

        system (record): Field data in record format.

        label (str): Record label used as base for the boundary determination.

        radius (float): Radius to include bins within.

        cutoff (float): Which interface height to cut the boundary at.

    Keyword Args:
        cutoff_bins (int, default=1): Number of bins inside the set radius
            which must pass the cut-off criteria.

        coord_labels (2-tuple, default=('X', 'Y'): Record labels for coordinates.

    Returns:
        bool: Whether or not the cell can be considered part of the liquid.

    Raises:
        KeyError: If coordinate labels could not be found in data record.

    """

    cutoff_bins = kwargs.pop('cutoff_bins', 1)

    try:
        assert (system[cell][label] >= cutoff)
        indices = _get_indices_in_radius(cell, system, radius, **kwargs)

        # Count number of bins of selected indices larger than the cutoff
        assert (np.sum(system[label][indices] >= cutoff) >= cutoff_bins)
    except AssertionError:
        return False
    except ValueError:
        raise KeyError("could not find set labels in system")
    else:
        return True


def _get_indices_in_radius(cell, system, radius, **kwargs):
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
    radius_sq = radius**2

    x, y = (system[cell][coord_labels[i]] for i in (0, 1))
    xs, ys = (system[coord_labels[i]] for i in (0, 1))

    # Extract part of dataset before radius search
    ixs = np.where((xs - x)**2 <= radius_sq)[0]
    indices = ixs[np.where((xs[ixs] - x)**2 + (ys[ixs] - y)**2 <= radius_sq)[0]]

    return indices[indices != cell]

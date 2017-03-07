import matplotlib.pyplot as plt
import numpy as np

from droplets.flow import FlowData
from droplets.sample import sample_viscous_dissipation
from strata.dataformats.read import read_from_files
from strata.utils import decorate_graph


def view_flowfields(*files, labels=('U', 'V'), cutoff_label='M', cutoff=None,
        colour=None, vlim=(None, None), pivot='middle', **kwargs):
    """View flow fields of input files.

    Args:
        files (paths): List of files to view.

    Keyword Args:
        labels (2-tuple, optional): 2-tuple with labels of mass flow along
            x and y.

        cutoff (float, optional): Cutoff for data to show fields for.

        cutoff_label (str, optional): Label for cutting data.

        colour (str, optional): Colour flow fields with data from this label,
            or enter 'flow' to colour by flow magnitude.

        pivot (str, optional): Pivot for flow field arrows.

        vlim (floats, optional): 2-tuple with limits for the colour map if
            a label has been supplied.

    See `strata.utils.decorate_graph` for more keyword arguments.

    """

    kwargs.setdefault('axis', 'scaled')

    # Order the data labels
    coord_labels = kwargs.get('coord_labels', ['X', 'Y'])

    # Get some limits on coordinates and an optional cut-off
    xlim, ylim = [kwargs.get(lims, (None, None)) for lims in ('xlim', 'ylim')]
    clim = (cutoff_label, cutoff)

    for i, (data, info, _) in enumerate(read_from_files(*files)):
        flow = FlowData(data, info=info)

        if colour == 'flow':
            flow.data = add_absolute_flow(flow.data)
        elif colour == 'visc':
            add_viscous_dissipation(flow, viscosity=8.77e-4)

        xs, ys, us, vs, weights = get_quiver_data(flow.data,
                list(labels), coord_labels, colour,
                clim, xlim, ylim)

        try:
            plot_quiver(xs, ys, us, vs, weights, pivot, vlim, **kwargs)
        except Exception as err:
            print("Could not draw/save figure: ", end='')
            print(err)
            break


def get_quiver_data(data, labels, coord_labels, colour, clim, xlim, ylim):
    """Return the flow data after cutting out empty cells.

    Empty cells are cells without any flow or with a value less
    than specified for an input cutoff and label and cutoff tuple.

    Args:
        data (record): Flow data as a `numpy.record` object.

        labels (2-tuple): 2-tuple with labels of mass flow along
            x and y.

        coord_labels (2-tuple): 2-tuple with coordinate labels.

        colour (str): Label to return as weights or None for
            unitary colour.

        clim (label, value): Optional cutoff label and value for
            data points to return. None means no cutoff is applied.

        xlim, ylim (2-tuples): Limits of axes.

    Returns:
        numpy.array's: 5-tuple of arrays containing (in order)
            coordinates along x and y, mass flow along x and y,
            weights to colour arrows by.

    """

    def cut_system(cs, lims):
        """Get indices of system within coordinate limits."""
        cmin, cmax = (float(v) if v != None else v for v in lims)
        if cmin == None: cmin = np.min(cs)
        if cmax == None: cmax = np.max(cs)

        return (cs >= cmin) & (cs <= cmax)

    xs, ys = (data[l] for l in coord_labels)
    us, vs = (data[l] for l in labels)

    # Cut bins without flow
    inds = (us != 0.) & (vs != 0.)

    # Apply limits on coordinates
    inds = inds & cut_system(xs, xlim) & cut_system(ys, ylim)

    # If there is an additional cutoff, apply
    clabel, cutoff = clim
    if clabel != None:
        inds = inds & cut_system(data[clabel], (cutoff, None))

    # Weights are either from input label or unit
    try:
        assert(colour != None)
        weights = data[colour]
    except Exception:
        weights = np.ones(data.size)

    return (cs[inds] for cs in (xs, ys, us, vs, weights))


@decorate_graph
def plot_quiver(xs, ys, us, vs, weights, pivot, vlim, **kwargs):
    """Draw a quiver plot of input data."""

    scale = kwargs.get('scale', 1.)
    width = kwargs.get('width', 0.0015)

    fig = plt.quiver(xs, ys, us, vs, weights, clim=vlim,
            scale=scale, width=width, pivot=pivot)

    return fig


def view_flowmap_2d(*files, label='M', type='heightmap', **kwargs):
    """View bin data of input files.

    Args:
        files (paths): List of files to view.

    Keyword Args:
        label (str, optional): Data label to use as height values.

        type (str, optional): Type of map to draw, can be 'height'
            or 'contour'.

        filled (bool): Contour: Draw a filled contour. (False)

    See `strata.utils.decorate_graph` for more keyword arguments.

    """

    # Select drawing function
    draw_functions = {
        'height': height,
        'contour': contour
        }
    try:
        func = draw_functions[type]
    except KeyError:
        raise TypeError("Can not draw type '%r'." % type)

    print_avg_value = kwargs.get('average_value', False)
    kwargs.setdefault('axis', 'scaled')
    kwargs.setdefault('coord_labels', ('X', 'Y'))

    for i, (data, info, _) in enumerate(read_from_files(*files)):
        flow = FlowData(data, info=info)

        if label == 'flow':
            flow.data = add_absolute_flow(flow.data)

        if print_avg_value:
            print_average_value(flow.data, label, kwargs)

        func(flow.data, info, label, **kwargs)
        plt.clf()


@decorate_graph
def height(data, info, label, **kwargs):
    """Draw a heightmap using a 2d histogram."""

    cmin, cmax = (float(c) if c != None else c for c in kwargs['clim'])
    vmin, vmax = (float(v) if v != None else v for v in kwargs['vlim'])

    xs, ys = (data[l] for l in kwargs['coord_labels'])
    fig = plt.hist2d(xs, ys, weights=data[label], bins=info['shape'],
            cmin=cmin, cmax=cmax, vmin=vmin, vmax=vmax)

    return fig


@decorate_graph
def contour(data, info, label, **kwargs):
    """Draw a contour map of the data."""

    # Get some contour specific options
    num_contours = kwargs.get('num_contours', 10)
    levels = kwargs.get('contour_levels', None)
    colors = kwargs.get('contour_colours', None)
    filled = kwargs.get('filled', False)

    vmin, vmax = (float(v) if v != None else v for v in kwargs['vlim'])
    if (vmin != None or vmax != None) and levels == None:
        data[label] = data[label].clip(vmin, vmax)

    grid_data = data.reshape(info['shape'])
    xs, ys = (grid_data[l] for l in kwargs['coord_labels'])
    cs = grid_data[label]

    if not filled:
        func = plt.contour
    else:
        func = plt.contourf

    fig = func(xs, ys, cs, num_contours, levels=levels, colors=colors)

    return fig


def add_absolute_flow(data):
    """Return a numpy array with absolute flow as a field named 'flow'.

    The absolute flow is added as a new label for the returned data
    array, which is calculated as sqrt(U**2 + V**2) where U and V
    are the flow arrays along the dimensions.

    Args:
        data (ndarray): Record which must contain flow labels 'U' and 'V'.

    Returns:
        ndarray: The same record with an added field with label 'flow'.

    """

    from numpy.lib.recfunctions import append_fields

    absolute_flow = np.sqrt(data['U']**2 + data['V']**2)

    return append_fields(data, 'flow', absolute_flow, dtypes='float')


def add_viscous_dissipation(flow, viscosity):
    """Add the viscous dissipation as a field named 'visc'.

    The dissipation is added as a new label for the data array.

    Args:
        data (ndarray): Record which must contain coordinate labels 'X' and 'Y'
            as well as flow labels 'U' and 'V'.

    """

    from numpy.lib.recfunctions import append_fields

    dx, dy = flow.spacing
    nx, ny = flow.shape


    flow.data = np.sort(flow.data, order=['Y', 'X']).reshape(ny, nx)
    U, V = [flow.data[l]*flow.data['M'] for l in ['U', 'V']]


    dudy, dudx = np.gradient(U, dy, dx, edge_order=2)
    dvdy, dvdx = np.gradient(V, dy, dx, edge_order=2)
    dvdy *= 0.0

    viscous_dissipation = 2*viscosity*(dudx**2 + dvdy**2 - (dudx + dvdy)**2/3.0) \
            + viscosity*(dvdx + dudy)**2
    viscous_dissipation = dudy

    flow.data = flow.data.ravel()
    viscous_dissipation = viscous_dissipation.ravel()
    viscous_dissipation /= flow.data['M']

    flow.data = append_fields(flow.data, 'visc', viscous_dissipation, dtypes='float')

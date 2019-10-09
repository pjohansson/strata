import matplotlib.pyplot as plt
import numpy as np

from droplets.flow import FlowData
from droplets.sample import sample_center_of_mass, sample_viscous_dissipation
from strata.dataformats.read import read_from_files
from strata.utils import decorate_graph


def view_flowfields(*files, labels=('U', 'V'), cutoff_label='M', cutoff=None,
        colour=None, vlim=(None, None), pivot='middle', streamlines=False, **kwargs):
    """View flow fields of input files.

    Args:
        files (paths): List of files to view.

    Keyword Args:
        labels (2-tuple, optional): 2-tuple with labels of mass flow along
            x and y.

        cutoff (float, optional): Cutoff for data to show fields for.

        cutoff_label (str, optional): Label for cutting data.

        colour (str, optional): Colour flow fields with data from this label,
            or enter 'flow' to colour by flow magnitude, 'visc' to colour by
            viscous dissipation, 'radial' by radial velocity from center
            of mass, 'evaporation' by phase field evaporation.

        pivot (str, optional): Pivot for flow field arrows.

        streamlines (bool, optional): Plot field as stream lines instead of quiver.

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
        elif colour == 'radial':
            com = sample_center_of_mass(flow)
            flow.data = add_radial_flow(flow.data, com)
        elif colour == 'evaporation':
            add_evaporation(flow)

        try:
            if not streamlines:
                xs, ys, us, vs, weights = get_quiver_data(flow.data,
                        list(labels), coord_labels, colour,
                        clim, xlim, ylim)

                plot_quiver(xs, ys, us, vs, weights, pivot, vlim, **kwargs)
            else:
                x, y, u, v, weights, line_weights = get_streamline_data(
                    flow, list(labels), coord_labels, colour, clim, xlim, ylim
                )

                plot_streamlines(x, y, u, v, weights, line_weights, **kwargs)
        except Exception as err:
            print("Could not draw/save figure: ", end='')
            print(err)
            break

def get_streamline_data(flow, labels, coord_labels, colour, clim, xlim, ylim):
    x0, y0 = flow.origin
    dx, dy = flow.spacing

    data = flow.data

    cutoff_label, cutoff_value = clim

    xlabel, ylabel = coord_labels
    xmin, xmax = xlim
    ymin, ymax = ylim

    mask = (data[cutoff_label] < cutoff_value)

    data = np.ma.array(data, mask=mask)

    x0 = max(data['X'].min(), xmin if xmin else -np.inf)
    x1 = min(data['X'].max(), xmax if xmax else np.inf)
    y0 = max(data['Y'].min(), ymin if ymin else -np.inf)
    y1 = min(data['Y'].max(), ymax if ymax else np.inf)

    inds = (data[xlabel] >= x0) \
        & (data[xlabel] <= x1) \
        & (data[ylabel] >= y0) \
        & (data[ylabel] <= y1)

    data = data[inds]

    x0 = data[xlabel].min()
    x1 = data[xlabel].max()
    y0 = data[ylabel].min()
    y1 = data[ylabel].max()

    nx = int((x1 - x0) / dx) + 2
    ny = int((y1 - y0) / dy) + 2

    data = data.reshape(nx, ny).transpose()

    x = x0 + np.arange(nx) * dx
    y = y0 + np.arange(ny) * dy

    u = data['U']
    v = data['V']

    weights = data[colour]
    line_weights = data['M']

    return x, y, u, v, weights, line_weights


@decorate_graph
def plot_streamlines(xs, ys, us, vs, weights, line_weights=None, **kwargs):
    """Draw a streamlines plot of input data."""

    ny, nx = us.shape

    density_x = nx // 30
    density_y = ny // 30

    linewidth = 1.5 * line_weights / line_weights.max()

    fig = plt.streamplot(xs, ys, us, vs, (density_x, density_y), color=weights, linewidth=linewidth)

    return fig


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

def view_flowmap_2d(*files, label='M', type='heightmap',
        cutoff_label='M', cutoff=None,
        **kwargs):
    """View bin data of input files.

    Args:
        files (paths): List of files to view.

    Keyword Args:
        label (str, optional): Data label to use as height values.

        cutoff (float, optional): Cutoff for data to show bins for.

        cutoff_label (str, optional): Label for cutting data.

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
    kwargs.setdefault('noaxis', True)
    kwargs.setdefault('coord_labels', ('X', 'Y'))

    for i, (data, info, _) in enumerate(read_from_files(*files)):
        flow = FlowData(data, info=info)

        if label == 'flow':
            flow.data = add_absolute_flow(flow.data)

        if label == 'visc_diss':
            label = 'visc'
            add_viscous_dissipation(flow, viscosity=8.77e-4)

        if label == 'evaporation':
            add_evaporation(flow)

        if label in ['grad_rho_x', 'grad_rho_y']:
            add_density_gradient(flow)

        if cutoff != None and cutoff_label != None:
            flow = flow.lims(cutoff_label, cutoff, None)

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
        ndarray: The same record with an added field of label 'flow'.

    """

    from numpy.lib.recfunctions import append_fields

    absolute_flow = np.sqrt(data['U']**2 + data['V']**2)

    return append_fields(data, 'flow', absolute_flow, dtypes='float')


def add_radial_flow(data, coord):
    """Return a numpy array with the radial flow as a field named 'radial'.

    The radial flow is calculated from the center of mass and is positive
    in the clockwise direction.

    Args:
        data (ndarray): Record which must contain coordinate labels 'X' and 'Y'
            as well as flow labels 'U' and 'V'.

    Returns:
        ndarray: The same record with an added field of label 'radial'.

    """

    from numpy.lib.recfunctions import append_fields

    x0, y0 = coord
    radial_angle = np.arctan2(data['Y'] - y0, data['X'] - x0)

    radial_flow = data['U'] * np.sin(radial_angle) \
        - data['V'] * np.cos(radial_angle)

    return append_fields(data, 'radial', radial_flow, dtypes='float')

def add_viscous_dissipation(flow, viscosity):
    """Add the viscous dissipation as a field named 'visc'.

    The dissipation is added as a new label for the data array.

    Args:
        data (ndarray): Record which must contain coordinate labels 'X' and 'Y'
            as well as flow labels 'U' and 'V'.

    """

    from numpy.lib.recfunctions import append_fields

    viscous_dissipation = sample_viscous_dissipation(flow, viscosity,
        weight_label='M')

    flow.data = append_fields(flow.data, 'visc', viscous_dissipation.ravel(),
        dtypes='float')


def add_density_gradient(flow):
    from numpy.lib.recfunctions import append_fields

    (nx, ny) = flow.shape
    spacing = flow.spacing

    flow.sort()

    data = flow.data.reshape(ny, nx)

    mass_gradient_y, mass_gradient_x = np.gradient(
        data['M'], *spacing, edge_order=1
        )

    flow.data = append_fields(flow.data, 'grad_rho_x', mass_gradient_x.ravel(),
        dtypes='float')
    flow.data = append_fields(flow.data, 'grad_rho_y', mass_gradient_y.ravel(),
        dtypes='float')



def add_evaporation(flow):
    """Add the `u . (grad rho)` evaporation term as a field named 'evaporation'.

    `u` here is the flow field and `rho` the density.
    
    Args:
        data (ndarray): Record which must contain coordinate labels 'X' and 'Y',
            flow labels 'U' and 'V' and mass label 'M'.

    """

    from numpy.lib.recfunctions import append_fields

    (nx, ny) = flow.shape
    spacing = flow.spacing

    flow.sort()

    data = flow.data.reshape(ny, nx)

    mass_gradient_y, mass_gradient_x = np.gradient(
        data['M'], *spacing, edge_order=1
        )

    us = data['U']
    vs = data['V']

    evaporation = us * mass_gradient_x + vs * mass_gradient_y

    flow.data = append_fields(
        flow.data, 'evaporation', evaporation.ravel(), dtypes='float'
    )

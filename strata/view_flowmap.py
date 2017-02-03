import matplotlib.pyplot as plt
import numpy as np

from droplets.flow import FlowData
from strata.dataformats.read import read_from_files
from strata.utils import decorate_graph


def view_flowfields(*files, labels=('U', 'V'), cutoff_label='M', cutoff=None,
        alpha_label=None, alpha_cutoff=None, alpha_percentile=None,
        colour=None, vlim=(None, None), pivot='middle', trans=None, **kwargs):
    """View flow fields of input files.

    Args:
        files (paths): List of files to view.

    Keyword Args:
        labels (2-tuple, optional): 2-tuple with labels of mass flow along
            x and y.

        cutoff (float, optional): Cutoff for data to show fields for.

        cutoff_label (str, optional): Label for cutting data.

        alpha_label (str, optional): Label for alpha data.

        alpha_cutoff (float, optional): Limit of alpha data. Prioritized
            over the percentile (see below).

        alpha_percentile (int, optional): Percentile of alpha data to use
            as limit. Overridden if a direct value is input (see above).

        colour (str, optional): Colour flow fields with data from this label,
            or enter 'flow' to colour by flow magnitude.

        trans (str, optional): Set flow field transparency using this label.

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
    alim = (alpha_label, alpha_percentile, alpha_cutoff)

    for i, (data, _, _) in enumerate(read_from_files(*files)):
        flow = FlowData(data)

        if colour == 'flow':
            flow.data = add_absolute_flow(flow.data)

        xs, ys, us, vs, weights, alphas = get_quiver_data(flow.data,
                list(labels), coord_labels, colour,
                clim, alim, xlim, ylim)

        try:
            plot_quiver(xs, ys, us, vs, weights, alphas, pivot, vlim, **kwargs)
        except Exception as err:
            print("Could not draw/save figure: ", end='')
            print(err)
            break


def get_quiver_data(data, labels, coord_labels, colour, clim, alim, xlim, ylim):
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

        alim (label, percentile, cutoff): Optional cutoff label, percentile
            and cutoff value for the alpha channel. The cutoff value
            is prioritized.

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
        weights = data[colour]

    except Exception:
        weights = np.ones(data.size)

    # Get alpha label and max
    alabel, percentile, cutoff = alim

    try:
        alphas = data[alabel]

        if cutoff != None:
            amax = cutoff
            alphas = alphas.clip(0., amax)/amax
        elif percentile != None:
            amax = np.percentile(data[alabel], percentile)
            alphas = alphas.clip(0., amax)/amax
        else:
            alphas /= np.max(alphas)

    except Exception:
        alphas = np.ones(data.size)

    return (cs[inds] for cs in (xs, ys, us, vs, weights, alphas))


@decorate_graph
def plot_quiver(xs, ys, us, vs, weights, alphas, pivot, vlim, **kwargs):
    """Draw a quiver plot of input data."""

    scale = kwargs.get('scale', 1.)
    width = kwargs.get('width', 0.0015)

    fig = plt.quiver(xs, ys, us, vs, clim=vlim,
            scale=scale, width=width, pivot=pivot)

    # Set the arrow colours manually here since they won't update
    # if they are previously set
    collection = fig.ax.collections[0]
    colors = collection.to_rgba(weights) # This translates from the set colormap
    colors[:,3] = alphas
    collection.set_color(colors)

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

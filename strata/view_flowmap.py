import matplotlib.pyplot as plt
import numpy as np

from droplets.flow import FlowData
from strata.dataformats.read import read_from_files
from strata.utils import decorate_graph


def view_flowfields(*files, labels=('U', 'V'), cutoff_label='M', cutoff=None,
        colour=None, vlim=(None, None), **kwargs):
    """View flow fields of input files.

    Args:
        files (paths): List of files to view.

    Keyword Args:
        labels (2-tuple, optional): 2-tuple with labels of mass flow along
            x and y.

        cutoff (float, optional): Cutoff for data to show fields for.

        cutoff_label (str, optional): Label for cutting data.

        colour (str, optional): Colour flow fields with data from this label.

        vlim (floats, optional): 2-tuple with limits for the colour map if
            a label has been supplied.

    See `strata.utils.decorate_graph` for more keyword arguments.

    """

    kwargs.setdefault('axis', 'scaled')

    # Order the data labels
    data_labels = kwargs.get('coord_labels', ['X', 'Y']) + list(labels)

    # Get some limits on coordinates and an optional cut-off
    xlim, ylim = [kwargs.get(lims, (None, None)) for lims in ('xlim', 'ylim')]
    clim = (cutoff_label, cutoff)

    for i, (data, _, _) in enumerate(read_from_files(*files)):
        flow = FlowData(data)
        xs, ys, us, vs, weights = get_quiver_data(flow.data, data_labels,
                colour, clim, xlim, ylim)

        plot_quiver(xs, ys, us, vs, weights, vlim, **kwargs)


def get_quiver_data(data, labels, colour, clim, xlim, ylim):
    """Return the flow data after cutting out empty cells."""

    def cut_system(cs, lims):
        """Get indices of system within coordinate limits."""
        cmin, cmax = (float(v) if v != None else v for v in lims)
        if cmin == None: cmin = np.min(cs)
        if cmax == None: cmax = np.max(cs)

        return (cs >= cmin) & (cs <= cmax)

    xs, ys, us, vs = (data[l] for l in labels)

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
def plot_quiver(xs, ys, us, vs, weights, vlim, **kwargs):
    """Draw a quiver plot of input data."""

    scale = kwargs.get('scale', 1.)
    width = kwargs.get('width', 0.0015)

    fig = plt.quiver(xs, ys, us, vs, weights, clim=vlim,
            scale=scale, width=width)

    return fig


def view_flowmap_2d(*files, label='M', type='heightmap', **kwargs):
    """View bin data of input files.

    Args:
        files (paths): List of files to view.

    Keyword Args:
        label (str, optional): Data label to use as height values.

        type (str, optional): Type of map to draw, can be 'height'
            or 'contour'.

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

    kwargs.setdefault('axis', 'scaled')
    kwargs.setdefault('coord_labels', ('X', 'Y'))

    for i, (data, info, _) in enumerate(read_from_files(*files)):
        flow = FlowData(data, info=info)
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

    vmin, vmax = (float(v) if v != None else v for v in kwargs['vlim'])
    if (vmin != None or vmax != None) and levels == None:
        data[label] = data[label].clip(vmin, vmax)

    grid_data = data.reshape(info['shape'])
    xs, ys = (grid_data[l] for l in kwargs['coord_labels'])
    cs = grid_data[label]

    fig = plt.contour(xs, ys, cs, num_contours, levels=levels, colors=colors)

    return fig

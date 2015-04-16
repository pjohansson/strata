import matplotlib.pyplot as plt

from droplets.flow import FlowData
from strata.dataformats.read import read_from_files
from strata.utils import decorate_graph


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

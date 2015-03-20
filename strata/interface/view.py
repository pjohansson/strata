import numpy as np
import os
import pandas as pd
import progressbar as pbar

from strata.interface.collect import write_interface_data
from strata.utils import gen_filenames, pop_fileopts, find_groups_to_singles, decorate_graph


def view_interfaces(base, average=1, save_xvg='', **kwargs):
    """Plot the interfaces of files at input base.

    Input files are plaintext files of Grace format, with two columns
    representing x and y coordinates of the interface, connected from
    the bottom left edge, over the droplet and ending at the bottom right
    edge. Positions should be in units of nanometer.

    Optionally averages the interface data over an input bundling length.
    The averaged interfaces can be written to disk as new Grace formatted
    files.

    Args:
        base (str): Base path to input files.

    Keyword args:
        average (int, default=1): Average interface data in bundles of
            this size.

        save_fig (path): Save drawn interfaces to base path.

        save_xvg (path): Save interfaces to base path.

        begin (int, default=1): First interface file number.

        end (int, default=inf): Final interface file number.

        ext (str, default='.dat'): File extension.

        extfig (str, default='.png'): Saved figure format extension.

    Returns:
        xs, ys: 2-tuple with lists of interface coordinates along x and y.

    See `strata.utils.decorate_graph` for more figure drawing options.

    """

    def gen_figure_filenames(base, fopts, average, figext):
        begin_at = np.ceil(fopts['begin']/average)
        filenames = gen_filenames(base, begin=begin_at, ext=figext)
        yield from filenames

    kwargs.setdefault('ext', '.xvg')
    fopts = pop_fileopts(kwargs)
    quiet = kwargs.pop('quiet', False)

    # Figure options
    kwargs.setdefault('axis', 'scaled')
    save_fig = kwargs.pop('save_fig', None)
    figext = kwargs.pop('extfig', '.png')
    fnfig = gen_figure_filenames(save_fig, fopts, average, figext)

    groups_singles = list(find_groups_to_singles(base, save_xvg, average,
            **fopts))

    xs, ys = [], []

    if not quiet:
        widgets = ['Reading files: ',
                pbar.Bar(), ' (', pbar.SimpleProgress(), ') ', pbar.ETA()]
        progress = pbar.ProgressBar(widgets=widgets, maxval=len(groups_singles))
        progress.start()

    for i, (fngroup, fnout) in enumerate(groups_singles):
        interface_list = read_interface_files(fngroup)
        interface = average_interfaces(interface_list)

        if save_xvg != '':
            write_interface_data(fnout, interface, fngroup, kwargs)

        if save_fig != None:
            kwargs['save_fig'] = next(fnfig)

        if len(interface) > 1:
            plot_interface_data(interface, **kwargs)

        xs.append(np.array(interface))
        ys.append(np.array(interface.index))

        if not quiet:
            progress.update(i+1)

    if not quiet:
        progress.finish()

    return xs, ys


@decorate_graph
def plot_interface_data(interface, **kwargs):
    """Plot an interface.

    See `strata.utils.decorate_graph` for input options.

    """

    import matplotlib.pyplot as plt

    plt.clf()
    plt.plot(interface, interface.index, **kwargs)


def read_interface_files(files):
    """Return a list of read interfaces coordinates.

    Args:
        files (str's): List of files to read interface data from.

    Returns:
        list: List of 2-tuples with x and y coordinates from each file.

    """

    return [np.genfromtxt(fn, unpack=True) for fn in files]


def average_interfaces(interfaces):
    """Return a Pandas Series's with the mean boundary of input interfaces.

    Args:
        interfaces (list): List of 2-tuples with x and y coordinates for
            each interface.

    Returns:
        pd.Series: An averaged interface.

    """

    def get_boundary_sides(xs, ys):
        split = int(len(xs)/2)
        left = pd.Series(xs[:split], index=ys[:split])
        right = pd.Series(xs[split:], index=ys[split:])

        return left, right

    def average_edge(edges_list, side):
        side_list = [interface[side] for interface in edges_list]
        df = pd.DataFrame(side_list).T

        return df.mean(axis=1, skipna=False).dropna()

    edges_list = [get_boundary_sides(xs, ys) for xs, ys in interfaces]
    left, right = (average_edge(edges_list, side) for side in (0, 1))

    return stitch_edge_series(left, right)


def combine_interfaces(xs, ys):
    """Return DataFrame objects of collected left and right interfaces.

    Args:
        xs, ys (ndarrays): Lists of input coordinate arrays.

    Returns:
        DataFrames: 2-tuple of left and right edges.

    """

    left, right = [pd.DataFrame(edge_coords).T
            for edge_coords in all_coords_to_edges(xs, ys)]

    return left, right


def all_coords_to_edges(xs, ys):
    """Return lists of coordinates as Series objects for both interface edges.

    Input lists of several interfaces to return as edges. Returned Series's
    will have y as indices and x as values.

    Args:
        xs, ys (ndarrays): Lists of input coordinate arrays.

    Returns:
        pd.Series, pd.Series: 2-tuple for left and right interface boundaries,
            with list of Pandas Series's objects for each edge.

    """

    split_at = lambda a: int(len(a)/2)

    left = [pd.Series(x[:split_at(x)], index=y[:split_at(x)])
            for x, y in zip(xs, ys)]
    right = [pd.Series(x[split_at(x):], index=y[split_at(x):])
            for x, y in zip(xs, ys)]

    return left, right


def stitch_edge_series(left, right):
    """Return coordinates from input left and right edges as single arrays.

    Args:
        left, right (Series): Pandas Series objects of edges.

    Returns:
        Series: Pandas Series of combined edges.

    """

    return left.append(right.sort_index(ascending=False))

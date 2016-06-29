import numpy as np
import os
import progressbar as pbar

from strata.utils import find_datamap_files, pop_fileopts, prepare_path, decorate_graph


def interface_contact_angle(base, height, delta_t=1., save_xvg=None, **kwargs):
    """Calculate the mean contact angles of interface files with input base.

    Input files are plaintext files of Grace format, with two columns
    representing x and y coordinates of the interface, connected from
    the bottom left edge, over the droplet and ending at the bottom right
    edge. Positions should be in units of nanometer.

    Args:
        base (str): Base path to input files.

        height (float): Measure the contact angle at this height over the base.

    Keyword args:
        delta_t (float, optional)): Time difference between interface files.

        save_xvg (path, optional): Save interfaces to base path.

        begin (int, default=1): First interface file number.

        end (int, default=inf): Final interface file number.

        ext (str, default='.dat'): File extension.

    Returns:
        times, angles: Tuple with lists of times and contact angles.

    See `strata.utils.decorate_graph` for more figure drawing options.

    """

    kwargs.setdefault('ext', '.xvg')
    fopts = pop_fileopts(kwargs)
    quiet = kwargs.pop('quiet', False)

    filenames = list(find_datamap_files(base, **fopts))

    if not quiet:
        widgets = ['Calculating angles: ',
                pbar.Bar(), ' (', pbar.SimpleProgress(), ') ', pbar.ETA()]
        progress = pbar.ProgressBar(widgets=widgets, maxval=len(filenames))
        progress.start()

    contact_angles = []

    for i, fn in enumerate(filenames):
        left, right = read_interface_file(fn)

        angles = [calc_angle(edge, direction, height)
                for edge, direction in zip([left, right], ('left', 'right'))]
        contact_angles.append(np.mean(angles))

        if not quiet:
            progress.update(i+1)

    times = [i*delta_t for i in range(len(contact_angles))]

    if save_xvg:
        write_angle_data(save_xvg, base, height, times, contact_angles,
            begin=fopts['begin'], end=fopts['end'])

    if kwargs.get('show', True) or kwargs.get('save_fig', None):
        kwargs.setdefault('axis', 'tight')
        draw_figure(times, contact_angles, **kwargs)

    if not quiet:
        progress.finish()

    return times, contact_angles


@decorate_graph
def draw_figure(times, angles, **kwargs):
    """Draw a figure with the contact angles as a function of time.

    Args:
        times, angles (floats): Time and measured contact angles to draw.

    See `matplotlib.pyplot.plot` for more keyword arguments.

    """

    import matplotlib.pyplot as plt

    plt.plot(times, angles, **kwargs)


@prepare_path
def write_angle_data(path, base, height, times, angles, **kwargs):
    """Write angle data to file at path.

    Data is output in whitespace separated xmgrace format. NaN values
    are converted to 0 for compliance with column based plotting tools.

    Args:
        path (str): Path to output file.

        times, angles (floats): Time and measured contact angles to write.

    """

    def get_header(path, base, height, **kwargs):
        import strata
        import time

        time_str = time.strftime('%c', time.localtime())

        header = (
                "# Measured contact angles of system\n"
                "# \n"
                "# Created by module: %s\n"
                "# Creation date: %s\n"
                "# Using module version: %s\n"
                "# \n"
                % (__name__, time_str, strata.strata.version))

        inputs = (
                "# Input:\n"
                "#   File base path: %r\n"
                "#   Begin, end: %r, %r\n"
                "#   Measured at height: %r\n"
                "#   Delta-t: %r\n"
                "#   Interface options: See original files.\n"
                "# \n"
                % (os.path.realpath(base),
                    kwargs.get('begin', None), kwargs.get('end', None),
                    height, kwargs.get('delta_t', 1.),
                    ))

        inputs += "# t (ps) mean angle (deg.)"

        return header + inputs

    header = get_header(path, base, height, **kwargs)
    data = np.array([times, angles]).T
    np.savetxt(path, data, fmt='%.3f', delimiter=' ',
            header=header, comments='')


def read_interface_file(fn):
    """Return the left and right interfaces read from a file."""

    xs, ys = np.genfromtxt(fn, unpack=True)

    length = len(xs)/2
    base = np.zeros(length, dtype=[('X', 'float'), ('Y', 'float')])

    left = base.copy()
    left['X'] = xs[:length]
    left['Y'] = ys[:length]
    left.sort(order='Y')

    right = base.copy()
    right['X'] = xs[-1:-(length+1):-1]
    right['Y'] = ys[-1:-(length+1):-1]
    right.sort(order='Y')

    return left, right


def calc_angle(edge, direction, height):
    """Return the contact angle adjusted for direction at height."""

    x0, y0 = [edge[l][0] for l in ('X', 'Y')]

    hindex = np.abs(edge['Y'] - y0 - height).argmin()
    x1, y1 = [edge[l][hindex] for l in ('X', 'Y')]

    dx = x1 - x0
    dy = y1 - y0

    if direction == 'right':
        dx = -dx

    return np.degrees(np.arccos(dx/np.sqrt(dx**2 + dy**2)))

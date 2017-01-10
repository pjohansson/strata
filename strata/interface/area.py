import numpy as np
import os
import progressbar as pbar

from strata.interface.view import read_interface_file
from strata.utils import find_datamap_files, pop_fileopts, prepare_path, decorate_graph, write_module_header

def integrate_interfaces(base, save_xvg='', delta_t=1., **kwargs):
    """Return the integrate areas of collected interfaces at base.

    Args:
        base (str): Base path to input interfaces.

    Keyword Args:
        save_xvg (str, optional): Write integrated areas to an output file.

        delta_t (float, optional): Time difference between input interfaces.

        begin (int, default=1): First interface number.

        end (int, default=inf): Final interface number.

        ext (str, default='.xvg'): File extension.

    See `strata.utils.decorate_graph` for more keyword arguments.

    """

    def prepare_output(save_xvg, base, dt, header_opts, fopts):
        header_opts.update(fopts)
        write_header(save_xvg, base, dt, header_opts)

    kwargs.setdefault('ext', '.xvg')
    fopts = pop_fileopts(kwargs)
    files = list(find_datamap_files(base, **fopts))

    if save_xvg:
        try:
            prepare_output(save_xvg, base, delta_t, kwargs.copy(), fopts)
        except PermissionError:
            print("[WARNING] Output disabled: could not open '%s' for writing."
                % save_xvg)
            save_xvg = None

    collected_areas = []
    times = []

    quiet = kwargs.pop('quiet', False)
    if not quiet:
        widgets = ['Calculating the area of files: ',
                pbar.Bar(), ' (', pbar.SimpleProgress(), ') ', pbar.ETA()]
        progress = pbar.ProgressBar(widgets=widgets, maxval=len(files))
        progress.start()

    for i, fn in enumerate(files):
        interface = read_interface_file(fn)
        try:
            area = get_area_of_interface(interface)
        except TypeError as err:
            print("Could not calculate area of interface file %r: " % fn, end='')
            print(err)
        else:
            collected_areas.append(area)
            times.append(i*delta_t)

            if save_xvg:
                with open(save_xvg, 'a') as fp:
                    fp.write("%.3f %.3f\n" % (i*delta_t, area))

        if not quiet:
            progress.update(i+1)

    if not quiet:
            progress.finish()

    plot_interface_area(collected_areas, times, **kwargs)

    return times, collected_areas


def get_area_of_interface(interface):
    """Return the integrated area of an interface.

    Args:
        interface (2-tuple): Lists of x and y coordinates of interface.

    Returns:
        float: The integrated area.

    """

    def get_trapezoid_base(xleft, xright):
        for i in range(len(xleft)-1):
            bottom = xright[i] - xleft[i]
            top = xright[i+1] - xleft[i+1]
            yield 0.5*(bottom + top)

    xs, ys = interface
    num_height = len(xs)/2

    yleft, yright = ys[:num_height], ys[num_height:][::-1]
    xleft, xright = xs[:num_height], xs[num_height:][::-1]

    try:
        assert(np.isclose(yleft, yright).all())
    except AssertionError:
        print(yleft, yright)
        raise TypeError("Bad interface coordinates encountered.")

    hs = np.diff(yleft)
    area = sum([h*m for h, m in zip(hs, get_trapezoid_base(xleft, xright))])

    return area


def calc_length(xs, ys):
    dxs, dys = [np.diff(xs), np.diff(ys)]
    return np.sum(np.sqrt(dxs**2 + dys**2))


@decorate_graph
def plot_interface_area(areas, times, **kwargs):
    """Plot the collected interface area per time."""

    import matplotlib.pyplot as plt

    fig = plt.plot(times, areas)

    return fig


@prepare_path
def write_header(output_path, input_base, dt, kwargs):
    """Verify that output path is writable and write header."""

    title = "Integrated areas of input interface files"
    write_module_header(output_path, __name__, title)

    with open(output_path, 'a') as fp:
        inputs = (
                "# Input:\n"
                "#   File base path: %r\n"
                "#   Begin, end: %r, %r\n"
                "#   Delta-t: %r\n"
                "# \n"
                "# Time (ps) Area (nm^2)\n"
                % (os.path.realpath(input_base),
                    kwargs.get('begin', None), kwargs.get('end', None),
                    kwargs.get('dt', 1.)
                    ))

        fp.write(inputs)

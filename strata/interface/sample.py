import numpy as np
import os
import progressbar as pbar

from strata.interface.view import read_interface_file
from strata.utils import find_datamap_files, pop_fileopts, prepare_path, decorate_graph, write_module_header

def sample_interfaces(base, variable, save_xvg='', delta_t=1., **kwargs):
    """Return the sampled variable of collected interfaces at base.

    The variable is either 'area' or 'length' (ie. circumference) of
    the interface.

    Args:
        base (str): Base path to input interfaces.

        variable (str): Sample either 'area' or 'length'.

    Keyword Args:
        save_xvg (str, optional): Write integrated areas to an output file.

        delta_t (float, optional): Time difference between input interfaces.

        begin (int, default=1): First interface number.

        end (int, default=inf): Final interface number.

        ext (str, default='.xvg'): File extension.

    See `strata.utils.decorate_graph` for more keyword arguments.

    """

    def prepare_output(save_xvg, base, variable, dt, header_opts, fopts):
        header_opts.update(fopts)
        write_header(save_xvg, base, variable, dt, header_opts)

    kwargs.setdefault('ext', '.xvg')
    fopts = pop_fileopts(kwargs)
    files = list(find_datamap_files(base, **fopts))

    if save_xvg:
        try:
            prepare_output(save_xvg, base, variable, delta_t, kwargs.copy(), fopts)
        except PermissionError:
            print("[WARNING] Output disabled: could not open '%s' for writing."
                % save_xvg)
            save_xvg = None

    collected_samples = []
    times = []

    quiet = kwargs.pop('quiet', False)
    if not quiet:
        widgets = ['Calculating the %s of files: ' % variable,
                pbar.Bar(), ' (', pbar.SimpleProgress(), ') ', pbar.ETA()]
        progress = pbar.ProgressBar(widgets=widgets, max_value=len(files))
        progress.start()

    for i, fn in enumerate(files):
        xs, ys = read_interface_file(fn)
        try:
            sample = None
            if variable == 'area':
                sample = get_area_of_interface(xs, ys)
            elif variable == 'length':
                sample = calc_length(xs, ys)
            else:
                raise ValueError("Invalid input variable '%s': Must be 'area' or 'length'" % variable)
        except TypeError as err:
            print("Could not calculate %s of interface file %r: " % (variable, fn), end='')
            print(err)
        else:
            collected_samples.append(sample)
            times.append(i*delta_t)

            if save_xvg:
                with open(save_xvg, 'a') as fp:
                    fp.write("%.3f %.3f\n" % (i*delta_t, sample))

        if not quiet:
            progress.update(i+1)

    if not quiet:
            progress.finish()

    plot_interface_area(collected_samples, times, **kwargs)

    return times, collected_samples


def get_area_of_interface(xs, ys):
    """Return the integrated area of an interface.

    Args:
        xs, ys (floats): Lists of x and y coordinates of interface.

    Returns:
        float: The integrated area.

    """

    def get_trapezoid_base(xleft, xright):
        for i in range(len(xleft)-1):
            bottom = xright[i] - xleft[i]
            top = xright[i+1] - xleft[i+1]
            yield 0.5*(bottom + top)

    num_height = int(len(xs)/2)

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
    """Return the circumference of the interface.

    Args:
        xs, ys (floats): Lists of x and y coordinates of interface.

    Returns:
        float: The circumference.

    """

    dxs, dys = [np.diff(xs), np.diff(ys)]

    return np.sum(np.sqrt(dxs**2 + dys**2))


@decorate_graph
def plot_interface_area(areas, times, **kwargs):
    """Plot the collected interface area per time."""

    import matplotlib.pyplot as plt

    fig = plt.plot(times, areas)

    return fig


@prepare_path
def write_header(output_path, input_base, variable, dt, kwargs):
    """Verify that output path is writable and write header."""

    title = "Samples of input interface files"
    write_module_header(output_path, __name__, title)

    ylabel = "Area (nm^2)" if variable == 'area' else "Circumference (nm)"

    with open(output_path, 'a') as fp:
        inputs = (
                "# Input:\n"
                "#   File base path: %r\n"
                "#   Variable: %r\n"
                "#   Begin, end: %r, %r\n"
                "#   Delta-t: %r\n"
                "# \n"
                "# Time (ps) %s\n"
                % (os.path.realpath(input_base), variable,
                    kwargs.get('begin', None), kwargs.get('end', None),
                    kwargs.get('dt', 1.), ylabel
                    ))

        fp.write(inputs)

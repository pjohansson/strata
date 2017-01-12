import os
import numpy as np
import progressbar as pbar

from droplets.flow import FlowData
from droplets.sample import sample_inertial_energy
from strata.dataformats.read import read_from_files
from strata.utils import find_datamap_files, pop_fileopts, prepare_path, write_module_header


def sample_average_files(base, label, output=None, sum=False, dt=1.,
        cutoff_label=None, cutoff=None, **kwargs):
    """Sample average collected data of input label from files.

    Returns lists with input file times and the averaged sampled value
    of bins for the corresponding time.

    Optionally the total value of the quantity in the system can be returned
    by supplying the keyword argument `sum`.

    A cutoff value for bins to sample in can be supplied. Bins with values
    below this cutoff are not included in the sample average. By default
    the data label used as a cutoff is the input sample label, but a
    separate label can also be supplied. This is useful when wanting
    to sample some data in cells with a minimum mass or number density.

    Args:
        base (str): Base path to input files.

        label (str): Label of data to sample.

    Keyword Args:
        output (str, optional): Write sample data to an output file.

        dt (float, optional): Time difference between input maps.

        sum (bool, optional): Sample the sum of the quantity in the system.

        cutoff (float, optional): Cutoff for data to sample bins of.
            Defaults to midpoint value.

        cutoff_label (str, optional): Label for data to use as cutoff,
            defaults to no cutoff.

        begin (int, default=1): First data map number.

        end (int, default=inf): Final data map number.

        ext (str, default='.dat'): File extension.

    Returns:
        times, values: 2-tuple with lists of sample data times and values.

    """

    def prepare_output(output, label, cutoff, cutoff_label, header_opts, fopts):
        header_opts.update(fopts)
        write_header(output, base, label, cutoff, cutoff_label, header_opts)

    fopts = pop_fileopts(kwargs)
    files = list(find_datamap_files(base, **fopts))

    if output:
        try:
            prepare_output(output, label, cutoff, cutoff_label, kwargs.copy(), fopts)
        except PermissionError:
            print("[WARNING] Output disabled: could not open '%s' for writing."
                % output)
            output = None

    sampled_values = []

    quiet = kwargs.pop('quiet', False)
    if not quiet:
        widgets = ['Sampling \'%s\' from files: ' % label,
                pbar.Bar(), ' (', pbar.SimpleProgress(), ') ', pbar.ETA()]
        progress = pbar.ProgressBar(widgets=widgets, maxval=len(files))
        progress.start()

    for i, (data, _, _) in enumerate(read_from_files(*files)):
        try:
            value = sample_value(data, label, cutoff, cutoff_label, sum)
        except KeyError:
            print("[ERROR] Bad label: no data with label '%s' in system." % label)
            return [], []
        except Exception as err:
            print("Encountered exception: %r" % err)
            return [], []

        if output:
            with open(output, 'a') as fp:
                fp.write('%.3f %.6f\n' % (i*dt, value))

        sampled_values.append(value)

        if not quiet:
            progress.update(i+1)

    if not quiet:
        progress.finish()

    times = [i*dt for i in range(len(sampled_values))]

    return times, sampled_values


def sample_value(data, label, cutoff, cutoff_label, sum):
    """Sample input data of label."""

    if label == 'inertial_energy':
        flow = FlowData(('U', data['U']), ('V', data['V']), ('M', data['M']))
        sample_data = sample_inertial_energy(flow)
    else:
        sample_data = data[label]

    if cutoff_label != None:
        if cutoff == None:
            cutoff = 0.5*(np.max(sample_data) + np.min(sample_data))

        try:
            inds = data[cutoff_label] >= cutoff
            sample_data = sample_data[inds]
        except KeyError:
            print("[WARNING] Bad label: cutoff label '%s' not in system, disabling cutoff"
                    % cutoff_label)

    if not sum:
        value = np.mean(sample_data)
    else:
        value = np.sum(sample_data)

    return value


@prepare_path
def write_header(output_path, input_base, label, cutoff, cutoff_label, kwargs):
    """Verify that output path is writable and write header."""

    title = "Sample average of data from a simulation"
    write_module_header(output_path, __name__, title)

    with open(output_path, 'a') as fp:
        inputs = (
                "# Input:\n"
                "#   Sample data label: %r\n"
                "#   File base path: %r\n"
                "#   Begin, end: %r, %r\n"
                "#   Delta-t: %r\n"
                "#   Cut-off: %r\n"
                "#   Cut-off label: %r\n"
                "# \n"
                "# Time (ps) %s\n"
                % (label, os.path.realpath(input_base),
                    kwargs.get('begin', None), kwargs.get('end', None),
                    kwargs.get('dt', 1.), cutoff,
                    cutoff_label, label
                    ))

        fp.write(inputs)

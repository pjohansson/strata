import os
import numpy as np
import progressbar as pbar

from droplets.flow import FlowData
from droplets.sample import sample_inertial_energy, sample_viscous_dissipation, sample_flow_angle
from strata.dataformats.read import read_from_files
from strata.utils import find_datamap_files, pop_fileopts, prepare_path, write_module_header


def sample_average_files(base, labels, output=None, sum=False, dt=1.,
        cutoff_label=None, cutoff=None, viscosity=8.77e-4, **kwargs):
    """Sample average collected data of input labels from files.

    Returns lists with input file times and the averaged sampled value
    of bins for the corresponding time.

    The labels have to be present in the read data files. If a label is
    'inertial_energy' the inertial energy is calculated for the system,
    in which case the data files must contain fields 'U' and 'V' for
    flow and 'M' for mass. If one label is 'visc_diss' the viscous
    dissipation is calculated for the system, which required fields
    'U' and 'V' for the flow, and 'X' and 'Y' for the coordinates. The
    output viscous dissipation is in energy per time and bin volume.
    The label 'flow_angle' samples the angle of flow along X and Y which
    requires the fields 'U' and 'V'.

    Optionally the total value of the quantity in the system can be returned
    by supplying the keyword argument `sum`.

    A cutoff value for bins to sample in can be supplied. Bins with values
    below this cutoff are not included in the sample average. By default
    the data label used as a cutoff is the input sample label, but a
    separate label can also be supplied. This is useful when wanting
    to sample some data in cells with a minimum mass or number density.

    Args:
        base (str): Base path to input files.

        labels (str's): Labels of data to sample. Can be a set.

    Keyword Args:
        output (str, optional): Write sample data to an output file.

        dt (float, optional): Time difference between input maps.

        sum (bool, optional): Sample the sum of the quantity in the system.

        cutoff (float, optional): Cutoff for data to sample bins of.
            Defaults to midpoint value.

        cutoff_label (str, optional): Label for data to use as cutoff,
            defaults to no cutoff.

        viscosity (float, optional): Viscosity of the liquid, used for
            the calculation of viscous dissipation.

        begin (int, default=1): First data map number.

        end (int, default=inf): Final data map number.

        ext (str, default='.dat'): File extension.

    Returns:
        times, values: 2-tuple with lists of sample data times and values.

    """

    def prepare_output(output, labels, cutoff, cutoff_label, header_opts, fopts):
        header_opts.update(fopts)
        write_header(output, base, labels, cutoff, cutoff_label, header_opts)

    fopts = pop_fileopts(kwargs)
    files = list(find_datamap_files(base, **fopts))

    if output:
        try:
            prepare_output(output, labels, cutoff, cutoff_label, kwargs.copy(), fopts)
        except PermissionError:
            print("[WARNING] Output disabled: could not open '%s' for writing."
                % output)
            output = None

    sampled_values = [[] for _ in labels]

    quiet = kwargs.pop('quiet', False)
    if not quiet:
        widgets = ['Sampling from files: ',
                pbar.Bar(), ' (', pbar.SimpleProgress(), ') ', pbar.ETA()]
        progress = pbar.ProgressBar(widgets=widgets, maxval=len(files))
        progress.start()

    for i, (data, info, _) in enumerate(read_from_files(*files)):
        flow = FlowData(*[(l, data[l]) for l in ['X', 'Y', 'U', 'V', 'M', 'N', 'T']], info=info)

        for j, label in enumerate(labels):
            try:
                value = sample_value(flow, label, cutoff, cutoff_label, sum, viscosity)
                sampled_values[j].append(value)
            except KeyError:
                print("[ERROR] Bad label: no data with label '%s' in system." % label)
                return
            except Exception as err:
                print("Encountered exception: %r" % err)
                return

        if output:
            with open(output, 'a') as fp:
                fp.write('%.3f ' % (i*dt))
                fp.write(' '.join(['%g' % values[-1] for values in sampled_values]))
                fp.write('\n')

        if not quiet:
            progress.update(i+1)

    if not quiet:
        progress.finish()

    times = [i*dt for i in range(len(sampled_values[0]))]

    return times, sampled_values


def sample_value(flow, label, cutoff, cutoff_label, sum, viscosity):
    """Sample input data of label."""

    if label == 'inertial_energy':
        sample_data = sample_inertial_energy(flow).ravel()
    elif label == 'visc_diss':
        sample_data = sample_viscous_dissipation(flow, viscosity).ravel()
    elif label == 'flow_angle':
        # If we are sampling the flow angle mean we do things a bit
        # differently and calculate it further below. We need the
        # full data set for it and thus keep all the data (minus the
        # data outside of the cutoff applied below).
        sample_data = flow.data
    else:
        sample_data = flow.data[label]

    if cutoff_label != None:
        if cutoff == None:
            cutoff = 0.5*(np.max(flow.data[cutoff_label]) + np.min(flow.data[cutoff_label]))

        try:
            inds = flow.data[cutoff_label] >= cutoff
            sample_data = sample_data[inds]
        except KeyError:
            print("[WARNING] Bad label: cutoff label '%s' not in system, disabling cutoff"
                    % cutoff_label)

    if not sum:
        if label == 'flow_angle':
            # We calculate the mean angle of the cut system, which we need
            # to recreate first as a FlowData object for the function.
            # This is a bit silly but the easiest way to do it and still
            # use the tested function in `droplets`.
            cut_flow = FlowData(*[(l, sample_data[l]) for l in ['U', 'V', 'M']])
            value = sample_flow_angle(cut_flow, mean=True, weight='M')
        else:
            value = np.mean(sample_data)

    else:
        if label == 'flow_angle':
            raise ValueError("Taking the sum of label `flow_angle` does not make any sense.")
        value = np.sum(sample_data)

    return value


@prepare_path
def write_header(output_path, input_base, labels, cutoff, cutoff_label, kwargs):
    """Verify that output path is writable and write header."""

    title = "Sample average of data from a simulation"
    write_module_header(output_path, __name__, title)

    with open(output_path, 'a') as fp:
        inputs = (
                "# Input:\n"
                "#   Sample data labels: %r\n"
                "#   File base path: %r\n"
                "#   Begin, end: %r, %r\n"
                "#   Delta-t: %r\n"
                "#   Cut-off: %r\n"
                "#   Cut-off label: %r\n"
                "# \n"
                "# Time (ps) %s\n"
                % (labels, os.path.realpath(input_base),
                    kwargs.get('begin', None), kwargs.get('end', None),
                    kwargs.get('dt', 1.), cutoff,
                    cutoff_label, ' '.join([l for l in labels])
                    ))

        fp.write(inputs)

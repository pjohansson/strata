import progressbar as pbar

from strata.dataformats.read import read_from_files
from strata.dataformats.write import write
from strata.utils import find_groups_to_singles, pop_fileopts


def average(base, output, group=1, **kwargs):
    """Average and output data maps from input base.

    Joins the file bases and extensions with a five-digit integer
    signifying the file number ('%s%05d%s').

    Output numbering of files will largely be compensated by the input
    frame numbering as K = ceil(N/M) where N is the first number of
    the current bundle and M is the bundle size.

    Args:
        base (str): Base path to input files.

        output (str): Base path to output files.

    Keyword Args:
        begin (int, default=1): First data map number.

        end (int, default=inf): Final data map number.

        group (int, default=1): Group input files in bundles of this length.
            Can be input as the third positional argument.

        ext (str, default='.dat'): File extension.

        quiet (bool, default=False): Do not print progress.

    """

    fopts = pop_fileopts(kwargs)
    quiet = kwargs.pop('quiet', False)

    groups_singles = list(find_groups_to_singles(base, output, group, **fopts))

    if not quiet:
        widgets = ['Averaging files: ',
                pbar.Bar(), ' (', pbar.SimpleProgress(), ') ', pbar.ETA()]
        progress = pbar.ProgressBar(widgets=widgets, maxval=len(groups_singles))
        progress.start()

    for i, (fn_group, fn_out) in enumerate(groups_singles):
        group_data = []
        used_modules = set([])

        for data, _, meta in read_from_files(*fn_group):
            group_data.append(data)
            used_modules.add(meta.pop('module'))

        # Assert that a single module was used and retrieve it
        assert (len(used_modules) == 1)
        module = used_modules.pop()

        avg_data = module.average_data(*group_data)
        write(fn_out, avg_data)

        if not quiet:
            progress.update(i+1)

    progress.finish()

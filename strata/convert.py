import progressbar as pbar
import strata

from strata.dataformats.read import read_data_file
from strata.dataformats.write import write
from strata.utils import find_singles_to_singles, pop_fileopts


def convert(base, output, **kwargs):
    """Convert files from one format to another.

    By default converts files to a simple binary format, this can be
    set using the keyword 'ftype' detailed below.

    Args:
        base (str): Base path to input files.

        output (str): Base path to output files.

    Keyword Args:
        begin (int, default=1): First data map number.

        end (int, default=inf): Final data map number.

        ftype (str, default='gmx'): File type to write. Choices:
            'gmx'          - Gromacs flow format (strata.dataformats.gmx_flow_version_1)
            'simple'       - Simple binary    (strata.dataformats.simple)
            'simple_plain' - Simple plaintext (strata.dataformats.simple)

        ext (str, default='.dat'): File extension.

        quiet (bool, default=False): Do not print progress.

    """

    fopts = pop_fileopts(kwargs)
    quiet = kwargs.pop('quiet', False)
    to_ftype = kwargs.pop('ftype', 'gmx')

    zip_files = list(find_singles_to_singles(base, output, **fopts))

    if not quiet:
        widgets = ['Converting files: ',
                pbar.Bar(), ' (', pbar.SimpleProgress(), ') ', pbar.ETA()]
        progress = pbar.ProgressBar(widgets=widgets, max_value=len(zip_files))
        progress.start()

    for i, (fnin, fnout) in enumerate(zip_files):
        data, info, metadata = read_data_file(fnin)

        write(fnout, data, info, ftype=to_ftype, **kwargs)

        if not quiet:
            progress.update(i+1)

    if not quiet:
        progress.finish()

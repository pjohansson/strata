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

        ftype (str, default='simple'): File type to write. Choices:
            'simple'       - Simple binary    (strata.dataformats.simple)
            'simple_plain' - Simple plaintext (strata.dataformats.simple)

        ext (str, default='.dat'): File extension.

    """

    fopts = pop_fileopts(kwargs)

    for fnin, fnout in find_singles_to_singles(base, output, **fopts):
        data, _, _ = read_data_file(fnin)
        write(fnout, data, **kwargs)

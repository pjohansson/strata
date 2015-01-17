import numpy as np
from strata.dataformats.read import read_from_files
from strata.dataformats.simple.main import average_data, write
from strata.utils import *

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

        end (int, default=inf): Final data map number. Can be input
            as the second positional argument.

        group (int, default=1): Group input files in bundles of this length.
            Can be input as the third positional argument.

        ext (str, default='.dat'): File extension.

    """

    opts = pop_fileopts(kwargs)

    for fn_group, fn_out in find_groups_to_singles(base, output, group, **opts):
        data = [d for d, _ in read_from_files(*fn_group)]
        avg_data = average_data(*data)
        write(fn_out, avg_data)

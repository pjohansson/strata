import numpy as np
import os

"""Utilities for interacting with data files."""


def gen_filenames(base, end=np.inf, **kwargs):
    """Generates a number of file names from a base.

    Joins the input base and extension with a five-digit integer
    signifying the file number ('%s%05d%s').

    Args:
        base (str): Base of data map files.

    Keyword Args:
        begin (int, default=1): First data map number.

        end (int, default=inf): Final data map number. Can be input
            as the second positional argument.

        ext (str, default='.dat'): File extension.

    Yields:
        str: File name.

    """

    num = kwargs.pop('begin', 1)
    ext = kwargs.pop('ext', '.dat')

    while num <= end:
        yield('%s%05d%s' % (base, num, ext))
        num += 1


def find_datamap_files(base, **kwargs):
    """Generates data map file names found at input path base.

    If the keyword 'group' is specified file names are yielded as
    bundled lists of input length. If not enough files are found to
    fill a group the remainder is cut off and not yielded.

    Finds file names by joining the file name base and extension with
    a five-digit integer signifying the map number ('%s%05d%s').

    Args:
        base (str): Base of data map files.

    Keyword Args:
        begin (int, default=1): First data map numbering to read.

        end (int, default=inf): Final data map numbering to read.

        group (int, default=1): Yield the file names bundled in groups
            of this length.

        ext (str, default='.dat'): File extension.

    Yields:
        str, list(str): Found file names in singles or bundles.

    """

    def yield_singles(base, begin, end, ext):
        for filename in gen_filenames(base, begin=begin, end=end, ext=ext):
            if os.access(filename, os.F_OK) == True:
                yield(filename)
            else:
                break

    def yield_groups(base, begin, end, ext):
        group_end = begin + group - 1

        while group_end <= end:
            files = list(yield_singles(base, begin, group_end, ext))

            if len(files) == group:
                yield(files)
                begin += group
                group_end += group
            else:
                break

    args = [
            kwargs.pop('begin', 1),
            kwargs.pop('end', np.inf),
            kwargs.pop('ext', '.dat')
            ]
    group = kwargs.pop('group', 1)

    if group == 1:
        yield from yield_singles(base, *args)
    else:
        yield from yield_groups(base, *args)

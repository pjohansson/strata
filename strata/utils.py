import numpy as np
import os

"""Utilities for interacting with data files."""

def find_datamap_files(base, **kwargs):
    """Generates data map file names at input path base.

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

    def yield_singles(base, num, end, ext):
        def get_filename(num):
            return os.path.join(directory, '%s%05d%s' % (fnbase, num, ext))

        directory, fnbase = os.path.split(base)
        fn = get_filename(num)

        while os.access(fn, os.F_OK) == True and num <= end:
            yield(fn)
            num += 1
            fn = get_filename(num)

    def yield_groups(base, num, end, ext):
        group_end = num + group - 1
        while group_end <= end:
            files = list(yield_singles(base, num, group_end, ext))

            if len(files) == group:
                yield(files)
            else:
                break
            num += group
            group_end += group

    begin = kwargs.pop('begin', 1)
    end = kwargs.pop('end', np.inf)
    ext = kwargs.pop('ext', '.dat')
    group = kwargs.pop('group', 1)

    if group == 1:
        yield from yield_singles(base, begin, end, ext)
    else:
        yield from yield_groups(base, begin, end, ext)

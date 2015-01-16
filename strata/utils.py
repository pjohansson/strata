import numpy as np
import os

def find_datamap_files(base, **kwargs):
    """Returns a list of found data map files at input path base.

    Joins the file name base and extension with a five-digit int
    signifying the map number ('%s%05d%s').

    Args:
        base (str): Base of data map files.

    Keyword Args:
        ext (str, default='.dat'): File extension.

        begin (int, default=1): First data map numbering to read.

        end (int, default=inf): Final data map numbering to read.

    Returns:
        list: List of found files, empty if none are found.

    """

    def get_filename(num):
        return os.path.join(directory, '%s%05d%s' % (fnbase, num, ext))

    num = kwargs.pop('begin', 1)
    end = kwargs.pop('end', np.inf)
    ext = kwargs.pop('ext', '.dat')

    directory, fnbase = os.path.split(base)
    fn = get_filename(num)

    paths = []
    while os.access(fn, os.F_OK) == True and num <= end:
        paths.append(fn)
        num += 1
        fn = get_filename(num)

    return paths

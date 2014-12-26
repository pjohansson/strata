import dataformats.simple.read as smp

def guess_read_function(filename):
    """Return a function handle to read a file.

    Makes a guess on the data format by trying to access the file and assess
    its characteristics, judging which of the implemented data formats is
    the best match. Defaults to dataformats.simple.read_plainsimple if
    file is not binary.

    Args:
        filename (str): File to read.

    Returns:
        function: Function handle.

    """

    def is_binary(filename, checksize=512):
        """Return True if a file is binary-like, else False.

        Tries to find a unicode '\n' character within checksize bytes
        of the file and catches a UnicodeDecodeError if raised.

        Args:
            filename (str): File to control.

            checksize (int): Number of bytes to control over.

        Returns:
            bool: Whether file is binary-like or not.

        """

        with open(filename, 'r') as fp:
            try:
                if '\n' in fp.read(checksize):
                    return False
                else:
                    raise UnicodeDecodeError
            except UnicodeDecodeError:
                return True

    if is_binary(filename):
        function = smp.read_binsimple
    else:
        function = smp.read_plainsimple

    return function


def read_flow_data(filename):
    """Return data and information about a flow field map.

    Data and information are separate dict's returned as tuple. The data
    dictionary structure is field names as keys and data in numpy.ndarrays
    as values:

        data = {'X': numpy.ndarray([0, 1, 2, ...]), ...}

    The information dictionary always contains the same information:

        info = {
            'shape': tuple of number of bins in X and Y.
            'size': dict with 'X' and 'Y' tuples of minimum
                    and maximum positions along the axis.
            'bin_size': tuple of bin size in X and Y.
            'num_bins': number of bins in system.
            }

    Args:
        filename (str): File to read data from.

    Returns:
        (dict, dict): data and information dictionaries.

    """

    function = guess_read_function(filename)
    data, info = function(filename)

    return data, info

import strata.dataformats as formats

"""Module for reading flow field data from specific file formats.

File formats are implemented as submodules, mainly to be called from
this module through the 'read_flow_data' function. This function calls
'guess_read_module' to determine the file format, then uses the returned
handle to call the correct submodule.

"""

def guess_read_module(filename):
    """Return a handle of a module to read the input file.

    Makes a guess on the data format by trying to access the file and assess
    its characteristics, judging which of the implemented data formats is
    the best match.

    Currently only one module is implemented: A very simple format.

    Args:
        filename (str): File to read.

    Returns:
        module: Module handle.

    """

    return formats.simple.main


def read_from_files(*files):
    """Yield data and information from a set of files to read.

    Args:
        files (str's): File names to yield data from, one per argument.

    Yields:
        (dict, dict): Data and information dictionaries.
            See read_flow_data for details.

    """

    for filename in files:
        yield read_flow_data(filename)


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

    module = guess_read_module(filename)
    data, info = module.read(filename)

    return data, info

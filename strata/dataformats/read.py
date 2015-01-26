import strata.dataformats as formats

"""Module for reading flow field data from specific file formats.

File formats are implemented as submodules, mainly to be called from
this module through the 'read_data_file' and 'read_files' functions.
These function call the 'guess_read_module' to determine the file format
and uses the returned handle to call the correct submodule.

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
        (dict, dict, module): 3-tuple of dict's with read data and
            information and a handle to the used read module. See
            read_data_file for details.

    """

    for filename in files:
        yield read_data_file(filename)


def read_data_file(filename):
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

    The returned metadata contains:

        metadata = {
            'module': handle to the module used for reading,
            'path': path to read file
            }

    Args:
        filename (str): File to read data from.

    Returns:
        (dict, dict, module): 3-tuple of dict's with read data and information
            from the data map and one with metadata.

    """

    module = guess_read_module(filename)
    data, info = module.read_data(filename)

    metadata = {'path': filename, 'module': module}

    return data, info, metadata

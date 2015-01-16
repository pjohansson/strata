import numpy as np

"""Read data from simple, brute force file formats."""

def calc_information(X, Y):
    """Return a dict of system information calculated from input cell positions.

    Calculates system size ('size'), bin sizes ('bin_size'), number of cells
    ('num_bins') and shape ('shape').

    Args:
        X, Y (array_like): Arrays with cell positions.

    Returns:
        dict: Information about system in dictionary.

    """

    if len(X) != len(Y):
        raise ValueError("Lengths of X and Y arrays not equal")

    xs, ys = map(np.unique, (X, Y))

    info = {}
    info['num_bins'] = len(X)
    info['shape'] = [len(a) for a in (xs, ys)]
    info['bin_size'] = [a[1] - a[0]
            if len(a) > 1 else 2*a[0]
            for a in (xs, ys)]
    info['size'] = {v: [np.min(a), np.max(a)]
            for v, a in (('X', xs), ('Y', ys))}

    return info


def read_binsimple(filename):
    """Return data and information read from a simple binary format.

    Args:
        filename (str): A file to read data from.

    Returns:
        (dict, dict): Dictionaries with data and information. See
                      dataformats.read.read_flow_data() for more information.

    """

    def read_data(filename):
        # Fixed field order of format
        fields = ['X', 'Y', 'N', 'T', 'M', 'U', 'V']
        raw_data = np.fromfile(filename, dtype='float32')

        # Unpack into dictionary
        data = {}
        stride = len(fields)
        for i, field in enumerate(fields):
            data[field] = raw_data[i::stride]

        return data

    data = read_data(filename)
    info = calc_information(data['X'], data['Y'])

    return data, info


def read_plainsimple(filename):
    """Return field data from a simple plaintext format.

    Args:
        filename (str): A file to read data from.

    Returns:
        (dict, dict): Dictionaries with data and information. See
                      dataformats.read.read_flow_data() for more information.


    """

    def read_data(filename):
        raw_data = np.genfromtxt(filename, names=True)

        # Unpack into dictionary
        data = {}
        for field in raw_data.dtype.names:
            data[field] = raw_data[field]

        return data

    data = read_data(filename)
    info = calc_information(data['X'], data['Y'])

    return data, info
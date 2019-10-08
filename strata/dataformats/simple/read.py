import numpy as np

"""Read data from simple, naive file formats."""

def read_data(filename, decimals=5):
    """Read field data from a file name.

    Determines which of the simple formats in this module to use and
    returns data read using the proper function.

    Coordinates are rounded to input number of decimals.

    Args:
        filename (str): A file to read data from.

    Keyword Args:
        decimals (int): Number of decimals for coordinates.

    Returns:
        (dict, dict): 2-tuple of dict's with data and information. See
            strata.dataformats.read.read_data_file for more information.

    """

    def guess_read_function(filename):
        """Return handle to binary or plaintext function."""

        def is_binary(filename, checksize=512):
            with open(filename, 'r') as fp:
                try:
                    fp.read(checksize)
                    return False
                except UnicodeDecodeError:
                    return True

        if is_binary(filename):
            return read_binsimple
        else:
            return read_plainsimple

    read_function = guess_read_function(filename)
    data = read_function(filename)
    for coord in ['X', 'Y']:
        data[coord] = data[coord].round(decimals)

    info = calc_information(data['X'], data['Y'])

    x0, y0 = info['origin']
    dx, dy = info['spacing']
    nx, ny = info['shape']

    x = x0 + dx * np.arange(nx, dtype=np.float64)
    y = y0 + dy * np.arange(ny, dtype=np.float64)

    xs, ys = np.meshgrid(x, y, indexing='ij')

    data['X'] = xs.ravel()
    data['Y'] = ys.ravel()

    return data, info


def calc_information(X, Y):
    """Return a dict of system information calculated from input cell positions.

    Calculates system origin ('origin'), bin spacing ('spacing'), number of cells
    ('num_bins') and shape ('shape').

    Args:
        X, Y (array_like): Arrays with cell positions.

    Returns:
        dict: Information about system in dictionary.

    """

    def calc_shape(X, Y):
        data = np.zeros((len(X), ), dtype=[('X', np.float), ('Y', np.float)])
        data['X'] = X
        data['Y'] = Y

        data.sort(order=['Y', 'X'])

        y0 = data['Y'][0]
        nx = 1

        while np.abs(data['Y'][nx] - y0) < 1e-4:
            nx += 1

        ny = len(X) // nx

        return nx, ny

    def calc_spacing(X, Y, nx, ny):
        def calc_1d(xs, n):
            x0 = np.min(xs)
            x1 = np.max(xs)

            try:
                return (x1 - x0) / (n - 1)
            except:
                return 0.0

        dx = calc_1d(X, nx)
        dy = calc_1d(Y, ny)

        return dx, dy

    def calc_origin(X, Y):
        return X.min(), Y.min()

    if len(X) != len(Y):
        raise ValueError("Lengths of X and Y arrays not equal")

    nx, ny = calc_shape(X, Y)
    dx, dy = calc_spacing(X, Y, nx, ny)
    x0, y0 = calc_origin(X, Y)

    info = {
        'shape': (nx, ny),
        'spacing': (dx, dy),
        'num_bins': nx * ny,
        'origin': (x0, y0),
    }

    return info


def read_binsimple(filename):
    """Return data and information read from a simple binary format.

    Args:
        filename (str): A file to read data from.

    Returns:
        dict: Data with field labels as keys.

    """

    def read_file(filename):
        # Fixed field order of format
        fields = ['X', 'Y', 'N', 'T', 'M', 'U', 'V']
        raw_data = np.fromfile(filename, dtype='float32')

        # Unpack into dictionary
        data = {}
        stride = len(fields)
        for i, field in enumerate(fields):
            data[field] = raw_data[i::stride]

        return data

    data = read_file(filename)

    return data


def read_plainsimple(filename):
    """Return field data from a simple plaintext format.

    Args:
        filename (str): A file to read data from.

    Returns:
        dict: Data with field labels as keys.
    """

    def read_file(filename):
        raw_data = np.genfromtxt(filename, names=True)

        # Unpack into dictionary
        data = {}
        for field in raw_data.dtype.names:
            data[field] = raw_data[field]

        return data

    data = read_file(filename)

    return data

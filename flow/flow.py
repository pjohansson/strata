import numpy as np

class FlowData(object):
    """Container for flow field data.

    Flow fields consist of a number of 2D bins, whose positions are
    specified in 1D arrays. Various data for these bins can be added
    and accessed through a common interface.

    Positions and flow velocities (if present) can be accessed through
    methods:

        self.X
        self.Y
        self.U
        Self.V

    Any field, including the above, can be added and accessed through
    a common data structure:

        self.add_data(label='M', data=M)
        self.data['M']

    Args:
        input_data (dict): Initialise with data contained in this dict,
            where keys are data labels and values are 1D array_like objects.

        info (dict, optional): Dict with system information.

    Example:
        import numpy as np

        xv, yv = np.arange(5), np.arange(8)
        X, Y = np.meshgrid(xv, yv, indexing='ij')
        U, V = np.sin(3*X), np.exp(Y-3)
        M = 100 - np.cos(12*X*Y)

        data = {'X': X, 'Y': Y, 'U': U, 'V': V, 'mass': M}
        info = {
            'shape': [len(xv), len(yv)],
            'bin_size': [xv[1]-xv[0], yv[1]-yv[0]],
            'size': {'X': [xv[0], xv[-1]], 'Y': [yv[0], yv[-1]]},
            'num_bins': len(X)
            }

        flow = FlowData(data, info)
        flow.X is X
        flow.V is V
        flow.data['mass'] is M

    """

    def __init__(self, input_data, info={}):
        self.set_data(input_data)
        self.set_info(info)
        return

    @property
    def properties(self):
        """Return list of data parameters."""
        return self.data.dtype.names

    @property
    def X(self):
        """Return array with label 'X', popularly used as a bin position."""
        return self.get_data('X')

    @property
    def Y(self):
        """Return array with label 'Y', popularly used as a bin position."""
        return self.get_data('Y')

    @property
    def U(self):
        """Return array with label 'U', popularly used for mass flow along 'X'."""
        return self.get_data('U')

    @property
    def V(self):
        """Return array with label 'V', popularly used for mass flow along 'Y'."""
        return self.get_data('V')

    def get_data(self, label):
        """Return data for a parameter label."""

        return self.data[label] if label in self.properties else None

    def set_data(self, input_data):
        """Create and set a data record from input data.

        Args:
            input_data (dict): Dictionary whose keys are parameter data
                labels and values are 1D array_likes containing the data.
                Data arrays must be of equal length.

        """

        num_data = len(input_data)
        len_data = len(input_data['X'].ravel())
        dtype = np.dtype([(i, 'float') for i in input_data.keys()])
        self.data = np.zeros((num_data,len_data), dtype=dtype)

        for label, bindata in input_data.items():
            self.data[label] = bindata.ravel()

    def set_info(self, info):
        """Set system information properties.

        Args:
            info (dict): Dictionary containing system information as keywords:
                'shape' (2-tuple): Number of cells in dimension 1 and 2.
                'size' (dict): {'X': (min(X), max(X)), 'Y': (min(Y), max(Y))}
                'bin_size' (2-tuple): Bin size in dimension 1 and 2.
                'num_bins' (int): Number of bins.

        """

        self.shape = info.get('shape', None)
        self.size = info.get('size', {'X': [None, None], 'Y': [None, None]})
        self.bin_size = info.get('bin_size', [None, None])
        self.binx = self.bin_size[0]
        self.biny = self.bin_size[1]
        self.num_bins = info.get('num_bins', None)

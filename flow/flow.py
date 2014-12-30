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
            'size': ([xv[0], xv[-1]], [yv[0], yv[-1]]),
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
    def bin_size(self):
        """Size of a bin in the system as a 2-tuple."""

        return self._bin_size

    @bin_size.setter
    def bin_size(self, bin_size):
        try:
            if np.shape(bin_size) == (2,):
                self._bin_size = tuple(float(val) for val in bin_size)
            elif bin_size == None:
                self._bin_size = (None, None)
            else:
                raise TypeError
        except TypeError:
            raise TypeError("bin_size must be two int's")
        except ValueError:
            raise ValueError ("bin_size must be two int's")

    @property
    def num_bins(self):
        """Number of bins in the system."""

        return self._num_bins

    @num_bins.setter
    def num_bins(self, num_bins):
        try:
            if num_bins == None:
                self._num_bins = None
            else:
                self._num_bins = int(num_bins)
        except TypeError:
            raise TypeError("num_bins must be a single int")
        except ValueError:
            raise ValueError("num_bins must be a single int")

    @property
    def size(self):
        """System boundaries as a (2,2)-tuple."""

        return self._size

    @size.setter
    def size(self, size):
        try:
            if np.shape(size) == (2,2):
                self._size = tuple((float(i), float(j)) for i, j in size)
            elif size == None:
                self._size = ((None, None), (None, None))
            else:
                raise TypeError
        except TypeError:
            raise TypeError("size must be (2,2)-tuple of int's")
        except ValueError:
            raise ValueError ("size must be 2-by-2 int's")

    @property
    def shape(self):
        """Shape of the system as a 2-tuple."""

        return self._shape

    @shape.setter
    def shape(self, shape):
        try:
            if np.shape(shape) == (2,):
                self._shape = tuple(int(val) for val in shape)
            elif shape == None:
                self._shape = (None, None)
            else:
                raise TypeError
        except TypeError:
            raise TypeError ("shape must be 2-tuple")
        except ValueError:
            raise ValueError ("shape must be two int's")

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

        # Get size of data
        num_data = len(input_data)
        key = list(input_data.keys())[0]
        sizeof = np.size(input_data[key])

        # Create dtype for array
        dtype = np.dtype([(i, 'float') for i in input_data.keys()])

        # Allocate and fill structured array
        self.data = np.zeros((num_data,sizeof), dtype=dtype)

        try:
            for label, bindata in input_data.items():
                self.data[label] = np.array(bindata).ravel()
        except ValueError:
            raise ValueError("added array_like objects not all of equal size.")


    def set_info(self, info):
        """Set system information properties.

        Values default to None (in corresponding tuple form) if not entered.

        Args:
            info (dict): Dictionary containing system information as keywords:
                'shape' (2-tuple): Number of cells in dimension 1 and 2.
                'size' ((2,2)-tuple): System boundaries in dimension 1 and 2.
                'bin_size' (2-tuple): Bin size in dimension 1 and 2.
                'num_bins' (int): Number of bins.

        """

        self.shape = info.get('shape', None)
        self.size = info.get('size', None)
        self.bin_size = info.get('bin_size', None)
        self.binx = self.bin_size[0]
        self.biny = self.bin_size[1]
        self.num_bins = info.get('num_bins', None)

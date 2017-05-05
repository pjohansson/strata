import numpy as np


class FlowData(object):
    """Container for flow field data.

    Flow fields consist of a number of 2D bins, whose positions are
    specified in 1D arrays. Various data for these bins can be added
    and accessed through a common interface:

        self.data['X']
        self.data['Y']

    Args:
        input_data (2-tuples, dict's): Each input argument must be
            either a dict or 2-tuple object with labeled data:
            {label: data} or (label, data). All data arrays must be
            of equal length.

    Keyword Args:
        info (dict, optional): Dict with system information.

        dtype (data-type, optional): The desired Numpy data-type of record.

    Example:
        import numpy as np

        xv, yv = np.arange(5), np.arange(8)
        X, Y = np.meshgrid(xv, yv, indexing='ij')
        U, V = np.sin(3*X), np.exp(Y-3)
        M = 100 - np.cos(12*X*Y)

        info = {
            'shape': [len(xv), len(yv)],
            'spacing': [xv[1]-xv[0], yv[1]-yv[0]],
            'size': ([xv[0], xv[-1]], [yv[0], yv[-1]]),
            'num_bins': len(X)
            }

        data = [('X', X), ('Y', Y), ('U', U), ('V', V), ('mass', M)]
        flow = FlowData(*data, info=info)
        flow.data['X'] is X
        flow.data['V'] is V
        flow.data['mass'] is M

    """

    def __init__(self, *input_data, **kwargs):
        self.set_data(*input_data, **kwargs)
        self.set_info(kwargs.pop('info', {}))
        return


    @property
    def num_bins(self):
        """Number of bins in the system."""

        return self._num_bins


    @num_bins.setter
    def num_bins(self, num_bins):
        try:
            assert num_bins != None
            self._num_bins = int(num_bins)
        except AssertionError:
            self._num_bins = None
        except Exception:
            raise ValueError("num_bins must be a single int")


    @property
    def origin(self):
        """Origin of coordinates as a 2-tuple."""

        return self._origin


    @origin.setter
    def origin(self, origin):
        try:
            assert origin not in [None, (None, None)]
            self._origin = tuple(float(origin[i]) for i in (0, 1))
        except AssertionError:
            self._origin = (None, None)
        except Exception:
            raise ValueError("origin must be of form (float, float)")


    @property
    def shape(self):
        """Shape of the system as a 2-tuple."""

        return self._shape


    @shape.setter
    def shape(self, shape):
        try:
            assert shape not in [None, (None, None)]
            self._shape = tuple(int(shape[i]) for i in (0, 1))
        except AssertionError:
            self._shape = (None, None)
        except Exception:
            raise ValueError("shape must be of form (int, int)")


    @property
    def spacing(self):
        """Spacing of bins in the system as a 2-tuple."""

        return self._bin_spacing


    @spacing.setter
    def spacing(self, spacing):
        try:
            assert spacing not in [None, (None, None)]
            self._bin_spacing = tuple(float(spacing[i]) for i in (0, 1))
        except AssertionError:
            self._bin_spacing = (None, None)
        except Exception:
            raise ValueError("spacing must be two floats")


    @property
    def properties(self):
        """Return list of data parameters."""
        return self.data.dtype.names


    def copy(self):
        """Return a deep copy of the FlowData object."""

        data_array = [(l, self.data[l]) for l in self.data.dtype.names]
        return FlowData(*data_array, info=self._info)


    def get_data(self, label):
        """Return data for a parameter label."""

        return self.data[label] if label in self.properties else None


    def cut(self, xlim=(None, None), ylim=(None, None)):
        max_coords = [o + n * dx
                for o, n, dx in zip(self.origin, self.shape, self.spacing)
                ]
        xmin, xmax = xlim
        xmin = xmin if xmin != None else self.origin[0]
        xmax = xmax if xmax != None else max_coords[0]
        flow = self.lims('X', xmin, xmax)

        ymin, ymax = ylim
        ymin = ymin if ymin != None else self.origin[1]
        ymax = ymax if ymax != None else max_coords[1]
        flow = flow.lims('Y', ymin, ymax)

        shape = [len(np.unique(flow.data[l])) for l in ('X', 'Y')]

        info = {
            'spacing': self.spacing,
            'origin': (xmin, ymin),
            'shape': shape,
            'num_bins': shape[0] * shape[1]
        }
        flow.set_info(info)

        return flow


    def lims(self, label, vmin, vmax):
        """Return new FlowData object with input data limits.

        This method cuts bins from the system abject to the input limits
        and returns a new object with the remaining bins. There is no
        guarantee that the created data set is on a regular grid and
        thus the `shape` and `origin` properties are unset. `spacing`
        is unchanged and `num_bins` is updated to the correct number.

        Since a new object is returned this method can be chained to
        select along many different data at once.

        Args:
            label (str): Label of data to limit values for.

            vmin/vmax (float/None): Minimum and maximum values of input
                label to include in the returned object. Either or both
                values can be `None` to not apply a cut in that direction.

        Return:
            FlowData: New object with selected bins.

        """

        vmin = vmin if vmin != None else -np.inf
        vmax = vmax if vmax != None else np.inf

        try:
            inds = (self.data[label] >= vmin) & (self.data[label] <= vmax)
        except ValueError:
            raise KeyError("FlowData object has no data with input label %r" % label)
        except TypeError:
            raise TypeError("bad input limits (%r, %r): must be float or None" % (vmin, vmax))

        data = self.data[inds]

        info = {
            'num_bins': data.size
        }

        if self.spacing != (None, None):
            info['spacing'] = self.spacing

        data_list = [(l, data[l]) for l in data.dtype.names]

        return FlowData(*data_list, info=info, dtype=data.dtype)


    def set_data(self, *data, **kwargs):
        """Create and set a data record from input data.

        Args:
            input_data (2-tuples, dict's): Each input argument must be
                either a dict or 2-tuple object with labeled data:
                {label: data} or (label, data). All data arrays must be
                of equal length.

        Keyword Args:
            dtype (data-type, optional): The desired Numpy data-type of record.
                If a single dtype_like, all fields are cast to that type.
                Can also be a complex data-type if the fields catch all
                labels of the input data.

        """

        def collate_input_data(input_data):
            """Return variable input data as a list of 2-tuples.

            """

            def add_data(label, values):
                collated_data.append((label, values))

            collated_data = []
            for data in input_data:
                if type(data) == tuple:
                    add_data(*data)
                elif type(data) == dict:
                    for label, values in data.items():
                        add_data(label, values)
                else:
                    raise TypeError

            return collated_data

        def get_dtype(input_data, dtype=None):
            """Return a dtype for the record.

            Reads the dtype of array_likes in input data and creates
            a record dtype. If several different types are present
            in the array_like a mixed type record is returned.

            """

            # If explicit data-type fields are set, return them
            if len(np.dtype(dtype)) != 0:
                return np.dtype(dtype)

            # Construct data-type for record from given or present dtypes
            types = []
            for label, values in input_data:
                if dtype:
                    atype = dtype
                else:
                    try:
                        atype = values.dtype
                    except AttributeError:
                        atype = type(values[0])
                types.append((label, atype))

            return np.dtype(types)

        try:
            data_list = collate_input_data(data)
        except TypeError:
            raise TypeError("input must be 2-tuples or dict's")

        # Get size of data
        num_data = len(data_list)
        sizeof = np.size(data_list[0][1])

        # Get record data-type and allocate memory
        array_type = get_dtype(data_list, kwargs.pop('dtype', None))
        self.data = np.zeros((1,sizeof), dtype=array_type).ravel()

        try:
            for label, bindata in data_list:
                self.data[label] = np.array(bindata).ravel()
        except ValueError:
            raise ValueError("added array_like objects not all of equal size.")


    def set_info(self, info):
        """Set system information properties.

        Values default to None (in corresponding tuple form) if not entered.

        Args:
            info (dict): Dictionary containing system information as keywords:
                'shape' (2-tuple): Number of cells in dimension 1 and 2.
                'origin' (2-tuple): System origin in dimension 1 and 2.
                'spacing' (2-tuple): Bin spacings in dimension 1 and 2.
                'num_bins' (int): Number of bins.

        """

        info_copy = info.copy()

        self.shape = info_copy.pop('shape', None)
        self.origin = info_copy.pop('origin', None)
        self.spacing = info_copy.pop('spacing', None)
        self.num_bins = info_copy.pop('num_bins', None)

        if info_copy != {}:
            bad_item = info_copy.popitem()
            raise KeyError("Unknown key/value pair for system information: %r, %r"
                           % (bad_item[0], bad_item[1]))


    def translate(self, label, value):
        """Return a copy with the data of input label translated by input value.

        Args:
            label: Data label to translate.

            value: Translate by this. Can be an array in which case it must
                broadcastable to the data.

        Returns:
            FlowData: New object with translated data.

        """

        flow = self.copy()

        try:
            flow.data[label] += value
        except ValueError as exc:
            if 'broadcast' in str(exc):
                raise ValueError(exc)
            else:
                raise KeyError("No label %r in object")

        return flow


    @property
    def _info(self):
        """Access to the set properties of the object."""

        return {
            'num_bins': self.num_bins,
            'origin': self.origin,
            'shape': self.shape,
            'spacing': self.spacing
        }

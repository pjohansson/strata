import numpy as np
from flow.flow import *

X = np.arange(9)
Y = np.arange(9)

good_info = {
        'shape': [len(X), len(Y)],
        'size': (
            (min(X), max(X)),
            (min(Y), max(Y))
            ),
        'bin_size': (X[1]-X[0], Y[1]-Y[0]),
        'num_bins': len(X)*len(Y)
        }

def test_set_info():
    flow = FlowData({'X': X, 'Y': Y}, good_info)

    assert (np.shape(flow.shape) == (2,))
    assert (flow.shape == good_info['shape'])

    assert (np.shape(flow.bin_size) == (2,))
    assert (flow.bin_size == good_info['bin_size'])
    assert (flow.binx == good_info['bin_size'][0])
    assert (flow.biny == good_info['bin_size'][1])

    assert (np.shape(flow.size) == (2,2))
    assert (flow.size[0] == good_info['size'][0])
    assert (flow.size[1] == good_info['size'][1])

    assert (type(flow.num_bins) == int)
    assert (flow.num_bins == good_info['num_bins'])

import numpy as np
from flow.flow import FlowData
from flow.flowfield import *

def test_get_indices_lims():
    X = np.arange(8)    #  0    1     2     3    4     5     6     7
    Y = np.arange(8)*2  #  0    2     4     6    8    10    12    14
    U = 1/(X+1)         #  1  0.5  0.33  0.25  0.2  0.17  0.14  0.13
    V = np.log(Y+1)-2   # -2 -0.9  -0.4 -0.05 0.19  0.39  0.56  0.70

    flow = FlowData({'X': X, 'Y': Y, 'U': U, 'V': V})

    lims = {'X': (2, 5)}
    indices = get_lim_indices(flow.data, lims)
    assert (np.array_equiv(indices, [2, 3, 4, 5]))

    lims = {'X': (2, None), 'Y': (7, None)}
    indices = get_lim_indices(flow.data, lims)
    assert (np.array_equiv(indices, [4, 5, 6, 7]))

    lims = {'X': (None, np.inf), 'U': (-np.inf, 0.4)}
    indices = get_lim_indices(flow.data, lims)
    assert (np.array_equiv(indices, [2, 3, 4, 5, 6, 7]))

    lims = {'X': (-3, 4), 'Y': (None, None), 'U': (None, 0.8), 'V': (-1, 0)}
    indices = get_lim_indices(flow.data, lims)
    assert (np.array_equiv(indices, [1, 2, 3]))

def test_get_indices_other_formats():
    X = np.arange(8)
    flow = FlowData({'X': X})

    lims = {'X': (4, 2)}
    indices = get_lim_indices(flow.data, lims)
    assert (np.array_equiv(indices, []))

    lims = {'X': [2, 4]}
    indices = get_lim_indices(flow.data, lims)
    assert (np.array_equiv(indices, [2, 3, 4]))

    lims = {'X': np.array([2, 4])}
    indices = get_lim_indices(flow.data, lims)
    assert (np.array_equiv(indices, [2, 3, 4]))

    lims = {}
    indices = get_lim_indices(flow.data, lims)
    assert (np.array_equiv(indices, np.arange(8)))

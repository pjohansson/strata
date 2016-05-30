import numpy as np
import pytest
from droplets.flow import *

def test_set_data():
    xs = np.arange(9)
    ys = np.arange(5)

    x, y = np.meshgrid(xs, ys, indexing='ij')
    u = np.sin(x)**2
    v = -np.cos(3*y)

    flow = FlowData({'X': x, 'Y': y, 'U': u, 'V': v})

    # Check data structure
    assert (np.array_equal(flow.data['X'], x.ravel()))
    assert (np.array_equal(flow.data['Y'], y.ravel()))
    assert (np.array_equal(flow.data['U'], u.ravel()))
    assert (np.array_equal(flow.data['V'], v.ravel()))

    # Check the set properties
    assert (set(flow.properties) == set(['X', 'Y', 'U', 'V']))

def test_set_input_args():
    X = np.arange(8)
    Y = np.arange(8)
    U = np.sin(X*Y)

    flow = FlowData({'X': X}, {'Y': Y}, {'U': U})
    assert (np.array_equal(flow.data['X'], X))
    assert (np.array_equal(flow.data['Y'], Y))
    assert (np.array_equal(flow.data['U'], U))

    flow = FlowData(('X', X), ('Y', Y), ('U', U))
    assert (np.array_equal(flow.data['X'], X))
    assert (np.array_equal(flow.data['Y'], Y))
    assert (np.array_equal(flow.data['U'], U))

def test_set_as_list():
    X = np.arange(5).tolist()

    flow = FlowData({'f0': X})
    assert (type(flow.get_data('f0')) == np.ndarray)
    assert (np.array_equal(X, flow.data['f0']))
    assert (set(flow.properties) == set(['f0']))

def test_set_empty():
    X = np.arange(0)
    flow = FlowData({'X': X})

def test_set_data_badformats():
    with pytest.raises(TypeError):
        flow = FlowData(['X', [1,2,3]])

def test_oddlists_error():
    X = np.arange(5)
    Y = np.arange(10)

    with pytest.raises(ValueError) as excinfo:
        flow = FlowData({'X': X, 'Y': Y})
    assert ("added array_like objects not all of equal size" in str(excinfo.value))

def test_infer_homogenous_dtype():
    types = ('float32', 'float64', 'int')
    for dtype in types:
        X = np.arange(5, dtype=dtype)
        flow = FlowData({'X': X})

        assert (np.array_equal(flow.data['X'], X))
        assert (flow.data['X'].dtype == dtype)

def test_infer_heterogenous_dtype():
    X = np.arange(5, dtype='int')
    Y = np.arange(5, dtype='float32')

    flow = FlowData({'X': X, 'Y': Y})
    assert (flow.data['X'].dtype == 'int')
    assert (flow.data['Y'].dtype == 'float32')
    assert (np.array_equal(flow.data['X'], X))
    assert (np.array_equal(flow.data['Y'], Y))

def test_set_simple_dtype():
    X = np.arange(5, dtype='int')
    Y = np.arange(5, dtype='float64')
    flow = FlowData({'X': X, 'Y': Y}, dtype='float32')

    assert (flow.data['X'].dtype == 'float32')
    assert (flow.data['Y'].dtype == 'float32')
    assert (np.array_equal(flow.data['X'], X))
    assert (np.array_equal(flow.data['Y'], Y))

def test_set_complex_dtype():
    X = np.arange(5)
    Y = np.arange(5)
    dtype = [('X', 'float32'), ('Y', 'int32')]

    flow = FlowData({'X': X, 'Y': Y}, dtype=dtype)

    assert (flow.data['X'].dtype == 'float32')
    assert (flow.data['Y'].dtype == 'int32')
    assert (np.array_equal(flow.data['X'], X))
    assert (np.array_equal(flow.data['Y'], Y))

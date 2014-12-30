import numpy as np
import pytest
from flow.flow import *

def test_set_data():
    xs = np.arange(9)
    ys = np.arange(5)

    x, y = np.meshgrid(xs, ys, indexing='ij')
    u = np.sin(x)**2
    v = -np.cos(3*y)

    flow = FlowData({'X': x, 'Y': y, 'U': u, 'V': v})

    # Check property getters
    assert (np.equal(flow.X, x.ravel()).all())
    assert (np.equal(flow.Y, y.ravel()).all())
    assert (np.equal(flow.U, u.ravel()).all())
    assert (np.equal(flow.V, v.ravel()).all())

    # Check data structure
    assert (np.equal(flow.data['X'], flow.X).all())
    assert (np.equal(flow.data['Y'], flow.Y).all())
    assert (np.equal(flow.data['U'], flow.U).all())
    assert (np.equal(flow.data['V'], flow.V).all())

    # Check the set properties
    assert (set(flow.properties) == set(['X', 'Y', 'U', 'V']))

def test_set_as_list():
    X = np.arange(5).tolist()

    flow = FlowData({'f0': X})
    assert (type(flow.get_data('f0')) == np.ndarray)
    assert (set(flow.properties) == set(['f0']))

def test_set_empty():
    X = np.arange(0)
    flow = FlowData({'X': X})

def test_oddlists_error():
    X = np.arange(5)
    Y = np.arange(10)

    with pytest.raises(ValueError) as excinfo:
        flow = FlowData({'X': X, 'Y': Y})
    assert ("added array_like objects not all of equal size" in str(excinfo.value))

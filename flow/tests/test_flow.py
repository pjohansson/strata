import numpy as np
from flow.flow import *

def test_init():
    xs = np.linspace(0, 2, 9)
    ys = np.linspace(0, 1, 5)

    x, y = np.meshgrid(xs, ys, indexing='ij')
    flow = FlowData({'X': x, 'Y': y})

    assert (np.equal(flow.X, x.ravel()).all())
    assert (np.equal(flow.Y, y.ravel()).all())
    assert (np.equal(flow.data['X'], flow.X).all())
    assert (np.equal(flow.data['Y'], flow.Y).all())
    assert (set(flow.properties) == set(['X', 'Y']))

def test_set_fields():
    xs = np.linspace(0, 2, 9)
    ys = np.linspace(0, 1, 5)

    x, y = np.meshgrid(xs, ys, indexing='ij')
    u = np.sin(x)**2
    v = -np.cos(3*y)

    flow = FlowData({'X': x, 'Y': y, 'U': u, 'V': v})

    assert (np.equal(flow.U, u.ravel()).all())
    assert (np.equal(flow.V, v.ravel()).all())
    assert (np.equal(flow.data['U'], flow.U).all())
    assert (np.equal(flow.data['V'], flow.V).all())
    assert (set(flow.properties) == set(['X', 'Y', 'U', 'V']))

def test_set_info():
    xs = np.linspace(0, 2, 9)
    ys = np.linspace(0, 1, 5)

    x, y = np.meshgrid(xs, ys, indexing='ij')
    info = {
            'shape': [len(xs), len(ys)],
            'size': {
                'X': [xs[0], xs[-1]],
                'Y': [ys[0], ys[-1]]
                },
            'bin_size': [xs[1]-xs[0], ys[1]-ys[0]],
            'num_bins': len(xs)*len(ys)
            }

    flow = FlowData({'X': x, 'Y': y}, info)

    assert (flow.shape == info['shape'])
    assert (flow.bin_size == info['bin_size'])
    assert (flow.binx == info['bin_size'][0])
    assert (flow.biny == info['bin_size'][1])
    assert (flow.size == info['size'])
    assert (flow.num_bins == info['num_bins'])

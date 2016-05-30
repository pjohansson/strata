import numpy as np
import pytest
from droplets.flow import *

X = np.arange(9)
Y = np.arange(9)
data = {'X': X, 'Y': Y}

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
    flow = FlowData(data, info=good_info)

    assert (np.shape(flow.shape) == (2,))
    assert (flow.shape == tuple(good_info['shape']))

    assert (np.shape(flow.bin_size) == (2,))
    assert (flow.bin_size == tuple(good_info['bin_size']))
    assert (flow.binx == good_info['bin_size'][0])
    assert (flow.biny == good_info['bin_size'][1])

    assert (np.shape(flow.size) == (2,2))
    assert (flow.size[0] == tuple(good_info['size'][0]))
    assert (flow.size[1] == tuple(good_info['size'][1]))

    assert (type(flow.num_bins) == int)
    assert (flow.num_bins == good_info['num_bins'])

def test_no_info():
    flow = FlowData(data)

    assert (flow.shape == (None, None))
    assert (flow.size == ((None, None), (None, None)))
    assert (flow.bin_size == (None, None))
    assert (flow.binx == None and flow.biny == None)
    assert (flow.num_bins == None)

def test_bad_shapes():
    flow = FlowData(data, info=good_info)

    # Verify shape control for shape, size and bin_size
    info = {}
    print("test_bad_shapes\n_______________")
    for key in ('shape', 'bin_size', 'size'):
        for value in (1, [1], [1, 2, 3], [[2], [2,3]]):
            info.update(good_info)
            info[key] = value
            with pytest.raises(TypeError) as excinfo:
                print("Last key, value pair: %r, %r" % (key, value))
                flow.set_info(info)
            assert ("%s must be" % key in str(excinfo.value))

    # For num_bins as a single integer
    info.update(good_info)
    key, value = 'num_bins', [1]
    info[key] = value
    with pytest.raises(TypeError) as excinfo:
        print("Last key, value pair: %r, %r" % (key, value))
        flow.set_info(info)
    assert ("%s must be" % key in str(excinfo.value))

def test_bad_types():
    flow = FlowData(data)

    bad_types = (
            ('shape', ('a', 'b')),
            ('size', (('a', 'b'), ('c', 'd'))),
            ('bin_size', ('a', 'b')),
            ('num_bins', 'a')
            )

    # Assert that ValueError is raised
    for key, value in bad_types:
        bad_info = {key: value}
        with pytest.raises(ValueError):
            flow.set_info(bad_info)

def test_type_conversion():
    mixed_types = {
        'shape': (1.0, 2.5),
        'size': ((3, 5), (2, 1)),
        'bin_size': (1, 2),
        'num_bins': 3.4
        }

    flow = FlowData(data, info=mixed_types)
    assert (type(flow.shape[1] == int) and flow.shape[1] == 2)
    assert (type(flow.size[0][0] == float))
    assert (type(flow.bin_size[0] == float))
    assert (type(flow.num_bins == int and flow.num_bins == 3))

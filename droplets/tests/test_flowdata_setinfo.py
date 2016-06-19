import numpy as np
import pytest

from droplets.flow import FlowData

X = np.arange(9)
Y = np.arange(9)
data = {'X': X, 'Y': Y}

good_info = {
        'spacing': (1., 1.),
        'origin': (0., 0.),
        'shape': (len(X), len(Y)),
        'num_bins': len(X)*len(Y)
        }


def test_set_info():
    flow = FlowData(data, info=good_info)

    assert np.array_equal(good_info['shape'], flow.shape)
    assert np.array_equal(good_info['spacing'], flow.spacing)
    assert np.array_equal(good_info['origin'], flow.origin)

    assert type(flow.num_bins) is int
    assert flow.num_bins == good_info['num_bins']


def test_set_info_None_None():
    info = {
        'spacing': (None, None),
        'origin': (None, None),
        'shape': (None, None),
        'num_bins': None
    }

    flow = FlowData(data, info=info)

    assert np.array_equal(flow.spacing, info['spacing'])
    assert np.array_equal(flow.origin, info['origin'])
    assert np.array_equal(flow.shape, info['shape'])
    assert flow.num_bins == None


def test_get_info_dict():
    flow = FlowData(data)

    assert flow._info['spacing'] == flow.spacing
    assert flow._info['shape'] == flow.shape
    assert flow._info['origin'] == flow.origin
    assert flow._info['num_bins'] == flow.num_bins


def test_no_info():
    flow = FlowData(data)

    assert flow.shape == (None, None)
    assert flow.origin == (None, None)
    assert flow.spacing == (None, None)
    assert flow.num_bins == None


def test_bad_shapes():
    flow = FlowData(data, info=good_info)

    # Verify shape control for shape, size and bin_size
    info = {}
    print("test_bad_shapes\n_______________")
    for key in ('shape', 'origin', 'spacing'):
        for value in (1, [1], [[2], [2,3]]):
            info.update(good_info)
            info[key] = value
            with pytest.raises(ValueError) as excinfo:
                print("Last key, value pair: %r, %r" % (key, value))
                flow.set_info(info)
            assert ("%s must be" % key in str(excinfo.value))

    # For num_bins as a single integer
    info.update(good_info)
    key, value = 'num_bins', [1]
    info[key] = value
    with pytest.raises(ValueError) as excinfo:
        print("Last key, value pair: %r, %r" % (key, value))
        flow.set_info(info)
    assert ("%s must be" % key in str(excinfo.value))


def test_bad_types():
    flow = FlowData(data)

    bad_types = (
            ('shape', ('a', 'b')),
            ('origin', ('a', 'b')),
            ('spacing', ('a', 'b')),
            ('num_bins', 'a')
            )

    # Assert that ValueError is raised
    for key, value in bad_types:
        bad_info = {key: value}
        with pytest.raises(ValueError):
            flow.set_info(bad_info)


def test_bad_info_labels():
    info = {'bad_label': None}

    with pytest.raises(KeyError) as exc:
        FlowData(data, info=info)
        assert "Unknown key for system information: 'bad_label'" in exc


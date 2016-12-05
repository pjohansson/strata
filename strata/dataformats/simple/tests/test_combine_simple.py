import numpy as np
import sys

from strata.dataformats.simple.average import combine_bins
from strata.dataformats.simple.read import read_data

labels_and_weights = ('N', None), ('M', None), ('T', 'N'), ('U', 'M'), ('V', 'M')

def test_combine_bins():
    x = np.arange(2)
    y = 2*x
    xs, ys = np.meshgrid(x, y)

    data = { 'X': xs, 'Y': ys }

    for l in ['M', 'N', 'T', 'U', 'V']:
        data[l] = np.random.sample(xs.shape)

    info = {
        'origin': [0., 0.],
        'spacing': [1.0, 2.0],
        'shape': [2, 2],
        'num_bins': 4
    }

    new_data, new_info = combine_bins(data, info, 2, 2)

    assert(np.array_equal([0.5], new_data['X']))
    assert(np.array_equal([1.0], new_data['Y']))

    # Assert that weighted averages are taken for U, V with respect
    # to the mass M and T with the respect of N.
    for label, weight in labels_and_weights:
        if weight == None:
            assert(np.allclose([data[label].mean()], new_data[label]))
        else:
            assert(np.allclose([np.average(data[label], weights=data[weight])], new_data[label]))

    assert(np.array_equal([0.5, 1.0], new_info['origin']))
    assert(np.array_equal([2., 4.], new_info['spacing']))
    assert(np.array_equal([1, 1], new_info['shape']))
    assert(1 == new_info['num_bins'])


def test_read_and_combine_bins():
    # This file has data of shape (3, 4), reduce to (1, 2) and verify it
    filename = 'strata/dataformats/simple/tests/data_plainsimple.dat'
    data, info = read_data(filename)

    combined_data, combined_info = combine_bins(data, info, 3, 2)

    # Coordinate checks
    assert(np.array_equal([133.375, 133.375], combined_data['X']))
    assert(np.array_equal([0.5, 1.0], combined_data['Y']))

    # More advanced checks
    for label, weight in labels_and_weights:
        ds = data[label].reshape(info['shape'])
        if weight == None:
            assert(np.isclose(ds[0:3, 0:2].mean(), combined_data[label][0]))
            assert(np.isclose(ds[0:3, 2:].mean(), combined_data[label][1]))
        else:
            ws = data[weight].reshape(info['shape'])
            assert(np.isclose(np.average(ds[0:3, 0:2], weights=ws[0:3, 0:2]), combined_data[label][0]))
            assert(np.isclose(np.average(ds[0:3, 2:], weights=ws[0:3, 2:]), combined_data[label][1]))

import numpy as np
import pytest

from droplets.average import *

size = 101
x = np.linspace(0, 10, size)
y = np.linspace(2, 7, size)
c = lambda x, y: 7*y*np.exp(-x/10)
d = lambda c: 1/c

def gen_data(size):
    data = np.zeros(size, dtype=[(l, 'float') for l in ('X', 'Y', 'C', 'M')])

    data['X'] = np.random.choice(x, size, replace=False)
    data['Y'] = np.random.choice(y, size, replace=False)
    data['C'] = c(data['X'], data['Y'])
    data['M'] = d(data['C'])

    return data

def get_minmax(data):
    x0 = np.min([np.min(d['X']) for d in data])
    x1 = np.max([np.max(d['X']) for d in data])
    y0 = np.min([np.min(d['Y']) for d in data])
    y1 = np.max([np.max(d['Y']) for d in data])

    return x0, x1, y0, y1

def test_find_common_grid():
    spacing = np.array([v[1] - v[0] for v in (x, y)])
    num_maps = 100
    num_samples = 99

    data = [gen_data(num_samples) for i in range(num_maps)]
    x0, x1, y0, y1 = get_minmax(data)

    combined_grid = get_combined_grid(data, spacing=spacing)

    # Assert dtype and null default
    assert (combined_grid.dtype == data[0].dtype)
    assert (np.array_equiv(combined_grid['C'], 0))
    assert (np.array_equiv(combined_grid['M'], 0))

    # Assert that all elements are present
    for ds in data:
        for d in ds:
            cx, cy = np.array([d[l] for l in ('X', 'Y')])
            ind = np.isclose(ds['X'], cx) & np.isclose(ds['Y'], cy)
            assert len(ds[ind]) == 1

def test_find_common_grid_nodata():
    with pytest.raises(ValueError):
        get_combined_grid([], spacing=(0.1, 0.1))

def test_fill_data():
    for i in range(100):
        spacing = np.array([v[1] - v[0] for v in (x, y)])
        num_samples = 10

        data = gen_data(num_samples)
        combined_grid = get_combined_grid([data], spacing=spacing)
        filled_data = transfer_data(combined_grid, data)

        for d in data:
            assert (d in filled_data)

def test_fill_data_baddata():
    spacing = np.array([v[1] - v[0] for v in (x, y)])
    num_samples = 10

    data = gen_data(num_samples)
    data[0] = data[1].copy()
    combined_grid = get_combined_grid([data], spacing=spacing)

    with pytest.raises(ValueError):
        filled_data = transfer_data(combined_grid, data)

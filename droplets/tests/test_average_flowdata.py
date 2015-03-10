import numpy as np
import pytest

from droplets.average import *
from droplets.flow import FlowData

size = 101
x = np.linspace(0, 10, size)
y = np.linspace(2, 7, size)
c = lambda x, y: 7*y*np.exp(-x/10)
d = lambda c: 1/c

def gen_random_data(size):
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
    bin_size = np.array([v[1] - v[0] for v in (x, y)])
    num_maps = 100
    num_samples = 99

    data = [gen_random_data(num_samples) for i in range(num_maps)]
    x0, x1, y0, y1 = get_minmax(data)

    combined_grid = get_combined_grid(data, bin_size=bin_size)

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
        get_combined_grid([], bin_size=(0.1, 0.1))

def test_fill_data():
    for i in range(100):
        bin_size = np.array([v[1] - v[0] for v in (x, y)])
        num_samples = 10

        data = gen_random_data(num_samples)
        combined_grid = get_combined_grid([data], bin_size=bin_size)
        filled_data = transfer_data(combined_grid, data)

        for d in data:
            assert (d in filled_data)

def test_fill_data_baddata():
    bin_size = np.array([v[1] - v[0] for v in (x, y)])
    num_samples = 10

    data = gen_random_data(num_samples)
    data[0] = data[1].copy()
    combined_grid = get_combined_grid([data], bin_size=bin_size)

    with pytest.raises(ValueError):
        filled_data = transfer_data(combined_grid, data)

def test_average_data():
    dtype = [(l, 'float') for l in ('X', 'Y', 'C', 'M')]
    d1 = np.zeros(2, dtype=dtype)
    d1['X'] = np.array([0, 1])
    d1['Y'] = np.array([0, 1])
    d1['C'] += 1
    d1['M'] += 1

    d2 = d1.copy()
    d2['C'] += 2
    d2['M'] += 4

    control = d1.copy()
    control['C'] = (d1['C'] + d2['C'])/2
    control['M'] = (d1['M'] + d2['M'])/2

    avg_data = average_data([d1, d2])
    assert (np.array_equal(avg_data, control))

def test_average_onedata():
    dtype = [(l, 'float') for l in ('X', 'Y', 'C', 'M')]
    d1 = np.zeros(2, dtype=dtype)
    d1['X'] = np.array([0, 1])
    d1['Y'] = np.array([0, 1])
    d1['C'] += 1
    d1['M'] += 1

    avg_data = average_data([d1])
    assert (np.array_equal(avg_data, d1))

def test_average_badgrids():
    dtype = [(l, 'float') for l in ('X', 'Y', 'C', 'M')]
    d1 = np.zeros(2, dtype=dtype)
    d1['X'] = np.array([0, 1])
    d1['Y'] = np.array([0, 1])

    for l in ('X', 'Y'):
        d2 = d1.copy()
        d2[l] += 0.2

        with pytest.raises(ValueError):
            average_data([d1, d2])

def test_average_nodata():
    assert np.array_equal(average_data([]), np.array([]))

def test_average_weights():
    dtype = [(l, 'float') for l in ('X', 'Y', 'U', 'M')]

    d1 = np.zeros(4, dtype=dtype)
    d1['X'] = np.array([0, 1, 2, 3])
    d1['Y'] = np.array([0, 1, 2, 3])
    d1['M'] += 1
    d1['U'] += 1

    d2 = d1.copy()
    d2['M'] += 4
    d2['U'] += 2

    control = d1.copy()
    control['M'] = (d1['M'] + d2['M'])/2
    control['U'] = (d1['M']*d1['U'] + d2['M']*d2['U'])/(2*control['M'])

    avg_data = average_data([d1, d2], weights=[('U', 'M')])
    assert (np.array_equal(avg_data, control))

def test_average_weights_badlabels():
    dtype = [(l, 'float') for l in ('X', 'Y', 'U', 'M')]
    d1 = np.zeros(2, dtype=dtype)

    with pytest.raises(KeyError):
        average_data([d1], weights=[('f0', 'M')])

    with pytest.raises(KeyError):
        average_data([d1], weights=[('U', 'f0')])

def test_average_weights_zeroes():
    dtype = [(l, 'float') for l in ('X', 'Y', 'U', 'M')]

    d1 = np.zeros(2, dtype=dtype)
    d1['X'] = np.array([0, 1])
    d1['Y'] = np.array([0, 1])
    d1['M'] += 1
    d1['U'] += 1

    d2 = d1.copy()
    d2['M'] += 4
    d2['U'] += 2

    d1['M'][1] = 0
    d2['M'][1] = 0

    avg_data = average_data([d1], weights=[('U', 'M')])
    assert (avg_data['U'][1] == 0.)

def test_average_flowdata():
    dtype = [(l, 'float') for l in ('X', 'Y', 'U', 'M')]
    gen_data = lambda data: [(l, data[l]) for l, _ in dtype]

    d1 = np.zeros(4, dtype=dtype)
    d1['X'] = np.array([0, 1, 2, 3])
    d1['Y'] = np.array([0, 0, 0, 0])
    d1['M'] += 1
    d1['U'] += 1

    d2 = d1.copy()
    d2['M'] += 4
    d2['U'] += 2

    control = d1.copy()
    control['M'] = (d1['M'] + d2['M'])/2
    control['U'] = (d1['U'] + d2['U'])/2

    flow1, flow2 = [FlowData(*gen_data(d), info={'bin_size': (1,1)})
            for d in (d1, d2)]
    avg_flow = average_flow_data([flow1, flow2])

    assert (avg_flow.shape == (4, 1))
    assert (avg_flow.bin_size == (1, 1))
    assert (avg_flow.size == ((0,3),(0,0)))
    assert (avg_flow.num_bins == len(d1))

    assert (np.array_equal(avg_flow.data, control))

def test_average_flowdata_otherbinsize():
    dtype = [(l, 'float') for l in ('X', 'Y', 'U', 'M')]
    gen_data = lambda data: [(l, data[l]) for l, _ in dtype]

    d1 = np.zeros(4, dtype=dtype)
    d1['X'] = np.array([0, 1, 2, 3])
    d1['Y'] = np.array([0, 0, 0, 0])
    d1['M'] += 1
    d1['U'] += 1

    d2 = d1.copy()
    d2['X'] *= 2
    d2['M'] += 4
    d2['U'] += 2
    flow1, flow2 = [FlowData(*gen_data(d), info={'bin_size': (1,1)})
            for d in (d1, d2)]
    flow2.bin_size = (2, 1)

    with pytest.raises(ValueError):
        average_flow_data([flow1, flow2])

def test_average_flowdata_nobinsize():
    dtype = [(l, 'float') for l in ('X', 'Y', 'U', 'M')]
    gen_data = lambda data: [(l, data[l]) for l, _ in dtype]

    data = np.zeros(4, dtype=dtype)
    data['X'] = np.array([0, 1, 2, 3])
    data['Y'] = np.array([0, 0, 0, 0])
    data['M'] += 1
    data['U'] += 1

    flow1 = FlowData(*gen_data(data))

    with pytest.raises(ValueError) as err:
        average_flow_data([flow1])

def test_average_flowdata_noinput():
    with pytest.raises(ValueError) as err:
        average_flow_data([])

def test_average_flowdata_badweights():
    dtype = [(l, 'float') for l in ('X', 'Y', 'U', 'M')]
    gen_data = lambda data: [(l, data[l]) for l, _ in dtype]

    data = np.zeros(4, dtype=dtype)
    data['X'] = np.array([0, 1, 2, 3])
    data['Y'] = np.array([0, 0, 0, 0])

    flow = FlowData(*gen_data(data), info={'bin_size': (1,1)})

    with pytest.raises(KeyError) as err:
        average_flow_data([flow, flow], weights=[('f0', 'M')])

    with pytest.raises(KeyError) as err:
        average_flow_data([flow, flow], weights=[('U', 'f0')])

def test_average_flowdata_othercoordlabels():
    dtype = [(l, 'float') for l in ('c0', 'c1', 'U', 'M')]
    gen_data = lambda data: [(l, data[l]) for l, _ in dtype]

    data = np.zeros(4, dtype=dtype)
    data['c0'] = np.array([0, 1, 2, 3])
    data['c1'] = np.array([0, 0, 0, 0])
    data['M'] += 1
    data['U'] += 1

    flow = FlowData(*gen_data(data), info={'bin_size': (1,1)})
    average_flow_data([flow, flow], coord_labels=('c0', 'c1'))

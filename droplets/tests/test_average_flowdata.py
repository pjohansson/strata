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

def test_create_combined_grid_yields_error_if_no_input_grid_is_given():
    with pytest.raises(ValueError):
        get_combined_grid([], spacing=(0.1, 0.1))

def test_create_combined_grid_from_given_regular_grids():
    x0 = np.array([0., 1.])
    y0 = np.array([0., 1.])
    xs0, ys0 = np.meshgrid(x0, y0, indexing='xy')

    x1 = np.array([2., 3.])
    y1 = np.array([2., 3.])
    xs1, ys1 = np.meshgrid(x1, y1, indexing='xy')

    data0 = np.zeros((xs0.size, ), dtype=[('X', np.float), ('Y', np.float)])
    data1 = data0.copy()

    data0['X'] = xs0.ravel()
    data0['Y'] = ys0.ravel()

    data1['X'] = xs1.ravel()
    data1['Y'] = ys1.ravel()

    combined_grid = get_combined_grid([data0, data1], spacing=(1., 1.))

    x = np.array([0., 1., 2., 3.])
    y = np.array([0., 1., 2., 3.])
    xs, ys = np.meshgrid(x, y, indexing='xy')

    assert np.array_equal(combined_grid['X'], xs)
    assert np.array_equal(combined_grid['Y'], ys)

def test_common_grid_is_created_from_min_and_max_values_and_spacing():
    # x from -5 to 10
    x0 = np.array([0., 10.])
    x1 = np.array([-5., 5.])

    # y from -10 to 5
    y0 = np.array([-10., 0.])
    y1 = np.array([-5., 5.])

    data0 = np.zeros((2, ), dtype=[('X', np.float), ('Y', np.float)])
    data1 = data0.copy()

    data0['X'] = x0
    data0['Y'] = y0
    data1['X'] = x1
    data1['Y'] = y1

    dx = 0.25
    dy = 0.5

    combined_grid = get_combined_grid([data0, data1], (dx, dy))

    x = np.arange(-5., 10. + dx, dx)
    y = np.arange(-10., 5. + dy, dy)
    xs, ys = np.meshgrid(x, y, indexing='xy')

    assert np.array_equal(combined_grid['X'], xs)
    assert np.array_equal(combined_grid['Y'], ys)

def test_combined_grid_is_returned_with_shape_which_is_y_major_x_minor():
    data = np.zeros((6, ), dtype=[('X', np.float), ('Y', np.float)])

    data['X'] = np.array([0., 1., 0., 1., 0., 1.])
    data['Y'] = np.array([0., 0., 1., 1., 2., 2.])

    combined_grid = get_combined_grid([data], spacing=(1., 1.))

    assert (combined_grid.shape == (3, 2))

def test_combined_grid_is_created_with_same_dtype_as_input():
    data = np.zeros(
        (4, ), dtype=[('X', np.float), ('Y', np.float), ('C', np.float)]
    )

    data['X'] = np.array([0., 1., 0., 1.])
    data['Y'] = np.array([0., 0., 1., 1.])

    combined_grid = get_combined_grid([data], spacing=(1., 1.))

    assert (data.dtype == combined_grid.dtype)
    assert np.array_equal(combined_grid['C'].ravel(), np.zeros((4, )))

def test_fill_data_baddata():
    spacing = np.array([v[1] - v[0] for v in (x, y)])
    num_samples = 10

    data = gen_data(num_samples)
    data[0] = data[1].copy()
    combined_grid = get_combined_grid([data], spacing=spacing)

    with pytest.raises(ValueError):
        filled_data = transfer_data(combined_grid, data)

def test_average_flow_maps():
    x = np.arange(5)
    xs, ys = np.meshgrid(x, x)

    num_maps = 5
    info = {'spacing': [1, 1]}

    f0_sets = [np.random.random(xs.shape) for _ in range(num_maps)]
    f1_sets = [np.random.random(xs.shape) for _ in range(num_maps)]
    flow_maps = [FlowData(('X', xs), ('Y', ys), ('f0', f0_sets[i]), ('f1', f1_sets[i]), info=info)
            for i in range(num_maps)]

    f0_mean = np.average(f0_sets, axis=0)
    f1_mean = np.average(f1_sets, axis=0, weights=f0_sets)

    flow_mean = average_flow_data(flow_maps, weights=[('f1', 'f0')])
    assert np.array_equal(xs.ravel(), flow_mean.data['X'])
    assert np.array_equal(ys.ravel(), flow_mean.data['Y'])
    assert np.isclose(f0_mean.ravel(), flow_mean.data['f0']).all()
    assert np.isclose(f1_mean.ravel(), flow_mean.data['f1']).all()

def test_average_flow_maps_close_coords():
    # Slightly disturb the coordinates of one set and ensure equality through
    # an optional rounding to close values

    x = np.arange(5, dtype=float)
    xs, ys = np.meshgrid(x, x)

    num_maps = 2
    info = {'spacing': [1, 1]}

    f0_sets = [np.random.random(xs.shape) for _ in range(num_maps)]
    f1_sets = [np.random.random(xs.shape) for _ in range(num_maps)]
    flow_maps = [FlowData(('X', xs), ('Y', ys), ('f0', f0_sets[i]), ('f1', f1_sets[i]), info=info)
            for i in range(num_maps)]

    f0_mean = np.average(f0_sets, axis=0)
    f1_mean = np.average(f1_sets, axis=0, weights=f0_sets)

    # Slightly perturb the values
    flow_maps[1].data['X'] += 0.01
    flow_maps[1].data['Y'] += 0.02

    # This should cause an error since the coordinates no longer match
    # when doing the average
    with pytest.raises(ValueError):
        flow_mean = average_flow_data(flow_maps, weights=[('f1', 'f0')])

    # Add a keyword to round coordinates to a decimal
    flow_mean = average_flow_data(flow_maps, weights=[('f1', 'f0')], coord_decimals=1)
    assert np.isclose(xs.ravel(), flow_mean.data['X']).all()
    assert np.isclose(ys.ravel(), flow_mean.data['Y']).all()
    assert np.isclose(f0_mean.ravel(), flow_mean.data['f0']).all()
    assert np.isclose(f1_mean.ravel(), flow_mean.data['f1']).all()

import numpy as np
import pytest
import sys

from droplets.flow import FlowData
from droplets.resample import *


def test_resample_to_one_bin():
    x = [1., 3., 5., 7.]
    y = [0., 1.]

    xs, ys = np.meshgrid(x, y)
    cs = np.random.sample(8)

    flow = FlowData(('X', xs), ('Y', ys), ('C', cs))
    flow.bin_size = (2., 1.)
    flow.shape = (4, 2)

    resampled_flow = downsample_flow_data(flow, (4, 2))

    # Check data
    assert np.array_equal(np.array([4.]),  resampled_flow.data['X'])
    assert np.array_equal(np.array([0.5]), resampled_flow.data['Y'])

    # Use `isclose` to account for possible rounding errors
    assert np.isclose(np.sum(cs), resampled_flow.data['C'])

    # ... and attributes
    assert np.array_equal((1, 1),   resampled_flow.shape)
    assert np.array_equal((8., 2.), resampled_flow.bin_size)


def test_resample_to_multiple_bins():
    x = np.arange(6)
    y = np.arange(4)
    xs, ys = np.meshgrid(x, y)
    print(xs.shape)

    cs = np.random.sample(xs.shape)

    flow = FlowData(('X', xs), ('Y', ys), ('C', cs))
    flow.bin_size = (1., 1.)
    flow.shape = (6, 4)

    resampled_flow = downsample_flow_data(flow, (2, 2))

    # Compare against the data
    # Remember that the data array is ordered as [y, x] (row, col)
    reshaped_data = resampled_flow.data['C'].reshape(2, 3)
    assert np.isclose(np.sum(cs[ :2,  :2]), reshaped_data[0,0])
    assert np.isclose(np.sum(cs[ :2, 2:4]), reshaped_data[0,1])
    assert np.isclose(np.sum(cs[ :2, 4:]), reshaped_data[0,2])
    assert np.isclose(np.sum(cs[2:,   :2]), reshaped_data[1,0])
    assert np.isclose(np.sum(cs[2:,  2:4]), reshaped_data[1,1])
    assert np.isclose(np.sum(cs[2:,  4:]), reshaped_data[1,2])


# If the input coordinates have weird sorting it has to be handled
def test_resample_without_sorted_coords():
    # Input coordinates are in reverse order
    x = [1., 0.]
    y = [1., 0.]

    xs, ys = np.meshgrid(x, y)
    cs = np.random.sample(4)

    flow = FlowData(('X', xs), ('Y', ys), ('C', cs))
    flow.bin_size = (1., 1.)
    flow.shape = (2, 2)

    # Resample 1:1 and ensure that both final results have identical sorting
    resampled_flow = downsample_flow_data(flow, (1, 1))
    resampled_flow.data.sort(order=['Y', 'X'])
    flow.data.sort(order=['Y', 'X'])

    assert np.array_equal(flow.data, resampled_flow.data)


def test_resample_with_weights():
    xs = [0., 1.]
    ys = [0., 0.]

    # cs is averaged by sum(cs*ws)/sum(ws) (the multiplication is broadcast)
    cs = np.random.sample(2)
    ws = np.random.sample(2)

    flow = FlowData(('X', xs), ('Y', ys), ('C', cs), ('W', ws))
    flow.bin_size = (1., 1.)
    flow.shape = (2, 1)

    resampled_flow = downsample_flow_data(flow, (2, 1), weights=[('C', 'W')])

    expected = np.sum(cs[0]*ws[0]+cs[1]*ws[1])/np.sum(ws)
    assert np.isclose(expected, resampled_flow.data['C'])

    # The weight itself is not weighed
    assert np.isclose(np.sum(ws), resampled_flow.data['W'])


def test_resample_with_weights_that_are_zero():
    xs = [0., 1.]
    ys = [0., 0.]

    # cs is averaged by weighing against zero: result should be zero
    cs = np.random.sample(2)
    ws = np.zeros(2)

    flow = FlowData(('X', xs), ('Y', ys), ('C', cs), ('W', ws))
    flow.bin_size = (1., 1.)
    flow.shape = (2, 1)

    resampled_flow = downsample_flow_data(flow, (2, 1), weights=[('C', 'W')])

    assert np.isclose(0., resampled_flow.data['C'])


def test_resample_other_coord_labels():
    xs = [0., 1.]
    ys = [0., 0.]
    cs = np.random.sample(2)

    flow = FlowData(('f0', xs), ('f1', ys), ('C', cs))
    flow.bin_size = [1., 1.]
    flow.shape = (2, 1)

    resampled_flow = downsample_flow_data(flow, (2, 1), coord_labels=('f0', 'f1'))

    assert 0.5 == resampled_flow.data['f0'][0]
    assert 0.0 == resampled_flow.data['f1'][0]
    assert np.isclose(np.sum(cs), resampled_flow.data['C'])


def test_resample_cutting_some_cells():
    xs = [0., 1., 2.]
    ys = [0., 0., 0.]
    cs = np.random.sample(3)

    flow = FlowData(('X', xs), ('Y', ys), ('C', cs))
    flow.bin_size = (1., 1.)
    flow.shape = (3, 1)

    # Combine 2 cells along x, leaving the third cell outside of the average
    resampled_flow = downsample_flow_data(flow, (2, 1))

    assert np.array_equal([0.5], resampled_flow.data['X'])
    assert np.isclose(np.sum(cs[:2]), resampled_flow.data['C'])


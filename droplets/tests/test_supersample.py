import numpy as np
import pytest
import sys

from droplets.flow import FlowData
from droplets.resample import *


def test_supersample_to_same_factor_changes_nothing():
    xs = [0., 1., 0., 1.]
    ys = [0., 0., 1., 1.]
    cs = [0., 1., 2., 3.]

    flow = FlowData(('X', xs), ('Y', ys), ('C', cs))

    for factor in [None, 1]:
        flow_unchanged = supersample_flow_data(flow, factor)
        assert np.array_equal(flow_unchanged.data, flow.data)

def test_supersample_changes_spacing_and_shape():
    xs = [0., 1., 0., 1., 0., 1.]
    ys = [0., 0., 1., 1., 2., 2.]

    flow = FlowData(
        ('X', xs),
        ('Y', ys),
        info={
            'spacing': (1., 1.),
            'shape': (2, 3),
        })

    supersampled = supersample_flow_data(flow, 2)

    assert supersampled.spacing == (0.5, 0.5)
    assert supersampled.shape == (4, 6)

    assert np.array_equal(
        supersampled.data['X'],
        [
            0., 0.5, 1., 1.5,
            0., 0.5, 1., 1.5,
            0., 0.5, 1., 1.5,
            0., 0.5, 1., 1.5,
            0., 0.5, 1., 1.5,
            0., 0.5, 1., 1.5
        ],
        )
    assert np.array_equal(
        supersampled.data['Y'],
        [
            0., 0., 0., 0.,
            0.5, 0.5, 0.5, 0.5,
            1., 1., 1., 1.,
            1.5, 1.5, 1.5, 1.5,
            2., 2., 2., 2.,
            2.5, 2.5, 2.5, 2.5
        ],
        )

def test_supersample_averages_non_coordinate_data():
    xs = [0., 1., 0., 1., 0., 1.]
    ys = [0., 0., 1., 1., 2., 2.]
    cs = [0., 1., 2., 3., 4., 5.]

    flow = FlowData(
        ('X', xs),
        ('Y', ys),
        ('C', cs),
        info={
            'spacing': (1., 1.),
            'shape': (2, 3),
        })

    supersampled = supersample_flow_data(flow, 2)

    assert np.array_equal(
        supersampled.data['C'],
        [
            0./4.,  2./6.,  4./6.,  4./4.,
            4./6.,  9./9., 12./9., 10./6.,
            8./6., 15./9., 18./9., 14./6.,
            16./6., 27./9., 30./9., 22./6.,
            20./6., 33./9., 36./9., 26./6.,
            16./4., 26./6., 28./6., 20./4.
        ]
    )

def test_supersample_uses_input_coordinate_labels():
    xs = [0., 1., 0., 1.]
    ys = [0., 0., 1., 1.]
    cs = [0., 1., 2., 3.]

    flow = FlowData(
        ('X0', xs),
        ('X1', ys),
        ('C', cs),
        info={
            'spacing': (1., 1.),
            'shape': (2, 2),
        })

    supersampled = supersample_flow_data(flow, 2, coord_labels=('X0', 'X1'))

    assert np.array_equal(
        supersampled.data['X0'],
        [
            0., 0.5, 1., 1.5,
            0., 0.5, 1., 1.5,
            0., 0.5, 1., 1.5,
            0., 0.5, 1., 1.5
        ],
        )

    assert np.array_equal(
        supersampled.data['X1'],
        [
            0., 0., 0., 0.,
            0.5, 0.5, 0.5, 0.5,
            1., 1., 1., 1.,
            1.5, 1.5, 1.5, 1.5
        ],
        )

def test_supersample_with_data_from_weight_labels():
    xs = [0., 1., 0., 1.]
    ys = [0., 0., 1., 1.]
    cs = [1., 1., 1., 1.]
    ws = [2., 0., 0., 0.]

    flow = FlowData(
        ('X', xs),
        ('Y', ys),
        ('C', cs),
        ('W', ws),
        info={
            'spacing': (1., 1.),
            'shape': (2, 2),
        })

    supersampled = supersample_flow_data(flow, 2, weights=[('C', 'W')])

    assert np.array_equal(
        supersampled.data['C'],
        np.array([
            1., 1., 1., 0.,
            1., 1., 1., 0.,
            1., 1., 1., 0.,
            0., 0., 0., 0.
        ])
    )

def test_supersample_with_weights_does_not_affect_coordinate_labels():
    xs = [0., 1., 0., 1.]
    ys = [0., 0., 1., 1.]
    cs = [1., 1., 1., 1.]
    ws = [2., 0., 0., 0.]

    flow = FlowData(
        ('X', xs),
        ('Y', ys),
        ('C', cs),
        ('W', ws),
        info={
            'spacing': (1., 1.),
            'shape': (2, 2),
        })

    supersampled = supersample_flow_data(flow, 2, weights=[('C', 'W')])

    assert np.array_equal(
        supersampled.data['X'],
        [
            0., 0.5, 1., 1.5,
            0., 0.5, 1., 1.5,
            0., 0.5, 1., 1.5,
            0., 0.5, 1., 1.5
        ],
        )

    assert np.array_equal(
        supersampled.data['Y'],
        [
            0., 0., 0., 0.,
            0.5, 0.5, 0.5, 0.5,
            1., 1., 1., 1.,
            1.5, 1.5, 1.5, 1.5
        ],
        )

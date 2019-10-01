import numpy as np
import pytest
import sys

from droplets.flow import FlowData
from droplets.resample import *


def test_bins_are_by_default_sorted_y_minor_x_major():
    xs = [0., 1., 2., 3.]
    ys = [1., 0., 0., 1.]

    flow = FlowData(('X', xs), ('Y', ys))

    flow.sort()

    assert np.array_equal(flow.data['Y'], [0., 0., 1., 1.])
    assert np.array_equal(flow.data['X'], [1., 2., 0., 3.])


def test_other_fields_do_not_influence_sorting():
    xs = [0., 1., 2., 3.]
    ys = [1., 0., 0., 1.]
    vs = [5., 6., 7., 8.]

    flow = FlowData(('X', xs), ('Y', ys), ('V', vs))

    flow.sort()

    assert np.array_equal(flow.data['V'], [6., 7., 5., 8.])


def test_other_coordinate_labels_can_be_used():
    xs = [0., 1., 2., 3.]
    ys = [1., 0., 0., 1.]

    flow = FlowData(('X0', xs), ('X1', ys))

    flow.sort(coord_labels=('X0', 'X1'))

    assert np.array_equal(flow.data['X1'], [0., 0., 1., 1.])
    assert np.array_equal(flow.data['X0'], [1., 2., 0., 3.])

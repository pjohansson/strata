import os
import numpy as np
import pytest
import tempfile as tmp

from droplets.flow import FlowData
from strata.view_flowmap import add_diffusion

def test_gradient_which_is_one_everywhere():
    xs = [0., 1., 0., 1.]
    ys = [0., 0., 1., 1.]
    ms = [0., 1., 1., 2.]
    us = [1., 1., 1., 1.]
    vs = [1., 1., 1., 1.]

    flow = FlowData(
        ('X', xs), ('Y', ys), ('M', ms), ('U', us), ('V', vs),
        info={'shape': (2, 2), 'spacing': (1., 1.)}
        )

    add_diffusion(flow)

    assert np.array_equal(flow.data['diff'], [2., 2., 2., 2.])


def test_calculate_gradient_with_non_unity_bin_spacing():
    xs = [0., 0.5, 0., 0.5]
    ys = [0., 0., 0.25, 0.25]
    ms = [0., 1., 1., 2.]
    us = [1., 1., 1., 1.]
    vs = [1., 1., 1., 1.]

    flow = FlowData(
        ('X', xs), ('Y', ys), ('M', ms), ('U', us), ('V', vs),
        info={'shape': (2, 2), 'spacing': (0.5, 0.25)}
        )

    add_diffusion(flow)

    assert np.array_equal(flow.data['diff'], [6., 6., 6., 6.])


def test_calculate_gradient_which_changes_along_x():
    xs = [0., 1., 2., 0., 1., 2.]
    ys = [0., 0., 0., 1., 1., 1.]
    ms = [0., 1., 0., 1., 0., 1.]
    us = [1., 1., 1., 1., 1., 1.]
    vs = [2., 2., 2., 2., 2., 2.]

    flow = FlowData(
        ('X', xs), ('Y', ys), ('M', ms), ('U', us), ('V', vs),
        info={'shape': (3, 2), 'spacing': (1.0, 1.0)}
        )

    add_diffusion(flow)

    assert np.array_equal(flow.data['diff'], [1., 1., 1., 1., 1., 1.])


def test_calculate_gradient_which_changes_along_y():
    xs = [0., 1., 0., 1., 0., 1.]
    ys = [0., 0., 1., 1., 2., 2.]
    ms = [0., 0., 0., 1., 1., 1.]
    us = [1., 1., 1., 1., 1., 1.]
    vs = [2., 2., 2., 2., 2., 2.]

    flow = FlowData(
        ('X', xs), ('Y', ys), ('M', ms), ('U', us), ('V', vs),
        info={'shape': (2, 3), 'spacing': (1.0, 1.0)}
        )

    add_diffusion(flow)

    assert np.array_equal(flow.data['diff'], [2., 2., 2., 2., 2., 2.])


def test_calculate_gradient_which_varies_in_top_right_corner_only_with_only_u():
    xs = [0., 1., 2., 0., 1., 2.]
    ys = [0., 0., 0., 1., 1., 1.]
    ms = [0., 0., 0., 0., 0., 1.]
    us = [1., 1., 1., 1., 1., 1.]
    vs = [0., 0., 0., 0., 0., 0.]

    flow = FlowData(
        ('X', xs), ('Y', ys), ('M', ms), ('U', us), ('V', vs),
        info={'shape': (3, 2), 'spacing': (1.0, 1.0)}
        )

    add_diffusion(flow)

    assert np.array_equal(flow.data['diff'], [0., 0., 0., 0., 0.5, 1.])


def test_calculate_gradient_which_varies_in_top_right_corner_only_with_only_v():
    xs = [0., 1., 2., 0., 1., 2.]
    ys = [0., 0., 0., 1., 1., 1.]
    ms = [0., 0., 0., 0., 0., 1.]
    us = [0., 0., 0., 0., 0., 0.]
    vs = [2., 2., 2., 2., 2., 2.]

    flow = FlowData(
        ('X', xs), ('Y', ys), ('M', ms), ('U', us), ('V', vs),
        info={'shape': (3, 2), 'spacing': (1.0, 1.0)}
        )

    add_diffusion(flow)

    assert np.array_equal(flow.data['diff'], [0., 0., 2., 0., 0., 2.])

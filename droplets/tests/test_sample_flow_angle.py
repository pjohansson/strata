import numpy as np
import pytest

from droplets.flow import FlowData
from droplets.sample import sample_flow_angle


def test_calc_flow_angles():
    us = np.array([+1, +1, +0, -1, -1, +1, +0, -1])
    vs = np.array([+0, +1, +1, +1, +0, -1, -1, -1])
    expected_angles = np.array([0., 45., 90., 135., 180., -45., -90., -135.])

    flow = FlowData(('U', us), ('V', vs))
    assert np.isclose(expected_angles, sample_flow_angle(flow)).all()

def test_calc_flow_angles_other_labels():
    us = np.array([+1, -1, +1, -1])
    vs = np.array([+1, +1, -1, -1])
    expected_angles = np.array([45., 135., -45., -135.])

    flow = FlowData(('f0', us), ('f1', vs))

    assert np.isclose(expected_angles, sample_flow_angle(flow, flow_labels=['f0', 'f1'])).all()

    # Verify that an error is properly raised if the labels were not found
    with pytest.raises(ValueError):
        assert np.isclose(expected_angles, sample_flow_angle(flow)).all()

def test_calc_flow_angle_mean():
    # Verify that the wrap-around is handled correctly when taking
    # the mean of angles. Two angles at -1,+1 degrees should have
    # a mean of 0 degrees.
    us = np.array([+1, +1])
    vs = np.array([-0.1, +0.1])
    expected_mean_angle = 0.

    flow = FlowData(('U', us), ('V', vs))

    assert np.isclose(expected_mean_angle, sample_flow_angle(flow, mean=True))

def test_calc_flow_angles_bad_number_of_labels():
    us = np.array([+1, -1, +1, -1])
    vs = np.array([+1, +1, -1, -1])

    flow = FlowData(('U', us), ('V', vs))

    with pytest.raises(ValueError):
        sample_flow_angle(flow, flow_labels=['f0', 'f1', 'f2'])


def test_calc_weighted_flow_angle_mean():
    # If we want to weigh the mean angle by their mass (ie. calculcate using
    # the momentum) this should be possible.
    angles = np.array([-20., -10., +20.])
    weights = np.array([1, 0, 1])
    us = np.cos(np.radians(angles))
    vs = np.sin(np.radians(angles))

    # With the middle value having no weight the mean should be 0 degrees
    expected_mean_angle = 0

    flow = FlowData(('U', us), ('V', vs), ('M', weights))

    assert np.isclose(expected_mean_angle, sample_flow_angle(flow, mean=True, weight='M'))

def test_calc_weighted_flow_angle_bad_label():
    angles = np.array([-20., -10., +20.])
    us = np.cos(np.radians(angles))
    vs = np.sin(np.radians(angles))

    flow = FlowData(('U', us), ('V', vs))

    with pytest.raises(ValueError):
        sample_flow_angle(flow, mean=True, weight='M')

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

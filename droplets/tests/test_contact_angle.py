import numpy as np
import pytest

from droplets.interface import get_contact_angle
from droplets.flow import FlowData

@pytest.fixture(params=[45, 90, 135])
def some_contact_angles(request):
    X = np.linspace(0, 10, 101)
    Y = np.linspace(0, 3, 7)

    xs, ys = np.meshgrid(X, Y)
    cs = np.zeros(xs.shape)

    target_angle = request.param

    # Create structure with angle
    ymin = Y[0]
    height = Y[2]
    xedge = 2

    # Bottom layer
    iedge = min(range(len(X)), key=lambda i: np.abs(X[i]-xedge))
    cs[0][iedge:-iedge] = 1

    # Fill in other layers
    for i, xlayer in enumerate(xs[1:]):
        dy = Y[i+1] - ymin
        dxs = xlayer - xedge

        angles = np.degrees(np.arctan2(dy, dxs))
        iangle = min(range(len(angles)),
                key=lambda j: np.abs(angles[j] - target_angle))

        if dy == height:
            contact_angle = angles[iangle]

        cs[i+1][iangle:-iangle] += 1

    data = {'X': xs, 'Y': ys, 'C': cs}
    label = 'C'

    return data, contact_angle, label, height

def test_contact_angles(some_contact_angles):
    data, contact_angle, label, height = some_contact_angles
    angles = get_contact_angle(FlowData(data), height, label,
        radius=1., num_bins=0, cutoff=None)
    assert (np.allclose(angles.mean(), contact_angle))

def test_floor(some_contact_angles):
    data, contact_angle, label, height = some_contact_angles

    # Set another floor and mess up the bottom layer
    data[label][0] = 1
    floor = data['Y'][1][0]
    assert (floor != height)

    angles = get_contact_angle(FlowData(data), height, label,
        radius=1., floor=floor, num_bins=0)
    assert (np.allclose(angles.mean(), contact_angle))

def test_nofloor(some_contact_angles):
    data, contact_angle, label, height = some_contact_angles

    # Remove bottom layer and use as floor
    data[label][0] = 0
    floor = data['Y'][0][0]

    angles = get_contact_angle(FlowData(data), height, label,
        radius=1., floor=floor, num_bins=0)
    assert (angles == (None, None))

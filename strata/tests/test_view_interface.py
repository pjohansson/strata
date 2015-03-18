import numpy as np

from droplets.flow import FlowData
from droplets.interface import get_interface
from strata.interface import get_interface_coordinates

coords = np.linspace(0, 10, 11)
xs, ys = np.meshgrid(coords, coords)
cs = np.zeros(xs.shape)
cs[:,1:10] = 1.

data = {'X': xs, 'Y': ys, 'C': cs}
flow = FlowData(data)

def test_get_interface_coords():
    x, y = get_interface_coordinates(flow, 'C')

    assert np.array_equiv(x[:len(coords)], 1.)
    assert np.array_equiv(x[len(coords):], 9.)
    assert np.array_equal(y[:len(coords)], coords)
    assert np.array_equal(y[len(coords):], coords[::-1])

def test_adjust_interface_to_com():
    com = np.average(flow.data['X'], weights=flow.data['C'])
    x, y = get_interface_coordinates(flow, 'C', adjust_com=True)

    assert np.isclose(com, 5.)
    assert np.array_equiv(x[:len(coords)], 1.-com)
    assert np.array_equiv(x[len(coords):], 9.-com)
    assert np.array_equal(y[:len(coords)], coords)
    assert np.array_equal(y[len(coords):], coords[::-1])

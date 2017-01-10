import numpy as np

from strata.interface.sample import calc_length, get_area_of_interface

# Test against a circle of radius 1
# The coordinates move clockwise from the bottom angle of -90 (+270) degrees
#angles = np.radians(np.arange(270, -91, -1))
angles = np.radians(np.linspace(270, -90, 3600))
xs = np.cos(angles)
ys = np.sin(angles)

def test_calc_interface_length():
    # Circumference should be close to 2*pi
    assert np.isclose(2*np.pi, calc_length(xs, ys))

def test_calc_interface_area():
    # Area should be close to pi
    assert np.isclose(np.pi, get_area_of_interface(xs, ys))

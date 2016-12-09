import numpy as np
import pytest

from droplets.flow import FlowData
from droplets.sample import sample_center_of_mass


def test_calc_center_of_mass():
    x = np.arange(3)
    y = 0.5*x
    xs, ys = np.meshgrid(x, y)

    # Center of mass should be at position (1.25, 0.5)
    # which for x and y gives (one possible set of) calculations:
    #   x: (0*1 + 1*1 + 2*2) / (1 + 1 + 2) = 5/4 = 1.25
    #   y: (0*1 + 0.5*2 + 1*1) / (1 + 1 + 2) = 2/4 = 0.5
    ms = np.array([
        [0, 1, 0],
        [0, 0, 2],
        [1, 0, 0]
    ])

    flow = FlowData(('X', xs), ('Y', ys), ('M', ms))
    assert np.allclose([1.25, 0.5], sample_center_of_mass(flow))

    # For non-default labels
    flow = FlowData(('c0', xs), ('c1', ys), ('f0', ms))
    assert np.allclose(
        [1.25, 0.5],
        sample_center_of_mass(flow, coord_labels=['c0', 'c1'], mass_label='f0')
    )


def test_calc_center_of_mass_divide_by_zero():
    x = np.arange(3)
    y = 0.5*x
    xs, ys = np.meshgrid(x, y)

    # If the mass sums to zero an error should be raised
    ms = np.zeros(xs.shape)
    flow = FlowData(('X', xs), ('Y', ys), ('M', ms))

    with pytest.raises(ZeroDivisionError) as exc:
        sample_center_of_mass(flow)

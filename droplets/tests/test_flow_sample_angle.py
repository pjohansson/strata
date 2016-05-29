import numpy as np

from droplets.flow import FlowData
from droplets.sample import sample_per_angle_from

# Measure the quantity from (0, 0)
origin = (0., 0.)

def test_measure_per_angle_single_cell():
    # Cell is at 45 deg. angle from (0, 0)
    x, y = [1.], [1.]
    c = [1.]

    flow = FlowData(('X', x), ('Y', y), ('C', c))
    result = sample_per_angle_from(flow, origin, 'C')

    # One index per degree
    assert np.array_equal(np.arange(360), result['angle'])

    # Check that the full cell appears only in the correct index
    assert np.array_equal([45], np.nonzero(result['C'])[0])
    assert c[0] == result['C'][45]


def test_measure_two_cells_same_angle_is_mean():
    # Two cells at 45 deg. angle
    xs = [1., 2.]
    ys = [1., 2.]
    cs = np.random.sample(2)

    flow = FlowData(('X', xs), ('Y', ys), ('C', cs))
    result = sample_per_angle_from(flow, origin, 'C')

    # Result should be average of the two cells
    assert np.isclose(np.mean(cs), result['C'][45])


def test_measure_per_angle_cutoff_radius():
    # Two cells at radii sqrt(2), 2*sqrt(2)
    # Only the first cell is within radius 2
    radius = 2.
    xs = [1., 2.]
    ys = [1., 2.]
    cs = np.random.sample(2)

    flow = FlowData(('X', xs), ('Y', ys), ('C', cs))
    result = sample_per_angle_from(flow, origin, 'C', rmax=radius)

    assert cs[0] == result['C'][45]


def test_measure_per_angle_in_radius_slice():
    # Two cells at radii sqrt(2), 2*sqrt(2)
    # Only the second cell is outside of radius 2
    rmin = 2.
    xs = [1., 2.]
    ys = [1., 2.]
    cs = np.random.sample(2)

    flow = FlowData(('X', xs), ('Y', ys), ('C', cs))
    result = sample_per_angle_from(flow, origin, 'C', rmin=rmin)

    assert cs[1] == result['C'][45]


def test_measure_cell_angle_negative_xy():
    # These cells have angles 135, 225, 270, 315 degrees
    xs = [-1., -1.,  0.,  1.]
    ys = [ 1., -1., -1., -1.]
    cs = [1., 1., 1., 1.]

    flow = FlowData(('X', xs), ('Y', ys), ('C', cs))
    result = sample_per_angle_from(flow, origin, 'C')

    assert np.array_equal([135, 225, 270, 315], np.nonzero(result['C'])[0])


def test_measure_cell_angles_other_coord_labels():
    x, y = [1.], [1.]
    c = [5.]

    # Use other coordinate labels
    flow = FlowData(('f0', x), ('f1', y), ('C', c))
    result = sample_per_angle_from(flow, origin, 'C', coord_labels=('f0', 'f1'))

    assert np.array_equal([45], np.nonzero(result['C'])[0])
    assert c[0] == result['C'][45]


def test_measure_cell_angles_within_angles():
    # Angle limits
    amin, amax = 90., 135.

    # These cells have angles 45, 90, 135, 180 degrees
    xs = [1., 0., -1., -1.]
    ys = [1., 1.,  1.,  0.]
    cs = np.random.sample(4)

    flow = FlowData(('X', xs), ('Y', ys), ('C', cs))
    result = sample_per_angle_from(flow, origin, 'C', amin=amin, amax=amax)

    assert np.array_equal(np.arange(90., 135+1., 1.), result['angle'])
    assert cs[1] == result['C'][0]
    assert cs[2] == result['C'][45]
    assert cs[0] not in result['C']
    assert cs[3] not in result['C']


def test_measure_cell_angles_minmax_angle_lims():
    amin, amax = -10., 400

    xs = [1.]
    ys = [1.]
    cs = np.random.sample(1)

    flow = FlowData(('X', xs), ('Y', ys), ('C', cs))
    result = sample_per_angle_from(flow, origin, 'C', amin=amin, amax=amax)

    assert np.array_equal(np.arange(0, 360, 1), result['angle'])

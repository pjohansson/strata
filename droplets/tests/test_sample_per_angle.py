import numpy as np

from droplets.flow import FlowData
from droplets.sample import sample_per_angle_from

# Measure the quantities from (0, 0)
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


def test_sample_per_angle_correct_angle_bins():
    amin = 45.

    # A bin at angle 44.9 should be excluded,
    # one at 45.9 should be in the first bin
    angles = np.radians([44.9, 45.9])
    ys = [1., 1.]
    xs = ys/np.tan(angles)
    cs = [1., 2.]

    flow = FlowData(('X', xs), ('Y', ys), ('C', cs))
    result = sample_per_angle_from(flow, origin, 'C', amin=amin)

    assert cs[1] == result['C'][0]


def test_sample_per_angle_angle_bin_size():
    # Make bins larger for the output
    bin_size = 90.

    # First two bins should be in the first output angle
    angles = np.radians([1e-9, 45., 90.])
    ys = [1., 1., 1.]
    xs = ys/np.tan(angles)
    cs = [1., 2., 3.]

    flow = FlowData(('X', xs), ('Y', ys), ('C', cs))
    result = sample_per_angle_from(flow, origin, 'C', size=bin_size)

    assert np.array_equal([0., 90., 180., 270.], result['angle'])
    assert np.isclose(np.mean(cs[:2]), result['C'][0])
    assert cs[2] == result['C'][1]


def test_sample_per_angle_weighted_average():
    # Two cells at 45 deg. angle
    xs = [1., 2.]
    ys = [1., 2.]
    cs = np.random.sample(2)
    ws = np.random.sample(2)

    flow = FlowData(('X', xs), ('Y', ys), ('C', cs), ('W', ws))

    # Weigh the C label by W
    result = sample_per_angle_from(flow, origin, 'C', weight='W')

    # Result should be weighted average of the two cells
    assert np.isclose(np.average(cs, weights=ws), result['C'][45])


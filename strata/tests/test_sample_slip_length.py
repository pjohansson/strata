import numpy as np

from droplets.flow import FlowData
from strata.sample_average import sample_slip_length

def test_slip_length_calculation():
    # Create a gradient with slope 0.5 in y, which is 1.0 at y = 0.0
    # and thus will be 0.0 at y = -2.0. The slip length is the distance
    # between the value being 0.0 and the floor, in this case 2.0.
    n = 10
    ys = np.arange(n)
    us = np.arange(n) * 0.5 + 1.0

    info = {'shape': (1, n)} # Has to be set
    flow = FlowData(('Y', ys), ('U', us), info=info)

    result, _ = sample_slip_length(flow)
    assert np.isclose(2.0, result)

def test_slip_length_is_column_average():
    y = np.arange(2.0)
    u = np.arange(2.0) # Slope is 1.0

    # Indexing is ij for x, y to ensure that the column order is correct:
    # We want us to change with y, not with x!
    _, ys = np.meshgrid(y, y, indexing='ij')
    us, _ = np.meshgrid(u, u, indexing='xy')

    us[0,:] += 1.0 # Will be 0.0 at y = -1.0 -> slip length 1.0
    us[1,:] += 2.0 # Will be 0.0 at y = -2.0 -> slip length 2.0

    info = {'shape': ys.shape} # Has to be set

    flow = FlowData(('Y', ys), ('U', us), info=info)

    # The average slip length should be 1.5
    result, _ = sample_slip_length(flow)
    assert np.isclose(1.5, result)

def test_slip_length_std_calculation():
    y = np.arange(3.0)
    u = np.arange(3.0) # Slope is 1.0

    # Indexing is ij for x, y to ensure that the column order is correct:
    # We want us to change with y, not with x!
    _, ys = np.meshgrid(y, y, indexing='ij')
    us, _ = np.meshgrid(u, u, indexing='xy')

    us[0,:] += 1.0 # Will be 0.0 at y = -1.0 -> 1.0
    us[1,:] += 2.0 # Will be 0.0 at y = -2.0 -> 2.0
    us[2,:] += 3.0 # Will be 0.0 at y = -3.0 -> 3.0

    info = {'shape': ys.shape} # Has to be set

    flow = FlowData(('Y', ys), ('U', us), info=info)

    # The average slip length is 2.0
    # Std: sqrt((1.0**2 + 0.0**2 + 1.0**2) / 3.0) = sqrt(2/3)
    _, std = sample_slip_length(flow)
    assert np.isclose(np.sqrt(2/3), std)

def test_slip_length_with_input_floor():
    # Measure the slip length to a floor different to the default 0.0.
    n = 5
    ys = np.arange(n)
    us = np.arange(n) + 1.0 # 1.0 at y = 0.0

    info = {'shape': (1, n)} # Has to be set

    flow = FlowData(('Y', ys), ('U', us), info=info)

    # The slip length should be 1.0 if we measure it at y = 0.0
    result, _ = sample_slip_length(flow)
    assert np.isclose(1.0, result)

    # But if we measure it at y = 1.0 the slip length should be 2.0
    result, _ = sample_slip_length(flow, floor=1.0)
    assert np.isclose(2.0, result)

    # And if we measure it at y = 2.0 the slip length should be 3.0
    result, _ = sample_slip_length(flow, floor=2.0)
    assert np.isclose(3.0, result)

def test_slip_length_other_labels():
    n = 10
    ys = np.arange(n)
    us = np.arange(n) * 0.5 + 1.0

    info = {'shape': (1, n)} # Has to be set
    flow = FlowData(('f0', ys), ('f1', us), info=info)

    result, _ = sample_slip_length(
                                flow,
                                coord_labels=(None, 'f0'),
                                flow_labels=('f1', None)
    )
    assert np.isclose(2.0, result)

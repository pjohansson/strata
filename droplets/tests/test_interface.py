import numpy as np
import pytest

import droplets.interface as intf
from droplets.flow import FlowData
from droplets.interface import get_interface

# Create grid
x = np.arange(11)
y = np.arange(11)
xs, ys = np.meshgrid(x, y)
info = {'shape': (11,11)}
dtype = [(l, 'float32') for l in ('X', 'Y', 'C')]

# Create droplet centered at (5,5)
mid, spread = 5, 10
cs = np.exp(-((xs-mid)**2 + (ys-mid)**2)/spread)

# System
system = np.zeros(xs.size, dtype=dtype).ravel()
system['X'] = xs.ravel()
system['Y'] = ys.ravel()
system['C'] = cs.ravel()

def test_find_indices_in_radius():
    cell = int(np.floor(system.size/2))

    radius = 1 - 1e-6
    indices = intf._get_indices_in_radius(cell, system, radius)
    assert (len(indices) == 0)

    radius = 1
    indices = intf._get_indices_in_radius(cell, system, radius)
    assert (set(indices) == set([49, 59, 61, 71]))

    radius = 1.5
    indices = intf._get_indices_in_radius(cell, system, radius)
    assert (set(indices) == set([48, 49, 50, 59, 61, 70, 71, 72]))

def test_find_indices_in_radius_general_names():
    dtype = [(l, 'float32') for l in ('f0', 'f1', 'C')]
    system = np.zeros(xs.size, dtype=dtype).ravel()
    system['f0'] = xs.ravel()
    system['f1'] = ys.ravel()

    cell = int(np.floor(system.size/2))

    radius = 1
    indices = intf._get_indices_in_radius(cell, system, radius, coord_labels=('f0', 'f1'))
    assert (set(indices) == set([49, 59, 61, 71]))

def test_cell_is_droplet():
    datasize = 5
    system = np.zeros(datasize, dtype=dtype).ravel()
    system['X'] = np.arange(datasize)
    system['Y'] = np.arange(datasize)
    system['C'][[1,2,3]] = 1

    label = 'C'
    radius = 1.5
    cutoff = 0.5

    droplet = []
    for cell in np.arange(system.size):
        droplet.append(intf._cell_is_droplet(cell, system, label, radius, cutoff))
    assert (droplet == [False, True, True, True, False])

def test_cell_is_droplet_cutoff():
    datasize = 5
    system = np.zeros(datasize, dtype=dtype).ravel()
    system['X'] = np.arange(datasize)
    system['Y'] = np.arange(datasize)
    system['C'][[1,2,3]] = 1

    label = 'C'
    radius = 1.5

    droplet = []
    cutoff = 2
    for cell in np.arange(system.size):
        droplet.append(intf._cell_is_droplet(cell, system, label, radius, cutoff))
    assert (droplet == [False, False, False, False, False])

    droplet = []
    cutoff = 0.5
    for cell in np.arange(system.size):
        droplet.append(intf._cell_is_droplet(cell, system, label, radius, cutoff))
    assert (droplet == [False, True, True, True, False])

def test_cell_is_droplet_numbins():
    datasize = 5
    system = np.zeros(datasize, dtype=dtype).ravel()
    system['X'] = np.arange(datasize)
    system['Y'] = np.arange(datasize)
    system['C'][[1,2,3]] = 1

    label = 'C'
    radius = 1.5
    cutoff = 0.5
    cutoff_bins = 2

    droplet = []
    for cell in np.arange(system.size):
        droplet.append(intf._cell_is_droplet(cell, system, label, radius, cutoff,
            cutoff_bins=cutoff_bins))
    assert (droplet == [False, False, True, False, False])

def test_cell_is_droplet_general_names():
    datasize = 5
    dtype = [(l, 'float32') for l in ('f0', 'f1', 'C')]
    system = np.zeros(datasize, dtype=dtype).ravel()
    system['f0'] = np.arange(datasize)
    system['f1'] = np.arange(datasize)
    system['C'][[1,2,3]] = 1

    coord_labels = ('f0', 'f1')
    label = 'C'
    cutoff = 1
    radius = 1.5

    droplet = []
    for cell in np.arange(system.size):
        droplet.append(intf._cell_is_droplet(cell, system, label, radius, cutoff,
            coord_labels=coord_labels))
    assert (droplet == [False, True, True, True, False])

def test_cell_is_droplet_bad_names():
    datasize = 1
    dtype = [(l, 'float32') for l in ('f0', 'f1', 'C')]
    system = np.zeros(datasize, dtype=dtype).ravel()
    system['f0'] = np.arange(datasize)
    system['f1'] = np.arange(datasize)

    coord_labels = ('c0', 'c1')
    label = 'E'
    radius = 1.5
    cutoff = 0.5

    with pytest.raises(KeyError):
        intf._cell_is_droplet(0, system, label, radius, cutoff)

def test_find_interface_bottom():
    datasize = 5
    system = {}
    system['X'] = np.arange(datasize)
    system['Y'] = np.zeros(datasize)
    system['C'] = np.array([0, 1, 1, 1, 0])
    flow = FlowData(system)

    label = 'C'
    radius = 1.5

    for i, (ileft, iright) in enumerate(get_interface(flow, label, cutoff_radius=radius)):
        assert (ileft == 1 and iright == 3)
        break

    cutoff_bins = 2
    for ileft, iright in get_interface(flow, label, cutoff_radius=radius, cutoff_bins=cutoff_bins):
        assert (ileft == 2 and iright == 2)
        break

def test_find_interface():
    flow = FlowData(('X', xs), ('Y', ys), ('C', cs))
    control = flow.data.copy()

    label = 'C'
    radius = 1.5

    interface = []
    for i, (ileft, iright) in enumerate(get_interface(flow, label, cutoff_radius=radius)):
        interface.append(flow.data[[ileft, iright]])
    assert (len(interface) == 5)

    interface = []
    cutoff = 0.2
    for i, (ileft, iright) in enumerate(get_interface(flow, label, cutoff_radius=radius,
            cutoff=cutoff)):
        interface.append(flow.data[[ileft, iright]])
    assert (len(interface) == 9)

    interface = []
    cutoff = 1.1
    for i, (ileft, iright) in enumerate(get_interface(flow, label, cutoff_radius=radius,
            cutoff=cutoff)):
        interface.append(flow.data[[ileft, iright]])
    assert (len(interface) == 0)
    assert (np.array_equal(control.data, flow.data))

def test_find_interface_ylims():
    flow = FlowData(('X', xs), ('Y', ys), ('C', cs))

    label = 'C'
    radius = 1.1
    ylims = (4.5, 7.5)

    yvalues = []
    for i, (ileft, _) in enumerate(get_interface(flow, label, cutoff_radius=radius,
            ylims=ylims)):
        yvalues.append(flow.data[ileft]['Y'])
    assert (yvalues == [5, 6, 7])

def test_find_interface_noboundaries(recwarn):
    cs = 0*xs
    flow = FlowData(('X', xs), ('Y', ys), ('C', cs))

    label = 'C'
    radius = 1.1

    for left, right in get_interface(flow, label, cutoff_radius=radius):
        assert (left == None and right == None)

def test_find_interface_noradius():
    flow = FlowData(('X', xs), ('Y', ys), ('C', cs))

    label = 'C'

    interface = []
    for i, (ileft, iright) in enumerate(get_interface(flow, label, cutoff_radius=None)):
        interface.append(flow.data[[ileft, iright]])
    assert (len(interface) == 5)

def test_find_interface_from_inside_out():
    # Layer bins:      1 1 0 1 2 1 1 0 1
    # Detect edges at:       x     x
    # I.e.                   3     6
    xs = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8.])
    ms = np.array([1., 1., 0., 1., 2., 1., 1., 0., 1.])
    ys = np.zeros(xs.shape)

    info = {
        'shape': (9, 1),
        'spacing': (1., 1.)
    }

    flow = FlowData(('X', xs), ('Y', ys), ('M', ms), info=info)
    interface = get_interface(flow, 'M',
        search_longest_connected=True,
        cutoff=0.5, cutoff_radius=1.1, cutoff_bins=1
    )

    left, right = next(interface)
    assert (left == 3)
    assert (right == 6)

def test_find_interface_from_inside_out_requires_system_size_for_pbc():
    xs = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8.])
    ms = np.array([1., 1., 0., 1., 2., 1., 1., 0., 1.])
    ys = np.zeros(xs.shape)

    flow = FlowData(('X', xs), ('Y', ys), ('M', ms))

    with pytest.raises(ValueError):
        interface = next(get_interface(flow, 'M',
            search_longest_connected=True))

def test_find_interface_from_inside_out_ensures_that_pbc_is_adjusted_for():
    # Layer bins:      1 0 1 2 1 0 1 1 1
    # Detect edges at: x           x
    # I.e.             0           6
    # But -- the left edge is at 6, due to the periodic image connection!
    xs = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8.])
    ms = np.array([1., 0., 1., 2., 1., 0., 1., 1., 1.])
    ys = np.zeros(xs.shape)

    info = {
        'shape': (9, 1),
        'spacing': (1., 1.)
    }

    flow = FlowData(('X', xs), ('Y', ys), ('M', ms), info=info)
    interface = get_interface(flow, 'M', search_longest_connected=True)

    left, right = next(interface)
    assert (left == 6)
    assert (right == 0)

def test_find_interface_from_inside_out_with_all_and_none_filled():
    # No interface in this!
    # Layer bins:      0 0 0 0 0 0 0 0 0
    xs = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8.])
    ms = np.zeros(xs.shape)
    ys = np.zeros(xs.shape)

    info = {
        'shape': (9, 1),
        'spacing': (1., 1.)
    }

    flow = FlowData(('X', xs), ('Y', ys), ('M', ms), info=info)
    interface = get_interface(flow, 'M', search_longest_connected=True)
    assert len(list(interface)) == 0

    # The interface is filled!
    # Layer bins:      1 1 1 1 1 1 1 1 1
    # Detect edges at: x               x
    # I.e.             0               8
    xs = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8.])
    ms = np.ones(xs.shape)
    ys = np.zeros(xs.shape)

    flow = FlowData(('X', xs), ('Y', ys), ('M', ms), info=info)
    interface = get_interface(flow, 'M',
        search_longest_connected=True, cutoff=0.5)

    left, right = next(interface)
    assert (left == 0)
    assert (right == 8)

def test_find_complete_interface_with_largest_connected_method():
    xs = np.array([
        0., 1., 2., 3., 4., 5., 6.,
        0., 1., 2., 3., 4., 5., 6.,
        0., 1., 2., 3., 4., 5., 6.,
        0., 1., 2., 3., 4., 5., 6.,
        0., 1., 2., 3., 4., 5., 6.,
        0., 1., 2., 3., 4., 5., 6.,
        0., 1., 2., 3., 4., 5., 6.
    ])

    ys = np.array([
        0., 0., 0., 0., 0., 0., 0.,
        1., 1., 1., 1., 1., 1., 1.,
        2., 2., 2., 2., 2., 2., 2.,
        3., 3., 3., 3., 3., 3., 3.,
        4., 4., 4., 4., 4., 4., 4.,
        5., 5., 5., 5., 5., 5., 5.,
        6., 6., 6., 6., 6., 6., 6.
    ])

    ms = np.array([
        0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 1., 0., 0., 0.,
        0., 0., 1., 1., 1., 0., 0.,
        0., 1., 1., 1., 1., 1., 0.,
        0., 0., 1., 1., 1., 0., 0.,
        0., 0., 0., 1., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0.
    ])

    info = {
        'shape': (7, 7),
        'spacing': (1., 1.),
    }

    flow = FlowData(('X', xs), ('Y', ys), ('M', ms), info=info)
    interface = get_interface(flow, 'M',
        search_longest_connected=True, cutoff=0.5)

    # The first layer is empty and skipped.
    # Second goes from (3, 3), but in the second layer so we add the size once
    left, right = next(interface)
    assert left == 3 + 7
    assert right == 3 + 7

    # Third goes from (2, 4), but in the third layer so we add the size twice
    left, right = next(interface)
    assert left == 2 + 2 * 7
    assert right == 4 + 2 * 7

    # Fourth goes from (1, 5), but in the fourth layer so we add the size 3x
    left, right = next(interface)
    assert left == 1 + 3 * 7
    assert right == 5 + 3 * 7

    # Fifth goes from (2, 4), but in the fifth layer so we add the size 4x
    left, right = next(interface)
    assert left == 2 + 4 * 7
    assert right == 4 + 4 * 7

    # Sixth goes from (3, 3), but in the sixth layer so we add the size 5x
    left, right = next(interface)
    assert left == 3 + 5 * 7
    assert right == 3 + 5 * 7

    # There are no further interfaces.
    assert len(list(interface)) == 0

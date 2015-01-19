import numpy as np
import pytest
from droplets.flow import FlowData
from droplets.droplet import *

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
    cell = np.floor(system.size/2)

    radius = 1 - 1e-6
    indices = get_indices_in_radius(cell, system, radius)
    assert (len(indices) == 0)

    radius = 1
    indices = get_indices_in_radius(cell, system, radius)
    assert (set(indices) == set([49, 59, 61, 71]))

    radius = 1.5
    indices = get_indices_in_radius(cell, system, radius)
    assert (set(indices) == set([48, 49, 50, 59, 61, 70, 71, 72]))

def test_find_indices_in_radius_general_names():
    dtype = [(l, 'float32') for l in ('f0', 'f1', 'C')]
    system = np.zeros(xs.size, dtype=dtype).ravel()
    system['f0'] = xs.ravel()
    system['f1'] = ys.ravel()

    cell = np.floor(system.size/2)

    radius = 1
    indices = get_indices_in_radius(cell, system, radius, coord_labels=('f0', 'f1'))
    assert (set(indices) == set([49, 59, 61, 71]))

def test_cell_is_droplet():
    datasize = 5
    system = np.zeros(datasize, dtype=dtype).ravel()
    system['X'] = np.arange(datasize)
    system['Y'] = np.arange(datasize)
    system['C'][[1,2,3]] = 1

    label = 'C'
    radius = 1.5

    droplet = []
    for cell in np.arange(system.size):
        droplet.append(cell_is_droplet(cell, system, label, radius))
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
        droplet.append(cell_is_droplet(cell, system, label, radius, cutoff=cutoff))
    assert (droplet == [False, False, False, False, False])

    droplet = []
    cutoff = 0.5
    for cell in np.arange(system.size):
        droplet.append(cell_is_droplet(cell, system, label, radius, cutoff=cutoff))
    assert (droplet == [False, True, True, True, False])



def test_cell_is_droplet_numcells():
    datasize = 5
    system = np.zeros(datasize, dtype=dtype).ravel()
    system['X'] = np.arange(datasize)
    system['Y'] = np.arange(datasize)
    system['C'][[1,2,3]] = 1

    label = 'C'
    radius = 1.5
    num_cells = 2

    droplet = []
    for cell in np.arange(system.size):
        droplet.append(cell_is_droplet(cell, system, label, radius,
            num_cells=num_cells))
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
        droplet.append(cell_is_droplet(cell, system, label, radius,
            coord_labels=coord_labels))
    assert (droplet == [False, True, True, True, False])

def test_cell_is_droplet_bad_names():
    datasize = 1
    dtype = [(l, 'float32') for l in ('f0', 'f1', 'C')]
    system = np.zeros(datasize, dtype=dtype).ravel()
    system['f0'] = np.arange(datasize)
    system['f1'] = np.arange(datasize)

    coord_labels = ('c0', 'c1')
    label = 'C'
    radius = 1.5

    with pytest.raises(IndexError):
        cell_is_droplet(0, system, label, radius)

def test_find_interface_bottom():
    datasize = 5
    system = {}
    system['X'] = np.arange(datasize)
    system['Y'] = np.zeros(datasize)
    system['C'] = np.array([0, 1, 1, 1, 0])
    flow = FlowData(system)

    label = 'C'
    radius = 1.5

    for i, (ileft, iright) in enumerate(get_interface(flow, label, radius)):
        assert (ileft == 1 and iright == 3)
        break

    num_cells = 2
    for ileft, iright in get_interface(flow, label, radius, num_cells=num_cells):
        assert (ileft == 2 and iright == 2)
        break

def test_find_interface():
    flow = FlowData(('X', xs), ('Y', ys), ('C', cs))

    label = 'C'
    radius = 1.5

    interface = []
    for i, (ileft, iright) in enumerate(get_interface(flow, label, radius)):
        interface.append(flow.data[[ileft, iright]])
    assert (len(interface) == 5)

    interface = []
    cutoff = 0.2
    for i, (ileft, iright) in enumerate(get_interface(flow, label, radius,
            cutoff=cutoff)):
        interface.append(flow.data[[ileft, iright]])
    assert (len(interface) == 9)

    interface = []
    cutoff = 1.1
    for i, (ileft, iright) in enumerate(get_interface(flow, label, radius,
            cutoff=cutoff)):
        interface.append(flow.data[[ileft, iright]])
    assert (len(interface) == 0)

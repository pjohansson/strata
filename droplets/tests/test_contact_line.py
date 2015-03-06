import numpy as np
import pytest

from droplets.flow import FlowData
from droplets.droplet import get_interface
from droplets.contact_line import *

def get_data(size, base, height):
    """Function to create a simple triangle data map."""

    x = np.arange(size)
    xs, ys = np.meshgrid(x, x)
    mid = int(size/2)
    cs = np.zeros((size, size))

    for h in range(height):
        try:
            cs[h][mid-base:mid+base] = 1.
            base += 1
        except Exception:
            break

    return {'X': xs}, {'Y': ys}, {'C': cs}

dtype = [('X', 'float'), ('Y', 'float'), ('C', 'float')]

def test_extract_cells():
    data = get_data(10, 2, 3)
    flow = FlowData(*data)

    interface = list(get_interface(flow, 'C', 1, num_bins=0))
    il, ir = interface[0]

    # Match top interface layer for several extractions
    for i, (il, ir) in enumerate(interface):
        left, right = get_contact_line_cells(flow, 'C', size=(1.5, i))
        imod = (2 + i)*i

        assert np.array_equal(left[imod:], flow.data[il:il+2+i])
        assert np.array_equal(right[imod:], flow.data[ir-i-1:ir+1])

def test_extract_cells_failure():
    data = get_data(10, 2, 3)
    flow = FlowData(*data, dtype=dtype)
    empty = np.array([], dtype=dtype)

    for cells in get_contact_line_cells(flow, 'C', cutoff=2):
        assert np.array_equal(cells, empty)

def test_extract_cells_nocutoff():
    data = get_data(10, 2, 3)
    flow = FlowData(*data)
    ir = len(np.unique(data[0]['X']))-1

    left, right = get_contact_line_cells(flow, 'C', size=(2, 0), cutoff=0)
    assert np.array_equal(left, flow.data[0:3])
    assert np.array_equal(right, flow.data[ir-2:ir+1])

def test_adjust_cells():
    data = get_data(10, 2, 3)
    flow = FlowData(*data)
    left, right = get_contact_line_cells(flow, 'C', size=(2, 0))

    for cells, dir_mod in zip((left, right), ('left', 'right')):
        (xadj, yadj), adj_cells = adjust_cell_coordinates(cells, dir_mod)
        assert (yadj == 0.)
        assert np.array_equal(adj_cells['X'] + xadj, cells['X'])

def test_adjust_cells_badkey():
    data = get_data(10, 2, 3)
    flow = FlowData(*data)
    left, _ = get_contact_line_cells(flow, 'C', size=(2, 0))

    with pytest.raises(KeyError):
        adjust_cell_coordinates(left, '')

def test_adjust_cells_nodata():
    cells = np.zeros(0, dtype=dtype)
    adj, adj_cells = adjust_cell_coordinates(cells, 'left')
    assert (adj == (0, 0))
    assert np.array_equal(cells, adj_cells)

def test_adjust_cells_height():
    data = get_data(10, 2, 3)
    data[2]['C'][0] = 0.
    flow = FlowData(*data)
    left, _ = get_contact_line_cells(flow, 'C', size=(2, 2))
    (_, yadj), adj_cells = adjust_cell_coordinates(left, 'left')
    assert (yadj == 1.)
    assert np.array_equal(adj_cells['Y'] + yadj, left['Y'])

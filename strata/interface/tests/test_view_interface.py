import numpy as np
import os
import tempfile as tmp

from droplets.flow import FlowData
from droplets.interface import get_interface
from strata.interface.collect import get_interface_coordinates
from strata.interface.view import *
from strata.dataformats.write import write, flowdata_to_dict
from strata.utils import gen_filenames, find_datamap_files

coords = np.linspace(0, 10, 11)
xs, ys = np.meshgrid(coords, coords)
cs = np.zeros(xs.shape)
cs[:,1:10] = 1.

data = {'X': xs, 'Y': ys, 'C': cs}
flow = FlowData(data)

def test_get_series():
    x, y = get_interface_coordinates(flow, 'C')
    xs = [x, x + 1]
    ys = [y, y + 2]

    left, right = all_coords_to_edges(xs, ys)

    for i, (l, r) in enumerate(zip(left, right)):
        split = int(len(xs[i])/2)
        assert (np.array_equal(l.index, ys[i][:split]))
        assert (np.array_equal(l.index, r.index[::-1]))
        assert (np.array_equal(l, xs[i][:split]))
        assert (np.array_equal(r, xs[i][split:]))

def test_combine_interfaces():
    x, y = get_interface_coordinates(flow, 'C')

    xs = [x, x + 1]
    ys = [y, y + 2]

    left, right = combine_interfaces(xs, ys)
    assert np.array_equal(np.unique(left.index), np.unique(ys))
    assert np.array_equal(left.index, right.index)

    for i, (x, y) in enumerate(zip(xs, ys)):
        inds = left[i].notnull()
        num = np.floor(len(y)/2)

        # Assert that non-NaN-values are identical to original arrays
        assert (np.array_equal(left[inds].index, y[:num]))
        assert (np.array_equal(left[i][inds], x[:num]))
        assert (np.array_equal(right[i][inds], x[num:]))

def test_stitch_edges():
    x, y = get_interface_coordinates(flow, 'C')
    control = pd.Series(x, index=y)
    left, right = all_coords_to_edges([x], [y])
    series = stitch_edge_series(left[0], right[0])

    assert (np.array_equal(series, control))
    assert (np.array_equal(series.index, control.index))

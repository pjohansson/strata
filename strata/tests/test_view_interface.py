import numpy as np
import os
import tempfile as tmp

from droplets.flow import FlowData
from droplets.interface import get_interface
from strata.interface import *
from strata.dataformats.write import write, flowdata_to_dict
from strata.utils import gen_filenames, find_datamap_files

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

def test_get_series():
    x, y = get_interface_coordinates(flow, 'C', adjust_com=True)
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

def test_read_interface():
    empty = np.empty_like(xs)
    data = {'X': xs, 'Y': ys, 'M': cs,
            'U': empty, 'V': empty, 'N': empty, 'T': empty
            }

    flow = FlowData(data)
    x, y = get_interface_coordinates(flow, 'M', adjust_com=True)

    with tmp.TemporaryDirectory() as tmpdir:
        fnbase = os.path.join(tmpdir, 'test_')
        path = next(gen_filenames(fnbase))
        write(path, flowdata_to_dict(flow))

        xs_list, ys_list = read_interfaces(fnbase)
        assert (np.array_equal(xs_list[0], x))
        assert (np.array_equal(ys_list[0], y))

def test_read_average_interfaces():
    empty = np.empty_like(xs)
    data1 = {'X': xs, 'Y': ys, 'M': cs,
            'U': empty, 'V': empty, 'N': empty, 'T': empty
            }
    flow1 = FlowData(data1)
    x1, y1 = get_interface_coordinates(flow1, 'M', adjust_com=True)

    cs[:,2:9] = 5.
    data2 = {'X': xs, 'Y': ys, 'M': cs,
            'U': empty, 'V': empty, 'N': empty, 'T': empty
            }
    flow2 = FlowData(data2)
    x2, y2 = get_interface_coordinates(flow2, 'M', adjust_com=True)
    xavg = np.array([x1, x2]).mean(axis=0)
    yavg = np.array([y1, y2]).mean(axis=0)

    with tmp.TemporaryDirectory() as tmpdir:
        fnbase = os.path.join(tmpdir, 'test_')
        outbase = os.path.join(tmpdir, 'out_')
        fngen = gen_filenames(fnbase)
        path1 = next(fngen)
        path2 = next(fngen)
        write(path1, flowdata_to_dict(flow1))
        write(path2, flowdata_to_dict(flow2))

        xs_list, ys_list = read_interfaces(fnbase, average=2, save_xvg=outbase)
        assert (np.array_equal(xs_list[0], xavg))
        assert (np.array_equal(ys_list[0], yavg))

        # Check output
        out = list(find_datamap_files(outbase, ext='.xvg'))
        assert (len(out) == 1)
        
        y_read, x_read = np.genfromtxt(out[0], comments='#', unpack=True)
        assert (np.array_equal(x_read, xavg))
        assert (np.array_equal(y_read, yavg))

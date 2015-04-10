import numpy as np
import os
import tempfile as tmp

from droplets.flow import FlowData
from droplets.interface import get_interface
from strata.interface.collect import *
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
    x, y = get_interface_coordinates(flow, 'C', recenter='com')

    assert np.isclose(com, 5.)
    assert np.array_equiv(x[:len(coords)], 1.-com)
    assert np.array_equiv(x[len(coords):], 9.-com)
    assert np.array_equal(y[:len(coords)], coords)
    assert np.array_equal(y[len(coords):], coords[::-1])

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

        xs_list, ys_list = collect_interfaces(fnbase, os.path.join(tmpdir,''))
        assert (np.array_equal(xs_list[0], x))
        assert (np.array_equal(ys_list[0], y))

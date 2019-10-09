import numpy as np
import os
import pytest
import tempfile as tmp

from droplets.flow import FlowData
from droplets.interface import get_interface
from strata.interface.collect import *
from strata.dataformats.write import write, flowdata_to_dict
from strata.spreading.collect import init_periodic_info, \
    check_and_update_periodic_info, add_pbc_multipliers_to_edges, PeriodicInfo
from strata.utils import gen_filenames, find_datamap_files

coords = np.linspace(0, 10, 11)
xs, ys = np.meshgrid(coords, coords)
cs = np.zeros(xs.shape)
cs[:,1:10] = 1.

data = {'X': xs, 'Y': ys, 'C': cs}
info = {'shape': (11, 11), 'spacing': (1., 1.)}
flow = FlowData(data, info=info)

def test_get_interface_coords():
    interface, _, _ = get_interface_coordinates(flow, 'C', None, None)
    x = interface['X']
    y = interface['Y']

    assert np.array_equiv(x[:len(coords)], 1.)
    assert np.array_equiv(x[len(coords):], 9.)
    assert np.array_equal(y[:len(coords)], coords)
    assert np.array_equal(y[len(coords):], coords[::-1])

def test_adjust_interface_to_com():
    com = np.average(flow.data['X'], weights=flow.data['C'])
    interface, _, _ = get_interface_coordinates(flow, 'C', None, None, recenter='com')
    x = interface['X']
    y = interface['Y']

    assert np.isclose(com, 5.)
    assert np.array_equiv(x[:len(coords)], 1.-com)
    assert np.array_equiv(x[len(coords):], 9.-com)
    assert np.array_equal(y[:len(coords)], coords)
    assert np.array_equal(y[len(coords):], coords[::-1])

def init_pbc_info_with_positions(xleft, xright):
    return PeriodicInfo(
        xleft_prev=xleft,
        xright_prev=xright,
        pbc_multiplier_left=0,
        pbc_multiplier_right=0,
    )

def test_get_initial_yvalue_meta_data():
    ys = 1. + np.arange(10.) / 9.
    yindex = get_yindex_data(ys)

    assert (yindex.ymin == 1.)
    assert (yindex.ymax == 1. + (10. - 1.) / 9.)
    assert (np.isclose(yindex.dy, 1. / 9.))
    assert (yindex.num == 10)

def test_get_yvalue_indices_using_meta_data():
    ys = 1. + np.arange(10.)
    yindex = get_yindex_data(ys)

    assert (get_yindex_for_value(1.0, yindex) == 0)
    assert (get_yindex_for_value(2.0, yindex) == 1)
    assert (get_yindex_for_value(3.0, yindex) == 2)
    assert (get_yindex_for_value(10.0, yindex) == 9)

def test_get_yvalue_indices_for_out_of_bounds_data_is_edges():
    ys = 1. + np.arange(10.)
    yindex = get_yindex_data(ys)

    assert (get_yindex_for_value(0.0, yindex) == 0)
    assert (get_yindex_for_value(11.0, yindex) == 9)

def test_get_yvalue_indices_rounds_to_closest_match():
    ys = 1. + np.arange(10.) / 9.
    yindex = get_yindex_data(ys)

    assert (get_yindex_for_value(1. + 1. / 9. - 1e-3, yindex) == 0)
    assert (get_yindex_for_value(1. + 1. / 9., yindex) == 1)
    assert (get_yindex_for_value(1. + 1. / 9. + 1e-3, yindex) == 1)

def test_create_superset_of_yvalues_with_none_when_getting_larger_meta_data():
    ys_old = 2. + np.arange(5.) # [2., 6.]
    ys_new = np.arange(9.)      # [0., 8.]

    xs_old = 0.5 * ys_old

    # The superset should initialize to None
    xs_new_expected = np.array([None for _ in range(ys_new.size)])
    xs_new_expected[2:7] = xs_old

    yindex_data_old = get_yindex_data(ys_old)
    yindex_data_new = get_yindex_data(ys_new)

    xs_new = transfer_values(xs_old, yindex_data_old, yindex_data_new)

    assert (np.array_equal(xs_new, xs_new_expected))

def test_create_set_of_yvalues_when_getting_non_overlapping_meta_data():
    ys_old = np.arange(9.)      # [0., 8.]
    ys_new = 2. + np.arange(8.) # [2., 9.]

    xs_old = 0.5 * ys_old

    # The set cuts the first 2 value from the old and initializes unknown
    # to None
    xs_new_expected = np.array([None for _ in range(ys_new.size)])
    xs_new_expected[0:7] = xs_old[2:]

    yindex_data_old = get_yindex_data(ys_old)
    yindex_data_new = get_yindex_data(ys_new)

    xs_new = transfer_values(xs_old, yindex_data_old, yindex_data_new)

    assert (np.array_equal(xs_new, xs_new_expected))

def test_create_subset_of_yvalues_when_getting_smaller_meta_data():
    ys_old = np.arange(9.)      # [0., 8.]
    ys_new = 2. + np.arange(5.) # [2., 6.]

    xs_old = 0.5 * ys_old

    yindex_data_old = get_yindex_data(ys_old)
    yindex_data_new = get_yindex_data(ys_new)

    xs_new = transfer_values(xs_old, yindex_data_old, yindex_data_new)

    assert (np.array_equal(xs_new, xs_old[2:7]))

def test_create_superset_of_yvalues_initializes_with_none():
    ys_old = None
    ys_new = np.arange(9.)      # [0., 8.]

    xs_old = None

    # The superset should initialize to None
    xs_new_expected = np.array([None for _ in range(ys_new.size)])

    yindex_data_old = None
    yindex_data_new = get_yindex_data(ys_new)

    xs_new = transfer_values(xs_old, yindex_data_old, yindex_data_new)

    assert (np.array_equal(xs_new, xs_new_expected))

def test_create_superset_of_yvalues_works_for_no_change():
    ys = 2. + np.arange(5.)
    xs = 0.5 * ys

    yindex_data = get_yindex_data(ys)

    xs_new = transfer_values(xs, yindex_data, yindex_data)

    assert (id(xs_new) == id(xs))

def test_create_superset_of_yvalues_fails_if_dy_are_not_the_same():
    ys_dy1 = np.arange(5.)
    ys_dy2 = 2 * ys_dy1
    xs = 0.5 * ys_dy1

    yindex_data_dy1 = get_yindex_data(ys_dy1)
    yindex_data_dy2 = get_yindex_data(ys_dy2)

    with pytest.raises(ValueError) as exc:
        transfer_values(xs, yindex_data_dy1, yindex_data_dy2)


def test_create_interface_from_ys_and_xs_values():
    n = 10
    ys = np.arange(n)
    xs_left = 0.5 * ys
    xs_right = 1.5 * ys

    interface = create_interface_array(ys, xs_left, xs_right)

    assert (np.array_equal(interface['Y'][:n], ys))
    assert (np.array_equal(interface['Y'][n:], ys[::-1]))

    assert (np.array_equal(interface['X'][:n], xs_left))
    assert (np.array_equal(interface['X'][n:], xs_right[::-1]))

def test_fill_pbc_info_for_y_values_picks_closest():
    pbc_info_1 = init_periodic_info(xleft=1., xright=2.)
    pbc_info_2 = init_periodic_info(xleft=0., xright=3.)

    pbc_info_per_y = [
        None,
        None,
        pbc_info_1,
        pbc_info_2,
        pbc_info_1,
        pbc_info_2,
        None,
        None
    ]

    pbc_info_per_y = fill_pbc_info_for_empty_y_values(pbc_info_per_y)

    assert (pbc_info_per_y[0] == pbc_info_1)
    assert (pbc_info_per_y[1] == pbc_info_1)
    assert (pbc_info_per_y[2] == pbc_info_1)
    assert (pbc_info_per_y[3] == pbc_info_2)
    assert (pbc_info_per_y[4] == pbc_info_1)
    assert (pbc_info_per_y[5] == pbc_info_2)
    assert (pbc_info_per_y[6] == pbc_info_2)
    assert (pbc_info_per_y[7] == pbc_info_2)

def test_fill_pbc_info_for_y_values_picks_default_if_none_are_found():
    pbc_info_default = init_periodic_info()

    pbc_info_per_y = [
        None,
        None,
        None,
        None
    ]

    pbc_info_per_y = fill_pbc_info_for_empty_y_values(pbc_info_per_y)

    for pbc_info in pbc_info_per_y:
        assert (pbc_info == pbc_info_default)

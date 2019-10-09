import numpy as np
import os
import pytest
import tempfile as tmp

from droplets.flow import FlowData
from strata.dataformats.write import write
from strata.spreading.collect import *
from strata.utils import gen_filenames

datasize = 12

class TestGetSpread(object):
    def test_one_layer(self):
        xs = np.linspace(0, 1, datasize)
        ys = np.zeros(datasize) + 0.5
        cs = np.zeros(datasize)

        xmin, xmax = (2, 8)
        cs[xmin:xmax+1] = 1

        label = 'C'
        radius = (xs[1] - xs[0])*1.1
        flow = FlowData(('X', xs), ('Y', ys), (label, cs))

        left, right = get_spreading_edges(flow, label, radius)
        assert (left == xs[xmin])
        assert (right == xs[xmax])

    def test_one_layer_nointerface(self):
        xs = np.linspace(0, 1, datasize)
        ys = np.zeros(datasize) + 0.5
        cs = np.zeros(datasize)

        label = 'C'
        radius = xs[1] - xs[0]
        cutoff = 1
        flow = FlowData(('X', xs), ('Y', ys), (label, cs))

        left, right = get_spreading_edges(flow, label, radius, cutoff=cutoff)
        assert (left == None and right == None)

    def test_two_layers(self):
        X = np.linspace(0, 1, datasize)
        Y = [1, 2]
        xs, ys = np.meshgrid(X, Y)
        cs = np.zeros((2, datasize))

        xmin, xmax = (2, 8)
        cs[1][xmin:xmax+1] = 1

        label = 'C'
        radius = (X[1] -  X[0])*1.1
        flow = FlowData(('X', xs), ('Y', ys), (label, cs))

        left, right = get_spreading_edges(flow, label, radius)
        assert (left == X[xmin] and right == X[xmax])

    def test_floor(self):
        X = np.linspace(0, 1, datasize)
        Y = [1, 2]
        xs, ys = np.meshgrid(X, Y)
        cs = np.zeros((2, datasize))

        xmin, xmax = (2, 8)
        cs[0][xmin-1:xmax+2] = 1
        cs[1][xmin:xmax+1] = 1

        label = 'C'
        radius = (X[1] -  X[0])*1.1
        floor = 1.1
        flow = FlowData(('X', xs), ('Y', ys), (label, cs))

        # Assert that bins are in second layer
        left, right = get_spreading_edges(flow, label, radius, floor=floor)
        assert (left == X[xmin] and right == X[xmax])

    def test_otheropts(self):
        xs = np.linspace(0, 1, datasize)
        ys = np.zeros(datasize) + 0.5
        cs = np.ones(datasize)

        xmin, xmax = (2, 8)
        cs[xmin:xmax+1] = 2

        label = 'C'
        clabels = ('f0', 'f1')
        radius = 2*(xs[1] - xs[0])*1.1
        cutoff_bins = 2
        cutoff = 2

        kwargs = {
                'coord_labels': clabels,
                'cutoff': cutoff,
                'cutoff_bins': cutoff_bins
                }

        flow = FlowData((clabels[0], xs), (clabels[1], ys), (label, cs))

        left, right = get_spreading_edges(flow, label, radius, **kwargs)
        assert (left == xs[xmin] and right == xs[xmax])

    def test_collect_edges_from_longest_filled_section_with_pbc(self):
        # Layer bins:      1 1 0 1 2 1 0 1 1
        # Detect edges at:   x           x
        # I.e.               1           7.
        # But -- reversed, since the edge at 7 is the left edge
        # when accounting for the periodic image
        xs = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8.])
        ms = np.array([1., 1., 0., 1., 2., 1., 0., 1., 1.])
        ys = np.zeros(xs.shape)

        info = {
            'shape': (9, 1),
            'spacing': (1., 1.),
        }

        flow = FlowData(('X', xs), ('Y', ys), ('M', ms), info=info)

        left, right = get_spreading_edges(flow, 'M', 1.1,
            search_longest_connected=True)

        assert (left == 7.0)
        assert (right == 1.0)

def test_init_periodic_info_to_0_and_none():
    pbc_info = init_periodic_info()

    assert (pbc_info.xleft_prev == None)
    assert (pbc_info.xright_prev == None)
    assert (pbc_info.pbc_multiplier_left == 0)
    assert (pbc_info.pbc_multiplier_right == 0)

def test_init_periodic_info_to_xvalues():
    pbc_info = init_periodic_info(xleft=1., xright=2.)

    assert (pbc_info.xleft_prev == 1.)
    assert (pbc_info.xright_prev == 2.)
    assert (pbc_info.pbc_multiplier_left == 0)
    assert (pbc_info.pbc_multiplier_right == 0)

def test_periodic_info_check_updates_previous_positions():
    pbc_info = init_periodic_info()

    x0 = 1.
    x1 = 2.
    pbc_info = check_and_update_periodic_info(pbc_info, 10., x0, x1)

    assert (pbc_info.xleft_prev == x0)
    assert (pbc_info.xright_prev == x1)

def init_pbc_info_with_positions(xleft, xright):
    return PeriodicInfo(
        xleft_prev=xleft,
        xright_prev=xright,
        pbc_multiplier_left=0,
        pbc_multiplier_right=0,
    )

def init_pbc_info_with_positions_and_multipliers(xleft, xright,
        pbcleft, pbcright):
    return PeriodicInfo(
        xleft_prev=xleft,
        xright_prev=xright,
        pbc_multiplier_left=pbcleft,
        pbc_multiplier_right=pbcright,
    )

def test_periodic_info_check_updates_multipliers_if_edges_are_switched():
    xleft_prev = 1.
    xright_prev = 2.
    pbc_info = init_pbc_info_with_positions(xleft_prev, xright_prev)

    no_update_info = check_and_update_periodic_info(
        pbc_info, 10., 5., 6.
    )
    assert (no_update_info.pbc_multiplier_left == 0)
    assert (no_update_info.pbc_multiplier_right == 0)

    crossed_info = check_and_update_periodic_info(
        pbc_info, 10., 6., 5.
    )
    assert (crossed_info.pbc_multiplier_left != 0 \
        or crossed_info.pbc_multiplier_right != 0)

def test_periodic_info_updates_the_multiplier_of_the_edge_that_moves_most():
    xleft_prev = 1.
    xright_prev = 2.
    pbc_info = init_pbc_info_with_positions(xleft_prev, xright_prev)

    # dx_left = 2 > abs(dx_right) = 1: left edge has crossed
    xleft = 3.
    xright = 1.

    left_pbc_info = check_and_update_periodic_info(pbc_info, 10, xleft, xright)

    assert (left_pbc_info.pbc_multiplier_left != 0)
    assert (left_pbc_info.pbc_multiplier_right == 0)

    # dx_left = 1 < abs(dx_right) = 1.5: right edge has crossed
    xleft = 2.
    xright = 0.5

    right_pbc_info = check_and_update_periodic_info(pbc_info, 10, xleft, xright)

    assert (right_pbc_info.pbc_multiplier_left == 0)
    assert (right_pbc_info.pbc_multiplier_right != 0)

def test_periodic_info_check_updates_the_multiplier_in_the_right_direction():
    box_x = 10.
    xleft_prev = 1.
    xright_prev = 9.
    pbc_info = init_pbc_info_with_positions(xleft_prev, xright_prev)

    # The right edge crosses the right boundary
    xleft = 2.
    xright = 1.
    crossed_right_pbc_info = check_and_update_periodic_info(
        pbc_info, box_x, xleft, xright
    )

    assert (crossed_right_pbc_info.pbc_multiplier_left \
        == pbc_info.pbc_multiplier_left)
    assert (crossed_right_pbc_info.pbc_multiplier_right \
        == pbc_info.pbc_multiplier_right + 1)

    # The left edge crosses the left boundary
    xleft = 9.
    xright = 8.
    crossed_left_pbc_info = check_and_update_periodic_info(
        pbc_info, box_x, xleft, xright
    )

    assert (crossed_left_pbc_info.pbc_multiplier_left \
        == pbc_info.pbc_multiplier_left - 1)
    assert (crossed_left_pbc_info.pbc_multiplier_right \
        == pbc_info.pbc_multiplier_right)

def test_periodic_info_check_accounts_for_the_current_pbc_multipliers():
    box_x = 10.

    # The right edge is further along in x when accounting for
    # the pbc multiplier
    xleft_prev = 9.
    xright_prev = 1.
    pbc_multiplier_left = 1
    pbc_multiplier_right = 2
    pbc_info = init_pbc_info_with_positions_and_multipliers(
        xleft_prev, xright_prev, pbc_multiplier_left, pbc_multiplier_right
    )

    # The left edge crossed the right edge
    xleft_current = 1.
    xright_current = 3.

    crossed_right_pbc_info = check_and_update_periodic_info(
        pbc_info, box_x, xleft_current, xright_current
    )

    assert (crossed_right_pbc_info.pbc_multiplier_left \
        == pbc_info.pbc_multiplier_left + 1)
    assert (crossed_right_pbc_info.pbc_multiplier_right \
        == pbc_info.pbc_multiplier_right)

    # The right edge crossed the left edge
    xleft_current = 8.
    xright_current = 9.

    crossed_left_pbc_info = check_and_update_periodic_info(
        pbc_info, box_x, xleft_current, xright_current
    )

    assert (crossed_left_pbc_info.pbc_multiplier_left \
        == pbc_info.pbc_multiplier_left)
    assert (crossed_left_pbc_info.pbc_multiplier_right \
        == pbc_info.pbc_multiplier_right - 1)

def test_periodic_info_adds_on_right_if_initially_we_are_crossed():
    pbc_info = init_periodic_info()

    xleft = 5.
    xright = 3.

    updated_pbc_info = check_and_update_periodic_info(
        pbc_info, 10., xleft, xright
    )

    assert (updated_pbc_info.pbc_multiplier_left == 0)
    assert (updated_pbc_info.pbc_multiplier_right == 1)
    assert (updated_pbc_info.xleft_prev == 5.)
    assert (updated_pbc_info.xright_prev == 3.)

def test_add_periodic_multiplier_to_coords():
    pbc_multiplier_left = -2
    pbc_multiplier_right = 3

    pbc_info = init_pbc_info_with_positions_and_multipliers(
        None, None, pbc_multiplier_left, pbc_multiplier_right
    )

    box_x = 7.
    xleft_relative = 3.
    xright_relative = 5.

    xleft, xright = add_pbc_multipliers_to_edges(
        pbc_info, box_x, xleft_relative, xright_relative
    )

    assert (xleft == xleft_relative + pbc_multiplier_left * box_x)
    assert (xright == xright_relative + pbc_multiplier_right * box_x)

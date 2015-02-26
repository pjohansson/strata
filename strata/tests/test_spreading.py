import numpy as np
import os
import pytest
import tempfile as tmp

from droplets.flow import FlowData
from strata.dataformats.write import write
from strata.collect import spreading_collect, get_spreading_edges
from strata.utils import gen_filenames

datasize = 11

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
        num_bins = 2
        cutoff = 2

        kwargs = {
                'coord_labels': clabels,
                'cutoff': cutoff,
                'num_bins': num_bins
                }

        flow = FlowData((clabels[0], xs), (clabels[1], ys), (label, cs))

        left, right = get_spreading_edges(flow, label, radius, **kwargs)
        assert (left == xs[xmin] and right == xs[xmax])

def test_spreading():
    xs = np.linspace(0, 1, datasize)
    ys = np.zeros(datasize) + 1.5
    ms = np.zeros(datasize)
    other = np.zeros(datasize)
    icenter = np.floor(datasize/2)

    dt = 10

    with tmp.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, 'temp_')
        fngen = gen_filenames(base)

        # Write spreading data
        time = 0.

        radii = []
        times = []
        for param in (None, 0, 1, 2, 3):
            fnout = next(fngen)

            if param != None and param > 0:
                ms[(icenter-param):(icenter+param+1)] = 1.
                xmin, xmax = min(xs[ms == 1.]), max(xs[ms == 1.])
                radius = 0.5*(xmax - xmin)

                times.append(time)
                radii.append(radius)
                time += dt

            data = {'X': xs, 'Y': ys, 'M': ms}
            for label in ('N', 'T', 'U', 'V'):
                data[label] = other

            write(fnout, data)

        dtype = [('t', 'float32'), ('r', 'float32')]
        control_array = np.zeros(len(times), dtype=dtype)
        control_array['t'] = times
        control_array['r'] = radii

        # Collect spreading from files
        output = os.path.join(tmpdir, 'spread.xvg')
        spreading_data = spreading_collect(base, output=output, dt=dt,
                floor=1.5, include_radius=1)
        assert (np.allclose(spreading_data['r'], control_array['r']))
        assert (np.allclose(spreading_data['t'], control_array['t']))

        # Verify optionally written file
        spreading_read = np.loadtxt(output)
        assert (np.allclose(spreading_read[:,0], spreading_data['t'], atol=1e-3))
        assert (np.allclose(spreading_read[:,1], spreading_data['r'], atol=1e-3))

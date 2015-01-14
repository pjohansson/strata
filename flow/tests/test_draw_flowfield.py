import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from flow.data_utils import get_lim_indices
from flow.flow import FlowData
from flow.flowfield import *

def test_draw_flowfield():
    X = np.arange(9)
    Y = np.arange(5)

    xs, ys = np.meshgrid(X, Y)
    us = np.sin(xs*ys)*5
    vs = np.cos(xs)*10 - np.exp(0.1*ys)

    flow = FlowData({'X': xs, 'Y': ys, 'U': us, 'V': vs})

    # Verify default drawing
    fig = draw_flowfield(flow.data)
    assert (type(fig) == matplotlib.quiver.Quiver)

    # Verify drawing just U, V
    fig = draw_flowfield(flow.data, fields=('U', 'V'))
    assert (type(fig) == matplotlib.quiver.Quiver)

    # Verify that drawing with limits set works
    lims = {'X': (2, 6), 'Y': (None, 3), 'U': (None, 1.5)}
    indices = get_lim_indices(flow.data, lims)
    fig = draw_flowfield(flow.data[indices])
    assert (type(fig) == matplotlib.quiver.Quiver)

def test_draw_flowfield_general_names():
    X = np.arange(9)
    Y = np.arange(5)

    xs, ys = np.meshgrid(X, Y)
    us = np.sin(xs*ys)*5
    vs = np.cos(xs)*10 - np.exp(0.1*ys)
    ms = us + vs

    flow = FlowData({'f0': xs, 'f1': ys, 'f2': us, 'f3': vs, 'f4': ms})

    fig = draw_flowfield(flow.data, fields=('f0', 'f1', 'f2', 'f3', 'f4'))
    assert (type(fig) == matplotlib.quiver.Quiver)

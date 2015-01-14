import numpy as np
import pytest
from flow.flow import FlowData
from flow.droplet import *

# Create grid
x = np.arange(11)
y = np.arange(11)
xs, ys = np.meshgrid(x, y)
info = {'shape': (11,11)}

# Create droplet centered at (5,5)
mid, spread = 5, 10
cs = np.exp(-((xs-mid)**2 + (ys-mid)**2)/spread)

def test_find_interface():
    flow = FlowData(('X', xs), ('Y', ys), ('C', cs), info=info)
    interface = get_droplet_interface(flow, 'C')
    assert (np.shape(interface) == (21,2))

def test_find_interface_cutoffs():
    flow = FlowData(('X', xs), ('Y', ys), ('C', cs), info=info)

    interface = get_droplet_interface(flow, 'C', cutoff=0.8)
    assert (np.shape(interface) == (13,2))

    interface = get_droplet_interface(flow, 'C', cutoff=5)
    assert (np.shape(interface) == (0,))

    with pytest.raises(AttributeError):
        interface = get_droplet_interface(flow, 'C', cutoff=-1)

def test_find_interface_general_names():
    flow = FlowData(('f0', xs), ('f1', ys), ('f2', cs), info=info)
    labels = ('f0', 'f1')
    interface = get_droplet_interface(flow, 'f2', None, coord_labels=labels)
    assert (np.shape(interface) == (21,2))

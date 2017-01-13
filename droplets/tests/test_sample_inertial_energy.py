import numpy as np

from droplets.flow import FlowData
from droplets.sample import sample_inertial_energy


def test_calc_inertial_energy():
    # Sample on 3x3 grid
    x = np.arange(3)
    y = 0.5*x

    xs, ys = np.meshgrid(x, y)
    us = np.random.sample((3, 3))
    vs = np.random.sample((3, 3))
    ms = np.random.sample((3, 3))

    flow = FlowData(('X', xs), ('Y', ys), ('U', us), ('V', vs), ('M', ms))

    energy = (0.5*ms*(us**2 + vs**2)).reshape(9, )

    assert np.isclose(energy, sample_inertial_energy(flow)).all()


def test_calc_inertial_energy_other_labels():
    x = np.arange(3)
    xs, ys = np.meshgrid(x, x)
    us = np.random.sample((3, 3))
    vs = np.random.sample((3, 3))
    ms = np.random.sample((3, 3))

    flow = FlowData(('c0', xs), ('c1', ys), ('f0', us), ('f1', vs), ('v0', ms))

    energy = (0.5*ms*(us**2 + vs**2)).reshape(9, )

    assert np.isclose(energy, sample_inertial_energy(flow, flow_labels=('f0', 'f1'),
                                                           mass_label='v0')).all()

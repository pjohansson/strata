import numpy as np

from droplets.flow import FlowData
from droplets.sample import sample_viscous_dissipation


def test_calc_viscous_dissipation():
    # Sample on 3x3 grid
    x = np.arange(3)
    y = 0.5*x

    xs, ys = np.meshgrid(x, y)
    us = np.random.sample((3, 3))
    vs = np.random.sample((3, 3))

    viscosity = 2.
    info = {
        'spacing': (1.0, 0.5), # Non-quadratic bins
        'shape': (3, 3)
        }

    flow = FlowData(('X', xs), ('Y', ys), ('U', us), ('V', vs), info=info)
    result = sample_viscous_dissipation(flow, viscosity)

    # Control dissipation in edge cell (1, 2) (center-bottom)
    # Edge cell gradients (dy terms) are second order
    dudx = (us[2, 2] - us[2, 0])/2
    dudy = np.gradient(us[:, 1], 0.5, edge_order=2)[2]
    dvdx = (vs[2, 2] - vs[2, 0])/2
    dvdy = np.gradient(vs[:, 1], 0.5, edge_order=2)[2]

    visc_diss = 2*viscosity*(dudx**2 + dvdy**2 - (dudx + dvdy)**2/3.0) \
            + viscosity*(dvdx + dudy)**2

    assert np.isclose(visc_diss, result[2, 1])


def test_calc_viscous_dissipation_other_labels():
    x = np.arange(3)
    xs, ys = np.meshgrid(x, x)
    us = np.random.sample((3, 3))
    vs = np.random.sample((3, 3))

    viscosity = 2.
    info = {
        'spacing': (1., 1.), # Non-quadratic bins
        'shape': (3, 3)
        }

    flow = FlowData(('c0', xs), ('c1', ys), ('f0', us), ('f1', vs), info=info)
    result = sample_viscous_dissipation(flow, viscosity,
                                        coord_labels=('c0', 'c1'),
                                        flow_labels=('f0', 'f1'))

    # Control all cells
    dudy, dudx = np.gradient(us, edge_order=2)
    dvdy, dvdx = np.gradient(vs, edge_order=2)

    visc_diss = 2*viscosity*(dudx**2 + dvdy**2 - (dudx + dvdy)**2/3.0) \
            + viscosity*(dvdx + dudy)**2

    assert np.isclose(visc_diss, result).all()


def test_calc_viscous_dissipation_unsorted_bins():
    x = np.arange(3)

    xs, ys = np.meshgrid(x, x)
    us = np.random.sample(xs.shape)
    vs = np.random.sample(ys.shape)

    viscosity = 2.
    info = {
        'spacing': (1., 1.),
        'shape': (3, 3)
        }

    flow = FlowData(('X', xs), ('Y', ys), ('U', us), ('V', vs), info=info)

    # Change sorting of bins to XY instead of the default YX
    # This will mess up the default calculation which assumes a certain order
    flow.data.sort(order=('X', 'Y'))
    result = sample_viscous_dissipation(flow, viscosity)

    # Control all cells against the default sorting
    # This ensures that the function properly sorts the data to match coordinates
    dudy, dudx = np.gradient(us, edge_order=2)
    dvdy, dvdx = np.gradient(vs, edge_order=2)

    visc_diss = 2*viscosity*(dudx**2 + dvdy**2 - (dudx + dvdy)**2/3.0) \
            + viscosity*(dvdx + dudy)**2

    assert np.isclose(visc_diss, result).all()


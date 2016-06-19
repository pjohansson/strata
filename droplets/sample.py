import numpy as np

"""Tools for sampling data from FlowData objects."""


def sample_per_angle_from(flow, origin, label,
        coord_labels=('X', 'Y'), weight=None,
        amin=0., amax=360.,
        rmin=0., rmax=np.inf,
        size=1.):
    """Sample data per angle from an input point.

    The angle is calculated starting from the positive x-axis (i.e.
    point (1, 0)) and going counter-clockwise in a full circle. The
    result has a precision of 1 degree and the data is binned for
    the half-open angular bin intervals [angle, angle+size) where
    size is the angular size of the bins. The data within the selected
    bins for each angular interval is averaged for the resulting value.
    Optionally the data can be weighed by data from another label
    input with the keyword `weight`.

    The angles to sample over can be chosen by using the keyword
    arguments `amin` and `amax`.

    Args:
        flow (FlowData): A FlowData object to sample from.

        origin (float, float): A point to sample around as a 2-tuple.

        label (str): Data label to sample values from.

    Keyword args:
        coord_labels (2-tuple, default=('X', 'Y'): Record labels for coordinates.

        amin/amax (float): Minimum and maximum angles to sample data over.
            By default samples the entire circular range.

        rmin/rmax (float): Minimum and maximum radius from point to sample
            data within. By default uses the entire data set.

        size (float): Size in degrees of the angular intervals.

        weight (str): Weigh the data of the input label by data
            from this label when averaging bins.

    Returns:
        ndarray: Numpy record with angles (degrees) in label 'angle'
            and the data in the used input label.

    """

    amin = max(0., amin)
    amax = min(360.-size, amax)
    out_angles = np.arange(amin, amax+size, size)

    # Collect values and weights in bins to average for each angle
    out_values_bins = [[] for _ in out_angles]
    weights_bins = [[] for _ in out_angles]

    dxs, dys = [flow.data[l] - o for l, o in zip(coord_labels, origin)]
    drs = np.sqrt(dxs**2 + dys**2)

    # Apply radius cutoffs and calculate angles for bins
    inds_rlim = (drs >= rmin) & (drs <= rmax)
    data_rlim = flow.data[inds_rlim]

    bin_rads = np.sign(dys[inds_rlim])*np.arccos(dxs[inds_rlim]/drs[inds_rlim])
    bin_degrees = np.mod(np.degrees(bin_rads), 360)

    # Apply angular cutoffs to remaining bins
    inds_alim = (bin_degrees >= amin) & (bin_degrees <= amax)
    data_alim = data_rlim[inds_alim]
    label_data = data_alim[label]

    # Get corresponding weighting data or 1.0 for no weighting
    if weight != None:
        weight_data = data_alim[weight]
    else:
        weight_data = [1.0 for _ in label_data]

    # Adjust angles by 1/2 step to center bins around the angle measurement
    # points, they will histogram correctly for the half-open interval
    # [angle, angle+step)
    in_angles = bin_degrees[inds_alim] - 0.5*size + 1e-6

    # Add data to bins
    for angle, value, optweight in zip(in_angles, label_data, weight_data):
        iangle = np.abs(out_angles - angle).argmin()
        out_values_bins[iangle].append(value)
        weights_bins[iangle].append(optweight)

    # Construct result array
    dtype = [(l, np.float) for l in ('angle', label)]
    result = np.zeros(out_angles.shape, dtype=dtype)

    # Add angles and the mean of values for each
    result['angle'] = out_angles
    result[label] = np.array(
            [np.average(vals, weights=weights) if vals != [] else 0.
             for vals, weights in zip(out_values_bins, weights_bins)])

    return result


def sample_viscous_dissipation(flow, viscosity,
        coord_labels=('X', 'Y'), flow_labels=('U', 'V')):
    """Return array of viscous dissipation from input flow data.

    Args:
        flow (FlowData): Object to calculate dissipation from. Must
            contain fields for X-Y coordinates and flow.

        viscosity (float): Viscosity of liquid.

    Keyword args:
        coord_labels (2-tuple, default=('X', 'Y'): Record labels for coordinates.

        flow_labels (2-tuple, default=('U', 'V'): Record labels for flow.

    """

    dx, dy = flow.spacing
    nx, ny = flow.shape

    coord_order = list(reversed(coord_labels))
    data = np.sort(flow.data, order=coord_order).reshape(ny, nx)
    U, V = [data[l] for l in flow_labels]

    dudy, dudx = np.gradient(U, dy, dx, edge_order=2)
    dvdy, dvdx = np.gradient(V, dy, dx, edge_order=2)

    return 2*viscosity*(dudx**2 + dvdy**2 - (dudx + dvdy)**2/3.0) \
            + viscosity*(dvdx + dudy)**2


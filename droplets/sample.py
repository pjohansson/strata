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

    # If weights sum to 0 the result of that angle should be 0.
    # Also ensure that there are values to average
    result[label] = np.array(
            [np.average(vals, weights=weights)
             if vals != [] and np.sum(weights) > 0. else 0.
             for vals, weights in zip(out_values_bins, weights_bins)
             ])

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

    try:
        nx, ny = [int(v) for v in flow.shape]
    except:
        raise ValueError(
                "the shape of the flow object is not correctly set "
                "(is {!r}, expected (int, int))".format(flow.shape)
            )

    try:
        dx, dy = [float(v) for v in flow.spacing]
    except:
        raise ValueError(
                "the spacing of the flow object is not correctly set"
                "(is {!r}, expected (float, float))".format(flow.spacing)
            )

    coord_order = list(reversed(coord_labels))
    data = np.sort(flow.data, order=coord_order).reshape(ny, nx)
    U, V = [data[l] for l in flow_labels]

    dudy, dudx = np.gradient(U, dy, dx, edge_order=2)
    dvdy, dvdx = np.gradient(V, dy, dx, edge_order=2)

    return 2 * viscosity * (dudx**2 + dvdy**2 - (dudx + dvdy)**2/3.0) \
            + viscosity * (dvdx + dudy)**2


def sample_center_of_mass(flow, mass_label='M', coord_labels=['X', 'Y']):
    """Returns the center of mass along x and y as a tuple.

    Args:
        flow (FlowData): Object to calculate center of mass of. Must
            contain fields for X-Y coordinates and mass.

    Keyword args:
        mass_label (str, default='M'): Label for mass of system.

        coord_labels (2-tuple, default=('X', 'Y'): Record labels for coordinates.

    Raises:
        ZeroDivisionError: If the mass of the system sums to zero.

    """

    return tuple(np.average(flow.data[l], weights=flow.data[mass_label])
                 for l in coord_labels)


def sample_inertial_energy(flow, mass_label='M', flow_labels=['U', 'V']):
    """Returns the total inertial energy of the system.

    Args:
        flow (FlowData): Object to calculate center of mass of. Must
            contain fields for flow along X-Y and mass.

    Keyword args:
        mass_label (str, default='M'): Label for mass of system.

        flow_labels (2-tuple, default=('U', 'V'): Record labels for flow.

    """

    ul, vl = flow_labels
    return 0.5*flow.data[mass_label]*(flow.data[ul]**2 + flow.data[vl]**2)


def sample_flow_angle(flow, flow_labels=['U', 'V'], mean=False, weight=None):
    """Returns the angle of the flow of all bins in the system in degrees.

    Optionally sample the mean angle by supplying the `mean` keyword. For
    this operation the bins can be weighed by other values by supplying
    the keyword `weight`.

    The angles are returned on the interval (-180, +180) degrees where
    the positive values correspond to a counter-clockwise direction.

    Args:
        flow (FlowData): Object to calculate angles from. Must contain fields
            flow along X-Y.

    Keyword args:
        flow_labels (2-tuple, default=('U', 'V'): Record labels for flow.

        mean (bool, default=False): Sample the mean angle.

        weight (str, default=None): Label to weigh the flow by when
            calculating the mean.

    Returns:
        ndarray: Angles in degrees, or a single angle if `mean` is True.

    """

    try:
        us, vs = [flow.data[label].copy() for label in flow_labels]
    except ValueError as exc:
        if "no field" in str(exc):
            raise ValueError("Flow labels %r were not found in the system"
                % flow_labels)
        else:
            raise ValueError("Exactly 2 flow labels must be input, but got %d (%r)"
                % (len(flow_labels), flow_labels))

    if mean:
        if weight:
            try:
                us *= flow.data[weight]
                vs *= flow.data[weight]
            except:
                raise ValueError("Weight label %r was not found in the system" % weight)

        us = us.sum()
        vs = vs.sum()

    angles = np.arctan2(vs, us)

    return np.degrees(angles)

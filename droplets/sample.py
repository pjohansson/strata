import numpy as np

"""Tools for sampling data from FlowData objects."""


def sample_per_angle_from(flow, origin, label,
        coord_labels=('X', 'Y'),
        amin=0., amax=360.,
        rmin=0., rmax=np.inf,
        size=1.):
    """Sample data per angle from an input point.

    The angle is calculated starting from the positive x-axis (i.e.
    point (1, 0)) and going counter-clockwise in a full circle. The
    result has a precision of 1 degree and the data is binned for
    the half-open angular bin intervals [angle, angle+size) where
    size is the angular size of the bins.

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

    Returns:
        ndarray: Numpy record with angles (degrees) in label 'angle'
            and the data in the used input label.

    """

    amin = max(0., amin)
    amax = min(360.-size, amax)
    out_angles = np.arange(amin, amax+size, size)

    # Collect values in bins to average for each angle
    out_values_bins = [[] for _ in out_angles]

    dxs, dys = [flow.data[l] - o for l, o in zip(coord_labels, origin)]
    drs = np.sqrt(dxs**2 + dys**2)

    # Apply radius cutoffs and calculate angles for bins
    inds_rlim = (drs >= rmin) & (drs <= rmax)
    data_rlim = flow.data[label][inds_rlim]

    bin_rads = np.sign(dys[inds_rlim])*np.arccos(dxs[inds_rlim]/drs[inds_rlim])
    bin_degrees = np.mod(np.degrees(bin_rads), 360)

    # Apply angular cutoffs to remaining bins
    inds_alim = (bin_degrees >= amin) & (bin_degrees <= amax)
    data_alim = data_rlim[inds_alim]

    # Adjust angles by 1/2 step to center bins around the angle measurement
    # points, they will histogram correctly for the half-open interval
    # [angle, angle+step)
    in_angles = bin_degrees[inds_alim] - 0.5*size + 1e-6

    # Add data to bins
    for angle, value in zip(in_angles, data_alim):
        iangle = np.abs(out_angles - angle).argmin()
        out_values_bins[iangle].append(value)

    # Construct result array
    dtype = [(l, np.float) for l in ('angle', label)]
    result = np.zeros(out_angles.shape, dtype=dtype)

    # Add angles and the mean of values for each
    result['angle'] = out_angles
    result[label] = np.array(
            [np.mean(vals) if vals != [] else 0.
             for vals in out_values_bins])

    return result

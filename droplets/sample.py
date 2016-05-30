import numpy as np

"""Tools for sampling data from FlowData objects."""


def sample_per_angle_from(flow, origin, label,
        coord_labels=('X', 'Y'),
        amin=0., amax=360.,
        rmin=0., rmax=np.inf):
    """Sample data per angle from an input point.

    The angle is calculated starting from the positive x-axis (i.e.
    point (1, 0)) and going counter-clockwise in a full circle. The
    result has a precision of 1 degree and the data is binned to the
    closest angle of those returned.

    The angles to sample over can be chosen by using the keyword
    arguments `amin` and `amax`. Do note that since the measured angles
    are binned as their closest bins, the angle bins at the edges will
    be cut off as half sized bins and hence undersampled. This should
    be improved.

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

    Returns:
        ndarray: Numpy record with angles (degrees) in label 'angle'
            and the data in the used input label.

    """

    step = 1.
    amin = max(0., amin)
    amax = min(360.-step, amax)
    out_angles = np.arange(amin, amax+step, step)

    # Collect values in bins to average for each angle
    out_values_bins = [[] for _ in out_angles]

    dxs, dys = [flow.data[l] - o for l, o in zip(coord_labels, origin)]
    drs = np.sqrt(dxs**2 + dys**2)

    # Extract data within radius slice
    inds = (drs >= rmin) & (drs <= rmax)
    data = flow.data[label][inds]

    measured_rads = np.sign(dys[inds])*np.arccos(dxs[inds]/drs[inds])
    measured_angles = np.mod(np.degrees(measured_rads), 360)

    # Add data to bins
    for angle, value in zip(measured_angles, data):
        if angle >= amin and angle <= amax:
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

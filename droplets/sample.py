import numpy as np


def sample_per_angle_from(flow, origin, label,
        coord_labels=('X', 'Y'), rmin=0., rmax=np.inf):
    """Sample data per angle from an input point.

    Args:
        flow (FlowData): A FlowData object to sample from.

        origin (float, float): A point to sample around as a 2-tuple.

        label (str): Data label to sample values from.

    Keyword args:
        coord_labels (2-tuple, default=('X', 'Y'): Record labels for coordinates.

        rmin/rmax (float): Minimum and maximum radius from point to sample
            data within. By default uses the entire data set.

    Returns:
        ndarray: Numpy record with angles in label 'angle' and the data
            in the used input label.

    """

    # Collect values in bins for each angle
    out_angles = np.arange(360)
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

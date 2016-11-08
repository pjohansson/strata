import numpy as np

def average_data(*data, atol=1e-3, rtol=1e-05):
    """Return a sample average of several plain maps.

    Note that flow ('U', 'V') are mass averaged and that the temperature
    ('T') is number averaged.

    Relative and absolute tolerances are used to ascertain that the input
    map coordinates are identical for each map. These can be controlled by
    the keyword arguments 'atol' and 'rtol'.

    Args:
        data (dict): List of dict's with data read from a simple data map,
            with fields ('X', 'Y', 'M', 'N', 'T', 'U', 'V').

    Keyword Args:
        atol (float): Absolute tolerance for the coordinate check.

        rtol (float): Relative tolerance for the coordinate check.

    Returns:
        dict: An averaged record. Empty if no data was input.

    Raises:
        ValueError: If coordinates of all input data are not identical.

    """

    def init_with_coords(data):
        coords = {l: data[0][l] for l in ('X', 'Y')}
        for d in data[1:]:
            for l in ('X', 'Y'):
                # Check that input coordinates of all maps match each other
                # An absolute tolerance value is used but might not be the
                # best solution, better might be for the caller to assert
                # that the coordinates match sufficiently before trying
                # to average the maps. Then again floating point comparisons
                # are what they are so it's probably best to keep it.
                assert (np.isclose(coords[l], d[l], atol=atol, rtol=rtol).all())
        return coords

    def get_sum_weights():
        weights = ('N', 'M')
        return {w: np.sum([d[w] for d in data], 0) for w in weights}

    def get_avg(field):
        return np.mean([d[field] for d in data], 0)

    def get_weighted_avg(field, weight):
        weighted_sum = np.sum([d[field]*d[weight] for d in data], 0)
        weighted_avg = weighted_sum/sum_weights[weight]
        return np.nan_to_num(weighted_avg)

    if list(data) == []:
        return {}

    try:
        avg_data = init_with_coords(data)
    except AssertionError:
        raise ValueError("coordinates of data to average does not match for all maps")

    sum_weights = get_sum_weights()

    avg_data['M'] = get_avg('M')
    avg_data['N'] = get_avg('N')
    avg_data['U'] = get_weighted_avg('U', 'M')
    avg_data['V'] = get_weighted_avg('V', 'M')
    avg_data['T'] = get_weighted_avg('T', 'N')

    return avg_data

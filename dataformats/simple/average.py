import numpy as np

def average_simple_data(*data):
    """Return a sample average of several plain maps.

    Note that flow ('U', 'V') are mass averaged and that the temperature
    ('T') is number averaged.

    Args:
        data (dict): List of dict's with data read from a simple data map,
            with fields ('X', 'Y', 'M', 'N', 'T', 'U', 'V').

    Returns:
        dict: An averaged record. Empty if no data was input.

    Raises:
        ValueError: If coordinates of all input data are not identical.

    """

    def init_with_coords(data):
        coords = {l: data[0][l] for l in ('X', 'Y')}
        for d in data[1:]:
            for l in ('X', 'Y'):
                assert (np.array_equal(coords[l], d[l]))
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
        raise ValueError("coordinates of data do not match")
    sum_weights = get_sum_weights()

    avg_data['M'] = get_avg('M')
    avg_data['N'] = get_avg('N')
    avg_data['U'] = get_weighted_avg('U', 'M')
    avg_data['V'] = get_weighted_avg('V', 'M')
    avg_data['T'] = get_weighted_avg('T', 'N')

    return avg_data

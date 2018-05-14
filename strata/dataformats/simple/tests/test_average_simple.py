import numpy as np
import pytest
from strata.dataformats.simple.average import *
from strata.dataformats.simple.main import read_data

data_fields = ('U', 'V', 'M', 'T', 'N')
all_fields = data_fields + ('X', 'Y')


def test_average_data():
    def run_test_with_settings(datasize, numdata):
        xs = np.arange(datasize)
        ys = np.arange(datasize)
        coords = {'X': xs, 'Y': ys}

        # Create random data for records
        data = []
        for _ in np.arange(numdata):
            d = coords
            for label in data_fields:
                d[label] = np.random.sample(datasize)
            data.append(d)

        # Calculate control values
        control = coords
        Nsum, Msum = [np.sum([d[label] for d in data], 0) for label in ('N', 'M')]
        control['U'] = np.sum([d['U']*d['M'] for d in data], 0)/Msum
        control['V'] = np.sum([d['V']*d['M'] for d in data], 0)/Msum
        control['T'] = np.sum([d['T']*d['N'] for d in data], 0)/Nsum
        control['M'] = np.mean([d['M'] for d in data], 0)
        control['N'] = np.mean([d['N'] for d in data], 0)

        # Verify averaging function
        avg_data = average_data(*data)
        for label in all_fields:
            assert (np.allclose(avg_data[label], control[label]))

    run_test_with_settings(datasize=4, numdata=1)
    run_test_with_settings(datasize=8, numdata=1e3)
    run_test_with_settings(datasize=int(1e5), numdata=2)


def test_avg_different_sizes():
    data = []
    for _ in np.arange(7):
        d = {l: np.arange(8) for l in ('X', 'Y')}
        for label in data_fields:
            d[label] = np.random.sample(8)
        data.append(d)

    # Append data with different size
    d = {l: np.arange(4) for l in ('X', 'Y')}
    for label in data_fields:
        d[label] = np.random.sample(4)
    data.append(d)

    with pytest.raises(ValueError):
        average_data(*data)


def test_avg_empty():
    # Assert that no data returns empty dict
    assert (average_data() == {})

    # Assert that empty data works
    data = {}
    for label in all_fields:
        data[label] = np.array([])

    avg_data = average_data(data)
    for label in all_fields:
        assert (np.array_equal(avg_data[label], data[label]))


def test_fields_nan():
    xs = np.arange(8)
    ys = np.arange(8)

    data = []

    # T is averaged with N, so assert that if N = 0,
    # so is T
    for _ in np.arange(4):
        d = {'X': xs, 'Y': ys}

        for label in data_fields:
            d[label] = np.random.sample(8)

        data.append(d)

    # Get reference data
    avg_data_ref = average_data(*data)

    # Set all N values to 0 and get the new values
    for d in data:
        d['N'] *= 0

    # Assert that no warning is raised
    with pytest.warns(None) as record:
        avg_data_nan = average_data(*data)
    assert (len(record.list) == 0)

    # The T (and N) fields should be averaged to zero
    assert (np.array_equiv(avg_data_nan['T'], 0.))
    assert (np.array_equiv(avg_data_nan['N'], 0.))

    # All non-N or T fields should be as before
    fields = list(data_fields)
    fields.remove('N')
    fields.remove('T')

    for label in fields:
        assert (np.array_equal(avg_data_ref[label], avg_data_nan[label]))


def test_read_and_average():
    filename = 'strata/dataformats/simple/tests/data_avgsimple.dat'
    data1, _ = read_data(filename)
    data2 = {l: v.copy() for l, v in data1.items()}

    # Triple flow velocities and temperatures of copy
    data2['U'] *= 3
    data2['V'] *= 3
    data2['T'] *= 3

    avg_data = average_data(data1, data2)
    for label in ('U', 'V', 'T'):
        assert (np.allclose(np.sum(avg_data[label]), 2*np.sum(data1[label])))

    # Triple mass and numbers separately since they affect others non-trivially
    data2['M'] *= 3
    data2['N'] *= 3

    avg_data = average_data(data1, data2)
    for label in ('M', 'N'):
        assert (np.allclose(np.sum(avg_data[label]), 2*np.sum(data1[label])))

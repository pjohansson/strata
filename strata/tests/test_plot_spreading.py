import numpy as np
import os
import pytest
import tempfile as tmp

from strata.view import *

fndir = os.path.dirname(os.path.realpath(__file__))
fndata1 = os.path.join(fndir, 'spread-01.xvg')
fndata2 = os.path.join(fndir, 'spread-02.xvg')

data1 = np.zeros(4, dtype=[('t', 'float'), ('d1', 'float'), ('d2', 'float')])
data1['t'] = np.arange(0, 40, 10)
data1['d1'] = np.arange(1, 5)
data1['d2'] = np.arange(2, 6)

data2 = np.zeros(4, dtype=[('t', 'float'), ('d1', 'float')])
data2['t'] = np.arange(100, 140, 10)
data2['d1'] = np.arange(3, 7)

def test_read_files():
    data = read_spreading_data(fndata1, fndata2)

    assert (len(data) == 3)
    assert all(type(d) == pd.Series for d in data)

    assert (data[0].name == ('%s.1' % fndata1))
    assert (np.array_equal(data[0].index, data1['t']))
    assert (np.array_equal(data[0], data1['d1']))

    assert (data[1].name == ('%s.2' % fndata1))
    assert (np.array_equal(data[1].index, data1['t']))
    assert (np.array_equal(data[1], data1['d2']))

    assert (data[2].name == ('%s.1' % fndata2))
    assert (np.array_equal(data[2].index, data2['t']))
    assert (np.array_equal(data[2], data2['d1']))

def test_catch_badfile():
    with pytest.raises(SyntaxError):
        read_spreading_data(tmp.NamedTemporaryFile())

def test_combine_data():
    data = read_spreading_data(fndata1, fndata2)
    df = combine_spreading_data(data)

    assert (np.array_equal(df.index, np.union1d(data[0].index, data[2].index)))
    for d in data:
        assert (df[d.name].name == d.name)
        assert (np.array_equal(df[d.name].dropna(), d))

def test_sync_time_at_radius():
    radius = 4.
    data = read_spreading_data(fndata1, fndata2)
    sync_data = sync_time_at_radius(data, radius)

    for test, control in zip(data, sync_data):
        assert (test.name == control.name)

    assert (np.array_equal(sync_data[0].index, data[0].index - 10))
    assert (np.array_equal(sync_data[1].index, data[1].index))
    assert (np.array_equal(sync_data[2].index, data[2].index - 90))

def test_write_xvg():
    fnout = 'tmpfile.xvg'
    data = read_spreading_data(fndata1, fndata2)
    df = combine_spreading_data(data)

    with tmp.TemporaryDirectory() as tmpdir:
        fntmp = os.path.join(tmpdir, fnout)
        write_spreading_data(fntmp, df)

        new_data = read_spreading_data(fntmp)

        assert (len(new_data) == len(data))
        assert all(len(d) == len(df) for d in new_data)

        for i, control in enumerate(df):
            ind_nan = df[control][df[control].isnull()].index
            ind_0 = new_data[i][new_data[i] == 0.].index
            assert np.array_equal(ind_0, ind_nan)

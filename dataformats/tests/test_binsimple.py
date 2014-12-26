from dataformats.simple import *

MAX_ERR = 1e-5
FIELDS = set(['X', 'Y', 'U', 'V', 'N', 'T', 'M'])

def test_bin_goodfile():
    filename = 'dataformats/tests/data_binsimple.dat'
    data = read_binsimple(filename)
    assert (data.keys() == FIELDS)

    # Control values
    NUM_BINS = 522720
    NUM_FULL_BINS = 127951
    MIN_X, MAX_X = 0.125, 296.875
    MIN_Y, MAX_Y = 0.125, 109.875
    AVG_MASS = 167.74342
    AVG_TEMP = 300.87469
    NUM_ATOMS = 3574098.75

    # Calculate number of full bins
    full_bins = np.nonzero(data['N'])
    num_full_bins = len(data['N'][full_bins])

    # Verify lengths of all read fields
    lengths = [len(field) for field in data.values()]
    assert (np.equal(lengths, NUM_BINS).all())
    assert (num_full_bins == NUM_FULL_BINS)

    # Verify control values
    assert (data['X'][0] == MIN_X)
    assert (data['X'][-1] == MAX_X)
    assert (data['Y'][0] == MIN_Y)
    assert (data['Y'][-1] == MAX_Y)
    assert (np.abs(np.mean(data['M'][full_bins]) - AVG_MASS) <= MAX_ERR)
    assert (np.abs(np.mean(data['T'][full_bins]) - AVG_TEMP) <= MAX_ERR)
    assert (np.abs(float(np.sum(data['N'])) - NUM_ATOMS) <= MAX_ERR)

def test_plain_goodfile():
    filename = 'dataformats/tests/data_plainsimple.dat'
    data = read_plainsimple(filename)
    assert (data.keys() == FIELDS)

    # Control values
    NUM_BINS = 12
    NUM_FULL_BINS = 9
    MIN_X, MAX_X = 133.125, 133.625
    MIN_Y, MAX_Y = 0.375, 1.125
    AVG_MASS = 117.66221
    AVG_TEMP = 298.64274
    NUM_ATOMS = 177.93100

    # Calculate number of full bins
    full_bins = np.nonzero(data['N'])
    num_full_bins = len(data['N'][full_bins])

    # Verify lengths of all read fields
    lengths = [len(field) for field in data.values()]
    assert (np.equal(lengths, NUM_BINS).all())
    assert (num_full_bins == NUM_FULL_BINS)

    # Verify control values
    assert (data['X'][0] == MIN_X)
    assert (data['X'][-1] == MAX_X)
    assert (data['Y'][0] == MIN_Y)
    assert (data['Y'][-1] == MAX_Y)
    assert (np.abs(np.mean(data['M'][full_bins]) - AVG_MASS) <= MAX_ERR)
    assert (np.abs(np.mean(data['T'][full_bins]) - AVG_TEMP) <= MAX_ERR)
    assert (np.abs(np.sum(data['N']) - NUM_ATOMS) <= MAX_ERR)

def test_calc_info():
    filename = 'dataformats/tests/data_binsimple.dat'
    data = read_binsimple(filename)
    info = calc_info(data['X'], data['Y'])

    # Control values
    NUM_BINS = 522720
    NUM_X, NUM_Y = 1188, 440
    MIN_X, MAX_X = 0.125, 296.875
    MIN_Y, MAX_Y = 0.125, 109.875
    BIN_X, BIN_Y = 0.25, 0.25

    # Verify control values
    assert (info['num_bins'] == NUM_BINS)
    assert (info['shape'] == [NUM_X, NUM_Y])
    assert (info['size']['X'] == [MIN_X, MAX_X])
    assert (info['size']['Y'] == [MIN_Y, MAX_Y])
    assert (info['bin_size'] == [BIN_X, BIN_Y])

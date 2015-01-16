import strata.dataformats as formats
from strata.dataformats.read import *

def test_guess_dataformat():
    bin_filename = 'strata/dataformats/simple/tests/data_binsimple.dat'
    plain_filename = 'strata/dataformats/simple/tests/data_plainsimple.dat'

    assert (guess_read_module(bin_filename) == formats.simple)
    assert (guess_read_module(plain_filename) == formats.simple)

def test_read_file():
    filename = 'strata/dataformats/simple/tests/data_plainsimple.dat'
    data, info = read_flow_data(filename)

    # Control values for simple file
    NUM_BINS = 12
    NUM_X, NUM_Y = 3, 4
    MIN_X, MAX_X = 133.125, 133.625
    MIN_Y, MAX_Y = 0.375, 1.125
    BIN_X, BIN_Y = 0.25, 0.25

    # Verify that plaintext file was read
    assert (info['num_bins'] == NUM_BINS)
    assert (info['shape'] == [NUM_X, NUM_Y])
    assert (info['size']['X'] == [MIN_X, MAX_X])
    assert (info['size']['Y'] == [MIN_Y, MAX_Y])
    assert (info['bin_size'] == [BIN_X, BIN_Y])

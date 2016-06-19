import strata.dataformats as formats
from strata.dataformats.read import *

bin_filename = 'strata/dataformats/simple/tests/data_binsimple.dat'
plain_filename = 'strata/dataformats/simple/tests/data_plainsimple.dat'

# Control values for simple file
NUM_BINS = 12
NUM_X, NUM_Y = 3, 4
MIN_X, MAX_X = 133.125, 133.625
MIN_Y, MAX_Y = 0.375, 1.125
BIN_X, BIN_Y = 0.25, 0.25

def test_guess_dataformat():
    assert (guess_read_module(bin_filename) == formats.simple.main)
    assert (guess_read_module(plain_filename) == formats.simple.main)

def test_read_file():
    _, info, _ = read_data_file(plain_filename)

    # Verify that plaintext file was read
    assert (info['num_bins'] == NUM_BINS)
    assert (info['shape'] == [NUM_X, NUM_Y])
    assert (info['origin'] == [MIN_X, MIN_Y])
    assert (info['spacing'] == [BIN_X, BIN_Y])

def test_read_files_generator():
    files = []
    for i, _ in enumerate(read_from_files(*files)):
        assert (False) # Shouldn't be reached

    files = [plain_filename]
    for i, (_, info, _) in enumerate(read_from_files(*files)):
        assert (info['num_bins'] == NUM_BINS)
        assert (info['shape'] == [NUM_X, NUM_Y])
        assert (info['origin'] == [MIN_X, MIN_Y])
        assert (info['spacing'] == [BIN_X, BIN_Y])

    files = [bin_filename, plain_filename]
    for i, (_, info, _) in enumerate(read_from_files(*files)):
        pass
    assert (i == 1)

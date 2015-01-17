import os
import numpy as np
import tempfile as tmp
from strata.dataformats.simple.main import write
from strata.dataformats.simple.read import read_plainsimple, read_binsimple

all_fields = ('X', 'Y', 'U', 'V', 'M', 'N', 'T')
tmpfile = 'tmpfile.dat'

datasize = 4
X = np.arange(datasize)
Y = np.arange(datasize)
data_save = {l: np.random.sample(datasize) for l in ('U', 'V', 'M', 'N', 'T')}
data_save.update({'X': X, 'Y': Y})

def test_write():
    with tmp.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, tmpfile)
        write(filename, data_save)
        data_read = read_binsimple(filename)

        for l in all_fields:
            assert (np.allclose(data_read[l], data_save[l]))

def test_write_plaintext():
    with tmp.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, tmpfile)
        write(filename, data_save, binary=False)
        data_read = read_plainsimple(filename)

        for l in all_fields:
            assert (np.allclose(data_read[l], data_save[l], atol=1e-6))

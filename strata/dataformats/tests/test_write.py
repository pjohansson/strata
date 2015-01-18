import os
import numpy as np
import pytest
import tempfile as tmp
import strata.dataformats as formats
from strata.dataformats.read import guess_read_module
from strata.dataformats.write import write
from strata.dataformats.simple.read import read_plainsimple, read_binsimple

datasize = 4
fields = ('U', 'V', 'N', 'T', 'M')

# Create data
save_data = {l: np.arange(datasize) for l in ('X', 'Y')}
for l in fields:
    save_data[l] = np.random.sample(datasize)

# Set defaults
tmpfn = 'tmp.dat'
default_module = formats.simple.main

def test_default_write():
    with tmp.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, tmpfn)
        write(path, save_data)
        assert (guess_read_module(path) == default_module)

def test_write_formats_simple():
    ftype_functions = (
            ('simple_plain', read_plainsimple),
            ('simple', read_binsimple)
            )

    for ftype, function in ftype_functions:
        with tmp.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, tmpfn)
            write(path, save_data, ftype=ftype)
            data = function(path)

            for l in save_data.keys():
                assert (np.allclose(data[l], save_data[l], atol=1e-6))

def test_bad_ftype():
    ftype = 'not-an-ftype'
    with tmp.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, tmpfn)
        with pytest.raises(KeyError):
            write(path, save_data, ftype=ftype)

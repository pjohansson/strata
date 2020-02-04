import numpy as np
import os
import tempfile as tmp
from strata.convert import convert
from strata.utils import gen_filenames, find_datamap_files
from strata.dataformats.read import *
from strata.dataformats.write import write
from strata.dataformats.simple.read import read_plainsimple, read_binsimple
from strata.dataformats.gmx_flow_version_1.read import read_data as read_gmx

num_files = 8
datasize = 4

x = np.arange(np.sqrt(datasize))
xs, ys = np.meshgrid(x, x, indexing='ij')
save_data = {
    'X': xs.ravel(),
    'Y': ys.ravel(),
}
fields = ('U', 'V', 'N', 'T', 'M')
all_fields = ('X', 'Y', 'U', 'V', 'N', 'T', 'M')

fnbase = 'tmp_'
out = 'out_'

def test_convert_from_simple_to_gmx():
    with tmp.TemporaryDirectory() as tmpdir:
        tmpbase = os.path.join(tmpdir, fnbase)
        outbase = os.path.join(tmpdir, out)

        save_data_list = []
        files = list(gen_filenames(tmpbase, num_files))

        for fn in files:
            for l in fields:
                save_data[l] = np.random.sample(datasize)

            save_data_list.append(save_data.copy())

            write(fn, save_data, ftype='simple')

        convert(tmpbase, outbase, ftype='gmx')

        for i, fn in enumerate(find_datamap_files(outbase)):
            data, _ = read_gmx(fn)

            for l in all_fields:
                if l in ('X', 'Y'):
                    assert (np.allclose(data[l], save_data_list[i][l] + 0.5, atol=1e-6))
                else:
                    assert (np.allclose(data[l], save_data_list[i][l], atol=1e-6))

def test_convert_from_simple_plain_to_simple():
    with tmp.TemporaryDirectory() as tmpdir:
        tmpbase = os.path.join(tmpdir, fnbase)
        outbase = os.path.join(tmpdir, out)

        save_data_list = []
        files = list(gen_filenames(tmpbase, num_files))

        for fn in files:
            for l in fields:
                save_data[l] = np.random.sample(datasize)

            save_data_list.append(save_data.copy())

            write(fn, save_data, ftype='simple_plain')

        convert(tmpbase, outbase, ftype='simple')

        for i, fn in enumerate(find_datamap_files(outbase)):
            data = read_binsimple(fn)

            for l in all_fields:
                assert (np.allclose(data[l], save_data_list[i][l], atol=1e-6))

def test_convert_fopts_plainsimple():
    with tmp.TemporaryDirectory() as tmpdir:
        tmpbase = os.path.join(tmpdir, fnbase)
        outbase = os.path.join(tmpdir, out)

        begin, end = 4, 6
        ext = '.tmp'
        fopts = {'begin': begin, 'end': end, 'ext': ext}

        save_data_list = []
        files = list(gen_filenames(tmpbase, begin=begin, end=end, ext=ext))
        for fn in files:
            for l in fields:
                save_data[l] = np.random.sample(datasize)
            save_data_list.append(save_data.copy())
            write(fn, save_data, ftype='simple')

        convert(tmpbase, outbase, begin=4, end=6, ftype='simple_plain', ext=ext)

        for i, fn in enumerate(find_datamap_files(outbase, **fopts)):
            data = read_plainsimple(fn)
            for l in all_fields:
                assert (np.allclose(data[l], save_data_list[i][l], atol=1e-6))

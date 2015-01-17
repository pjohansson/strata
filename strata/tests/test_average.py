import os
import numpy as np
import tempfile as tmp
from strata.average import *
from strata.utils import gen_filenames, find_datamap_files
from strata.dataformats.read import read_data_file
from strata.dataformats.simple.main import average_data, write

tmpfn = 'tmp_'
outfn = 'out_'
ext = '.dat'
num_maps = 12
group = 5

datasize = 4
fields = ('U', 'V', 'N', 'T', 'M')
save_data = {l: np.arange(datasize) for l in ('X', 'Y')}

def test_average_datamaps():
    with tmp.TemporaryDirectory() as tmpdir:
        tmpbase = os.path.join(tmpdir, tmpfn)
        outbase = os.path.join(tmpdir, outfn)

        # Create files with random data
        tmp_data = []
        for path in gen_filenames(tmpbase, num_maps):
            for l in fields:
                save_data.update({l: np.random.sample(datasize)})
            tmp_data.append(save_data.copy())

            write(path, save_data)

        # Average data
        average(tmpbase, outbase, group)

        # Verify output files against averaging directly
        out_files = find_datamap_files(outbase)
        for i, filename in enumerate(out_files):
            control_tmpdata = tmp_data[i*group:(i+1)*group]
            control = average_data(*control_tmpdata)

            data, _, _ = read_data_file(filename)
            for l in data.keys():
                assert (np.allclose(data[l], control[l], atol=1e-6))

        # Assert that the correct number of files were output
        assert (i+1 == np.floor(num_maps/group))

def test_average_datamaps_othernums():
    with tmp.TemporaryDirectory() as tmpdir:
        tmpbase = os.path.join(tmpdir, tmpfn)
        outbase = os.path.join(tmpdir, outfn)

        num_maps = 8
        begin = 4
        end = 9
        group = 3
        ext = '.tmp'

        tmp_data = []
        for path in gen_filenames(tmpbase, num_maps, ext=ext):
            for l in fields:
                save_data.update({l: np.random.sample(datasize)})
            tmp_data.append(save_data.copy())
            write(path, save_data)

        # Average data
        average(tmpbase, outbase, group, begin=begin, end=end, ext=ext)

        # Data starts at compensated numbering
        begin_out = np.ceil(begin/group)

        # Assert that the correct number of files were output
        out_files = list(find_datamap_files(outbase, begin=begin_out, ext=ext))
        assert (len(out_files) == 1)

import os
import numpy as np
import pytest
import tempfile as tmp

from strata.average import *
from strata.utils import gen_filenames, find_datamap_files
from strata.dataformats.read import read_data_file
from strata.dataformats.simple.main import average_data, write_data

tmpfn = 'tmp_'
outfn = 'out_'
ext = '.dat'
num_maps = 12
group = 5

datasize = 4
fields = ('U', 'V', 'N', 'T', 'M')

x = np.arange(np.sqrt(datasize))
xs, ys = np.meshgrid(x, x, indexing='ij')

save_data = {
    'X': xs.ravel(),
    'Y': ys.ravel(),
}


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

            write_data(path, save_data)

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
            write_data(path, save_data)

        # Average data
        average(tmpbase, outbase, group, begin=begin, end=end, ext=ext)

        # Data starts at compensated numbering
        begin_out = np.ceil(begin/group)

        # Assert that the correct number of files were output
        out_files = list(find_datamap_files(outbase, begin=begin_out, ext=ext))
        assert (len(out_files) == 1)


def get_datamap(dsize):
    data = {
        'X': np.arange(dsize),
        'Y': np.arange(dsize)
    }

    return data


def test_recenter_maps():
    # Create two datamaps with:
    #   x: 0, 1, 2, 3, 4, 5
    #   y: 0, 1, 2, 3, 4, 5
    data_maps = [get_datamap(6) for _ in (1, 2)]

    # Recenter x around 1 and 2
    recenter = [1., 2.]
    recentered_maps, info = recenter_maps(data_maps, recenter)

    # The remaining subset of x should be:
    xs = [-1., 0., 1., 2., 3.]

    # The surviving indices of the data maps are:
    # (i.e. the subset of x in both maps after recentering)
    indices = [
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5]
    ]

    for inds, recentered, orig in zip(indices, recentered_maps, data_maps):
        assert(np.array_equal(xs, recentered['X']))
        assert(np.array_equal(orig['Y'][inds], recentered['Y']))

    assert(np.array_equal([1., 1.], info['spacing']))


# Throw an error if the input number of x values do not match the number of maps
def test_recenter_non_broadcast():
    data_maps = [get_datamap(6) for _ in (1, 2)]

    with pytest.raises(IndexError):
        recenter_maps(data_maps, [1.])


def test_recenter_bad_xvals():
    data_maps = [get_datamap(6) for _ in (1, )]

    with pytest.raises(TypeError):
        recenter_maps(data_maps, [None])

import numpy as np
import os
import tempfile as tmp
from scripts.utils import find_datamap_files

fnbase = 'data_'
begin = 1
end=np.inf
num_maps = 10
ext = '.dat'

def run_datamaps_test(start, num_maps, ext, end):
    with tmp.TemporaryDirectory() as tmp_dir:
        for i in np.arange(start, min(start+num_maps, end)):
            fndata = '%s%05d%s' % (fnbase, i, ext)
            fnpath = os.path.join(tmp_dir, fndata)
            open(fnpath, 'w')

        paths_base = os.path.join(tmp_dir, fnbase)
        paths = find_datamap_files(paths_base, ext=ext, begin=start, end=end)

        print(paths)
        assert (len(paths) == min(num_maps, end - start))
        for p in paths:
            assert (os.access(p, os.F_OK) == True)

def test_find_datamaps():
    run_datamaps_test(begin, num_maps, ext, end)

def test_find_datamaps_otherext():
    run_datamaps_test(begin, num_maps, '.tmp', end)

def test_find_datamaps_othernums():
    begin, end = 23, 25
    run_datamaps_test(begin, num_maps, ext, end)

    begin, end = 0, 9
    run_datamaps_test(begin, num_maps, ext, end)

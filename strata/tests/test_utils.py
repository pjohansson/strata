import numpy as np
import os
import tempfile as tmp
from strata.utils import find_datamap_files

fnbase = 'data_'
begin_def = 1
end_def=np.inf
num_files_def = 10
extension_def = '.dat'
group_def = 1

def run_datamaps_test(**kwargs):
    first = kwargs.setdefault('begin', begin_def)
    last = kwargs.setdefault('end', end_def)
    ext = kwargs.setdefault('ext', extension_def)
    num_files = kwargs.setdefault('num_files', num_files_def)
    group = kwargs.setdefault('group', group_def)

    with tmp.TemporaryDirectory() as tmp_dir:
        # Create a number of files to find
        files = []
        num_files = min(num_files, last - first + 1)
        for i in np.arange(first, first+num_files):
            fndata = '%s%05d%s' % (fnbase, i, ext)
            files.append(os.path.join(tmp_dir, fndata))
            open(files[-1], 'w')

        # Create generator of paths from options
        paths_base = os.path.join(tmp_dir, fnbase)
        paths = find_datamap_files(paths_base, **kwargs)

        # If singles are requested, try to access each file
        if group == 1:
            for p in paths:
                files.remove(p)
                assert (os.access(p, os.F_OK) == True)
            # Verify that all files were found
            assert (files == [])

        # If grouped bundles are requested, check group-by-group
        else:
            for ps in paths:
                assert (len(ps) == group)
                for p in ps:
                    files.remove(p)
                    assert (os.access(p, os.F_OK) == True)
            # Verify that a possible remainder of non-accessed files is left
            assert (len(files) == num_files % group or num_files < 0)


def test_find_datamaps():
    run_datamaps_test()

def test_find_datamaps_otherext():
    run_datamaps_test(ext='.tmp')

def test_find_datamaps_othernums():
    run_datamaps_test(begin=23, end=25)
    run_datamaps_test(begin=0, end=9)
    run_datamaps_test(begin=5, end=1)

def test_find_datamaps_group():
    run_datamaps_test(num_files=20, end=20, group=5)
    run_datamaps_test(num_files=20, group=5)
    run_datamaps_test(num_files=23, group=5)
    run_datamaps_test(begin=5, end=1, group=5)

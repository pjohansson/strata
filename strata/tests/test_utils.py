import numpy as np
import os
import tempfile as tmp
from strata.utils import *

fnbase = 'data_'
begin_def = 1
end_def=np.inf
num_files_def = 10
ext_def = '.dat'
group_def = 1

def test_gen_filenames():
    filenames = list(gen_filenames(fnbase, 5, begin=4))
    assert (filenames == ['%s%05d%s' % (fnbase, i, ext_def) for i in (4,5)])
    filenames = list(gen_filenames(fnbase, 2, ext='.tmp'))
    assert (filenames == ['%s%05d%s' % (fnbase, i, '.tmp') for i in (1,2)])
    filenames = list(gen_filenames(fnbase, ext='.tmp', begin=3, end=5))
    assert (filenames == ['%s%05d%s' % (fnbase, i, '.tmp') for i in (3,4,5)])
    filenames = list(gen_filenames(fnbase, ext='.tmp', begin=5, end=3))
    assert (filenames == [])

def test_gen_filenames_inf():
    for i, filename in enumerate(gen_filenames(fnbase)):
        if i == 5: break
    assert (filename == '%s%05d%s' % (fnbase, i+1, ext_def))

def find_datamaps_test_runner(**kwargs):
    first = kwargs.setdefault('begin', begin_def)
    last = kwargs.setdefault('end', end_def)
    ext = kwargs.setdefault('ext', ext_def)
    num_files = kwargs.setdefault('num_files', num_files_def)
    group = kwargs.setdefault('group', group_def)

    # Create a number of files to find in a tmpdir
    with tmp.TemporaryDirectory() as tmp_dir:
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
    find_datamaps_test_runner()

def test_find_datamaps_otherext():
    find_datamaps_test_runner(ext='.tmp')

def test_find_datamaps_othernums():
    find_datamaps_test_runner(begin=23, end=25)
    find_datamaps_test_runner(begin=0, end=9)
    find_datamaps_test_runner(begin=5, end=1)

def test_find_datamaps_group():
    find_datamaps_test_runner(num_files=20, end=20, group=5)
    find_datamaps_test_runner(num_files=20, group=5)
    find_datamaps_test_runner(num_files=23, group=5)
    find_datamaps_test_runner(begin=5, end=1, group=5)

def test_prepare_path_decorator():
    @prepare_path
    def wrapped_output(*args, **kwargs):
        assert (len(args) == 1)
        assert (kwargs == {})

    cwd = os.getcwd()
    with tmp.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        # Test creation of a new directory
        path = os.path.join(tmp_dir, 'dirone/dirtwo/tmpfile')
        directory, filename = os.path.split(path)
        wrapped_output(path)
        assert (os.path.isdir(directory))
        assert (os.access(path, os.F_OK) == False)

        # Test back-up functionality of two backups
        path = os.path.basename(path)
        open(path, 'w')
        wrapped_output(path, _pp_verbose=True)
        assert (os.access(path, os.F_OK) == False)
        open(path, 'w')
        wrapped_output(path, _pp_verbose=False)

        backup_one = os.path.join('#%s.1#' % filename)
        backup_two = os.path.join('#%s.2#' % filename)
        assert (os.access(path, os.F_OK) == False)
        assert (os.access(backup_one, os.F_OK) == True)
        assert (os.access(backup_two, os.F_OK) == True)
        os.chdir(cwd)

def test_prepare_path_moreargs():
    @prepare_path
    def wrapped_output_twoargs(*args):
        assert (len(args) == 2)
    @prepare_path
    def wrapped_output_twokeys(path, **kwargs):
        assert (kwargs == {'k0': 'key0', 'k1': 'key1'})

    with tmp.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, 'tmpfile')
        wrapped_output_twoargs(path, [1,2,3])
        wrapped_output_twokeys(path, k0='key0', k1='key1')


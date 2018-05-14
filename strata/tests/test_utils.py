import numpy as np
import os
import pytest
import tempfile as tmp
import warnings

from droplets.flow import FlowData
from strata.utils import *

pytestmark = pytest.mark.filterwarnings('ignore')

fnbase = 'data_'
begin_def = 1
end_def=np.inf
num_files_def = 10
ext_def = '.dat'
group_def = None

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
        fn_gen = gen_filenames(fnbase, begin=first, end=first+num_files-1, ext=ext)
        for fndata in fn_gen:
            files.append(os.path.join(tmp_dir, fndata))
            open(files[-1], 'w')

        # Create generator of paths from options
        paths_base = os.path.join(tmp_dir, fnbase)
        paths = find_datamap_files(paths_base, **kwargs)

        # If singles are requested, try to access each file
        if group == None:
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
    find_datamaps_test_runner(num_files=5, group=1)
    find_datamaps_test_runner(num_files=20, group=5)
    find_datamaps_test_runner(num_files=23, group=5)
    find_datamaps_test_runner(begin=5, end=1, group=5)

def test_find_datamaps_raises_warning_if_no_files_are_found():
    with tmp.TemporaryDirectory() as tmp_dir:
        with pytest.warns(UserWarning):
            list(find_datamap_files('does_not_exist'))

def test_groups_to_singles_filenames():
    num_files = 10
    group = 5
    with tmp.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, fnbase)
        outbase = os.path.join(tmpdir, 'out_')
        files = []
        for filename in gen_filenames(base, num_files):
            files.append(filename)
            open(filename, 'w')

        # Assert that groups and singles are correct
        out_gen = gen_filenames(outbase)
        gen = find_groups_to_singles(base, outbase, group)
        i, j = 0, 0
        for from_fns, to_fn in gen:
            assert (to_fn == next(out_gen))
            for fn in from_fns:
                assert (fn == files[j])
                j += 1
            i += 1
        assert (i == np.floor(num_files/group))
        assert (j == num_files)

        # Assert that begin and end can be entered
        begin, end = 3, 5
        gen = find_groups_to_singles(base, outbase, group, begin=begin, end=end)
        for _, _ in gen:
            assert (False) # Should not be reached

        begin, end = 5, 3
        gen = find_groups_to_singles(base, outbase, group, begin=begin, end=end)
        for _, _ in gen:
            assert (False)

        # Assert that numbering is correct for larger begins
        group = 3
        for begin in (4, 5, 6):
            end = begin + group - 1
            gen = find_groups_to_singles(base, outbase, group, begin=begin, end=end)
            out_gen = gen_filenames(outbase, begin=2)
            for _, to_fn in gen:
                assert (to_fn == next(out_gen))


def test_groups_to_singles_otherext():
    num_files = 10
    group = 5
    with tmp.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, fnbase)
        outbase = os.path.join(tmpdir, 'out_')
        outext = '.xvg'
        files = []
        for filename in gen_filenames(base, num_files):
            files.append(filename)
            open(filename, 'w')

        # Assert that groups and singles are correct
        out_gen = gen_filenames(outbase, ext=outext)
        gen = find_groups_to_singles(base, outbase, group, outext=outext)
        for from_fns, to_fn in gen:
            assert (to_fn == next(out_gen))

def test_find_singles_to_singles():
    num_files = 10

    with tmp.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, fnbase)
        outbase = os.path.join(tmpdir, 'out_')

        files = []
        for fn in gen_filenames(base, num_files):
            open(fn, 'w')
            files.append(fn)

        out_gen = gen_filenames(outbase, num_files)
        gen = find_singles_to_singles(base, outbase)
        for i, (fnin, fnout) in enumerate(gen):
            assert (fnin == files[i])
            assert (fnout == next(out_gen))

def test_find_singles_to_singles_fopts():
    num_files = 10
    fopts = {'begin': 4, 'end': 6, 'ext': '.tmp'}

    with tmp.TemporaryDirectory() as tmpdir:
        base = os.path.join(tmpdir, fnbase)
        outbase = os.path.join(tmpdir, 'out_')

        files = []
        for fn in gen_filenames(base, **fopts):
            open(fn, 'w')
            files.append(fn)

        out_gen = gen_filenames(outbase, **fopts)
        gen = find_singles_to_singles(base, outbase, **fopts)
        for i, (fnin, fnout) in enumerate(gen):
            assert (fnin == files[i])
            assert (fnout == next(out_gen))

def test_catch_fileattr():
    kwargs = {'begin': 2, 'end': 4, 'ext': '.tmp', 'extra': [1,2,3]}
    attr = pop_fileopts(kwargs)
    assert (kwargs == {'extra': [1,2,3]})
    assert (attr == {'begin': 2, 'end': 4, 'ext': '.tmp', 'outext': '.tmp'})
    for i, _ in enumerate(gen_filenames('test_', **attr)):
        pass
    assert (i == 2)

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

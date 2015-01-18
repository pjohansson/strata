import strata.dataformats as formats

"""Module for outputting data."""

# Set module handles for ftype
default_module = formats.simple.main
modules = {
        'default': default_module,
        'simple': formats.simple.main,
        'simple_plain': formats.simple.main
        }

def write(path, data, *args, **kwargs):
    """Output data to a path.

    This function selects a default module to use for output and calls
    its 'write_data' function. The module can be selected explicitly
    using the keyword argument 'ftype'.

    Arguments and keyword arguments are passed on to the functions.

    Args:
        path (str): Write to a file at this path.

        data (dict): Data to write.

    Keyword Args:
        ftype (str, default='simple'): File type to write. Choices:
            'simple'       - Simple binary    (strata.dataformats.simple.write)
            'simple_plain' - Simple plaintext (strata.dataformats.simple.write)

    Raises:
        KeyError: If a non-existant 'ftype' is specified.

    """

    def write_simple(ftype):
        if ftype == 'simple_plain':
            kwargs.update({'binary': False})
        modules[ftype].write_data(path, data, *args, **kwargs)

    ftype = kwargs.pop('ftype', 'default')
    try:
        write_simple(ftype)
    except KeyError:
        raise KeyError("specified 'ftype' not existing for writing.")

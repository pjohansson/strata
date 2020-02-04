import strata.dataformats as formats

"""Module for outputting data."""

# Set module handles for ftype
default_module = formats.gmx_flow_version_1.main
modules = {
        'default': default_module,
        'gmx': formats.gmx_flow_version_1.main,
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
        ftype (str, default='gmx'): File type to write. Choices:
            'gmx'          - Gromacs flow format (strata.dataformats.gmx_flow_version_1)
            'simple'       - Simple binary    (strata.dataformats.simple)
            'simple_plain' - Simple plaintext (strata.dataformats.simple)

    Raises:
        KeyError: If a non-existant 'ftype' is specified.

    """

    def write_simple(ftype):
        if ftype == 'simple_plain':
            kwargs.update({'binary': False})

        modules[ftype].write_data(path, data, **kwargs)

    ftype = kwargs.pop('ftype', 'default')

    try:
        assert (ftype in modules.keys())
    except AssertionError:
        raise KeyError("specified 'ftype' not existing for writing.")
    
    if ftype in ['gmx', 'default']:
        modules[ftype].write_data(path, data, *args)
    else:
        write_simple(ftype)


def flowdata_to_dict(flow):
    """Convert a FlowData object to write-compatible dictionary.

    Args:
        flow (FlowData): Object to convert.

    Returns:
        dict: Data to write.

    """

    return {key: flow.data[key] for key in flow.data.dtype.names}

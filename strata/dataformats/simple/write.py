import numpy as np
from strata.utils import prepare_path

"""Module for writing data to disk in simple formats."""

@prepare_path
def write_data(path, data, binary=True):
    """Write data to disk in a simple data format.

    Input data must contain fields ('X', 'Y', 'U', 'V', 'N', 'T', 'M').

    Args:
        path (str): Write to a file at this path.

        data (dict): Data to write.

        binary (bool): If True write to the simple binary format else
            to simple plaintext.

    """

    def write_binsimple():
        save_data.tofile(path)

    def write_plainsimple():
        header = ' '.join(fields_ordered)
        np.savetxt(path, save_data, fmt='%6f', delimiter=' ',
                header=header, comments='')


    fields_ordered = ['X', 'Y', 'N', 'T', 'M', 'U', 'V']

    btype = 'float32'
    save_data = np.array([data[l] for l in fields_ordered], dtype=btype).T

    if binary:
        write_binsimple()
    else:
        write_plainsimple()

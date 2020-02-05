import numpy as np
from strata.utils import prepare_path

"""Module for writing data to disk in a specified format."""

FIELDS_ORDERED = ['N', 'T', 'M', 'U', 'V']

@prepare_path
def write_data(path, data, info, check_label='M'):
    """Write data to disk in a data format with only non-empty bins.

    Args:
        path (str): Write to a file at this path.

        data (dict): Data to write.

        shape (2-tuple, int): Shape of grid along x and y.

        spacing (2-tuple, float): Spacing between bins along x and y.

        origin (2-tuple, float): Position of bottom-left cell along x and y.

    Keyword args:
        check_label (str): Label in dict which must be non-zero for a bin.

    """

    nx, ny = info['shape']

    inds = data[check_label] != 0.0
    num_elements = data[check_label][inds].size

    ix = np.arange(nx, dtype=np.uint64)
    iy = np.arange(ny, dtype=np.uint64)
    ixs, iys = np.meshgrid(ix, iy, indexing='ij')

    output_data = [
            ixs.ravel()[inds],
            iys.ravel()[inds]
        ] + [np.array(data[l][inds], dtype=np.float32) for l in FIELDS_ORDERED]

    with open(path, "wb") as fp:
        write_header(
            fp, info['shape'], info['spacing'], info['origin'], num_elements
        )

        for vs in output_data:
            vs.tofile(fp)

def write_header(fp, shape, spacing, origin, num_elements):
    fp.write("FORMAT GMX_FLOW_1\n".encode())
    fp.write("ORIGIN {:12f} {:12f}\n".format(origin[0], origin[1]).encode())
    fp.write("SHAPE {} {}\n".format(shape[0], shape[1]).encode())
    fp.write("SPACING {:12f} {:12f}\n".format(spacing[0], spacing[1]).encode())
    fp.write("NUMDATA {}\n".format(num_elements).encode())

    fp.write("FIELDS IX IY".encode())
    for l in FIELDS_ORDERED:
        fp.write(" {}".format(l).encode())
    fp.write("\n".encode())

    fp.write("COMMENT Grid is regular but only non-empty bins are output\n".encode())
    fp.write("COMMENT There are 'NUMDATA' non-empty bins and that many values are stored for each field\n".encode())
    fp.write("COMMENT 'FIELDS' is the different fields for each bin:\n".encode())
    fp.write("COMMENT 'IX' and 'IY' are bin indices along x and y respectively\n".encode())
    fp.write("COMMENT 'N' is the average number of atoms\n".encode())
    fp.write("COMMENT 'M' is the average mass\n".encode())
    fp.write("COMMENT 'T' is the temperature\n".encode())
    fp.write("COMMENT 'U' and 'V' is the mass flow along x and y respectively\n".encode())
    fp.write("COMMENT Data is stored as 'NUMDATA' counts for each field in 'FIELDS', in order\n".encode())
    fp.write("COMMENT 'IX' and 'IY' are 64-bit unsigned integers\n".encode())
    fp.write("COMMENT Other fields are 32-bit floating point numbers\n".encode())
    fp.write("COMMENT Example: with 'NUMDATA' = 4 and 'FIELDS' = 'IX IY N T', "
             "the data following the '\\0' marker is 4 + 4 64-bit integers "
             "and then 4 + 4 32-bit floating point numbers\n".encode())

    fp.write(b"\0")

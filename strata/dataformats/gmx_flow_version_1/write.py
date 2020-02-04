import numpy as np
from strata.utils import prepare_path

"""Module for writing data to disk in a specified format."""

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

    with open(path, "w") as fp:
        columns = list(data.keys())

        try:
            columns.remove('X')
        except:
            pass
        try:
            columns.remove('Y')
        except:
            pass

        write_header(fp, info['shape'], info['spacing'], info['origin'], columns)

        grid_data = { l: vs.reshape(nx, ny) for l, vs in data.items() }
        inds = data[check_label] != 0.0

        for ix in np.arange(nx):
            for iy in np.arange(ny):
                if grid_data[check_label][ix, iy] != 0.0:
                    fp.write("{} {}".format(ix, iy))

                    for l in columns:
                        fp.write(" {:9f}".format(grid_data[l][ix, iy]))
                    
                    fp.write("\n")

def write_header(fp, shape, spacing, origin, columns):
    fp.write("FORMAT GMX_FLOW_1\n")
    fp.write("ORIGIN {:12f} {:12f}\n".format(origin[0], origin[1]))
    fp.write("SHAPE {} {}\n".format(shape[0], shape[1]))
    fp.write("SPACING {:12f} {:12f}\n".format(spacing[0], spacing[1]))
    fp.write("COMMENT Grid is regular but only non-empty bins are output\n")
    fp.write("COMMENT Every output bin has a separate row\n")
    fp.write("COMMENT 'ix' and 'iy' are bin indices along x and y respectively\n")
    fp.write("COMMENT Values are white space separated\n")
    fp.write("COMMENT Bin data begins after 'COLUMNS' row\n")
    fp.write("COMMENT 'N' is the average number of atoms\n")
    fp.write("COMMENT 'M' is the average mass\n")
    fp.write("COMMENT 'T' is the temperature\n")
    fp.write("COMMENT 'U' and 'V' is the mass flow along x and y respectively\n")

    fp.write("COLUMNS IX IY")

    for l in columns:
        fp.write(" {}".format(l))
    
    fp.write("\n")


        
        
        

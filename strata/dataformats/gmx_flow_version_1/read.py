import numpy as np

FIELDS = ['X', 'Y', 'N', 'T', 'M', 'U', 'V']

def read_data(filename):
    """Read field data from a file.

    Args:
        filename (str): A file to read data from.
    
    Returns:
        (dict, dict): 2-tuple of dict's with data and informatioin. See 
            strata.dataformats.read.read_data_file for more information.

    """

    with open(filename, 'r') as fp:
        columns, info = read_header(fp)

        dtype = [(l, np.uint) if l in ('IX', 'IY') else (l, np.float) for l in columns]
        data = np.genfromtxt(fp, dtype=dtype)

    x0, y0 = info['origin']
    nx, ny = info['shape']
    dx, dy = info['spacing']

    x = x0 + dx * (np.arange(nx) + 0.5)
    y = y0 + dy * (np.arange(ny) + 0.5)
    xs, ys = np.meshgrid(x, y, indexing='ij')

    grid = np.zeros((nx, ny), dtype=[(l, np.float) for l in FIELDS])
    grid['X'] = xs
    grid['Y'] = ys

    for cell in data:
        ix = cell['IX']
        iy = cell['IY']

        for l in ['N', 'T', 'M', 'U', 'V']:
            grid[ix, iy][l] = cell[l]
    
    grid = grid.ravel()
    
    return {l: grid[l] for l in FIELDS}, info

def read_header(fp):
    """Read header information and forward the pointer to the data."""

    def read_shape(line):
        return tuple(int(v) for v in line.split()[1:3])

    def read_spacing(line):
        return tuple(float(v) for v in line.split()[1:3])
    
    def parse_field_labels(line):
        return line.split()[1:]

    line = fp.readline().strip()
    line_type = line.split(maxsplit=1)[0].upper()

    info = {}

    while not line_type == "COLUMNS":
        if line_type == "SHAPE":
            info['shape'] = read_shape(line)
        elif line_type == "SPACING":
            info['spacing'] = read_spacing(line)
        elif line_type == "ORIGIN":
            info['origin'] = read_spacing(line)
        
        line = fp.readline().strip()
        line_type = line.split(maxsplit=1)[0].upper()
    
    columns_line = line
    
    info['num_bins'] = info['shape'][0] * info['shape'][1]
    
    columns = parse_field_labels(columns_line)

    return columns, info

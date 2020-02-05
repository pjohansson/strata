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

    with open(filename, 'rb') as fp:
        fields, num_values, info = read_header(fp)
        data = read_values(fp, num_values, fields)

    x0, y0 = info['origin']
    nx, ny = info['shape']
    dx, dy = info['spacing']

    x = x0 + dx * (np.arange(nx) + 0.5)
    y = y0 + dy * (np.arange(ny) + 0.5)
    xs, ys = np.meshgrid(x, y, indexing='ij')

    grid = np.zeros((nx, ny), dtype=[(l, np.float) for l in FIELDS])
    grid['X'] = xs
    grid['Y'] = ys

    for l in ['N', 'T', 'M', 'U', 'V']:
        grid[l][data['IX'], data['IY']] = data[l]

    grid = grid.ravel()

    return {l: grid[l] for l in FIELDS}, info

def read_values(fp, num_values, fields):
    dtypes = {
        'IX': np.uint64,
        'IY': np.uint64,
        'N': np.float32,
        'T': np.float32,
        'M': np.float32,
        'U': np.float32,
        'V': np.float32,
    }

    return {
        l: np.fromfile(fp, dtype=dtypes[l], count=num_values)
        for l in fields
    }

def read_header(fp):
    """Read header information and forward the pointer to the data."""

    def read_shape(line):
        return tuple(int(v) for v in line.split()[1:3])

    def read_spacing(line):
        return tuple(float(v) for v in line.split()[1:3])

    def read_num_values(line):
        return int(line.split()[1].strip())

    def parse_field_labels(line):
        return line.split()[1:]

    def read_header_string(fp):
        buf_size = 1024
        header_str = ""

        while True:
            buf = fp.read(buf_size)

            pos = buf.find(b'\0')

            if pos != -1:
                header_str += buf[:pos].decode("ascii")
                offset = buf_size - pos - 1
                fp.seek(-offset, 1)
                break
            else:
                header_str += buf.decode("ascii")

        return header_str

    info = {}
    header_str = read_header_string(fp)

    for line in header_str.splitlines():
        line_type = line.split(maxsplit=1)[0].upper()

        if line_type == "SHAPE":
            info['shape'] = read_shape(line)
        elif line_type == "SPACING":
            info['spacing'] = read_spacing(line)
        elif line_type == "ORIGIN":
            info['origin'] = read_spacing(line)
        elif line_type == "FIELDS":
            fields = parse_field_labels(line)
        elif line_type == "NUMDATA":
            num_values = read_num_values(line)

    info['num_bins'] = info['shape'][0] * info['shape'][1]

    return fields, num_values, info

import matplotlib.pyplot as plt
import numpy as np

def get_droplet_interface(flow, label, cutoff=None, coord_labels=('X', 'Y')):
    """Return droplet interface coordinates.

    Currently works only with a fully initialised, rectangular system.
    Returns the longest contour found in the system.

    Args:
        flow (FlowData): A FlowData object. Must contain a data record
            with coordinates and selected height map label, as well
            as an input shape of the system.

        label (str): Record label used as base for the interface height map.

        cutoff (float, optional): Which interface height to cut the contour at.
            Defaults to the halfway height.

        coord_labels (2-tuple, optional): Record labels for coordinates.

    Returns:
        ndarray: Array with 2-tuples of interface coordinates.

    Raises:
        AttributeError: If no unique contour was found.

    """

    def get_levels(cutoff):
        if cutoff != None:
            return [cutoff if cutoff > hdata.min() else hdata.min()]
        else:
            return 1

    def get_vertices(paths):
        if paths == []:
            return paths

        lengths = np.array([len(apath.vertices) for apath in paths])
        if len(lengths == lengths.max()) == 1:
            return paths[lengths.argmax()].vertices
        else:
            raise AttributeError("could not find unique contour as interface")

    # Get shaped view of system
    shape = flow.shape
    data = flow.data.reshape(shape)

    x, y = [data[clabel] for clabel in coord_labels]
    hdata = data[label]

    # Get one contour level from the data
    levels = get_levels(cutoff)
    cx = plt.contour(x, y, hdata, levels)
    paths = cx.collections[0].get_paths()

    return get_vertices(paths)

import numpy as np
import matplotlib.pyplot as plt

def draw_flowfield(data, fields=('X', 'Y', 'U', 'V')):
    """Draw a flow field from input data as a quiver graph.

    Args:
        data (record): Field data in record format.

        fields (array_like, optional): Ordered list of record labels in
            which data is entered to the draw command.

    Returns:
        fig: Figure handle to drawn graph.

    """

    fields = [data[label] for label in fields]
    fig = plt.quiver(*fields)

    return fig


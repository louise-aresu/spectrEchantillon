import numpy as np

def frame_dim(size):
    """
    @param:
        - size: Array or Integer
    @return:
        - tuple of the frame dimensions
    """
    assert (type(size) == int or len(size) <= 3)
    if type(size) == int:
        N1 = size
        N2 = 1
    else:
        N1 = size[0]
        N2 = size[1] if len(size) >= 2 else 1

    return (N1, N2)
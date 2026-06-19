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

def progress_bar(n, maxn, mean_time, progress_bar_length=50):
    print("", end='')
    print(f'\r{n}/{max}\t- {int(n/maxn * 100) : 4d}% ', end='')
    print('[', end='')
    print('#' * int(np.ceil(n/maxn * progress_bar_length)), end='')
    print('.' * int(progress_bar_length - np.ceil(n/maxn * progress_bar_length)), end='')
    print('] ', end='')
    print('[ est. ', end='')
    if mean_time.tm_hour > 0:
        print(f'{mean_time.tm_hour}h ', end='')
        print(f'{mean_time.tm_min}m ', end='')
    elif mean_time.tm_min > 0:
        print(f'{mean_time.tm_min}m ', end='')
        print(f'{mean_time.tm_sec}s ', end='')
    elif mean_time.tm_sec > 0:
        print(f'{mean_time.tm_sec}s ', end='')
    print(']', end='')
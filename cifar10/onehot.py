import numpy as np
# y is list
# num_labels is number_of_labels

def np_onehot(y, num_labels, dtype='float'):
    if not isinstance(num_labels, int):
        return
    y = np.reshape(y,-1)

    if isinstance(y, list):
        yt = np.asarray(y)
    else:
        yt = y

    if not len(yt.shape) == 1:
        raise AttributeError('y array must be 1-dimensional')
    uniq = np.max(yt + 1)
    ary = np.zeros((len(y), uniq))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary.astype(dtype)

import numpy as np
def KnuthSampling(total, m, left = -1):
    '''
    :param total: total
    :param m: sample
    :return:
    '''
    res = []
    n = total
    for i in range(total):
        if i != left:
            if np.random.random() < m / n:
                res.append(i)
                m -= 1
        n -= 1
    return res

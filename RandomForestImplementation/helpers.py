# TODO: Im Moment ist dieser Helfer-Code **nicht** von mir geschrieben

import math


def entropy_func(c, n):
    """
    The math formula
    """
    return -(c * 1.0 / n) * math.log(c * 1.0 / n, 2)


def entropy_cal(c1, c2):
    """
    Returns entropy of a group of data
    c1: count of one class
    c2: count of another class
    """
    if c1 == 0 or c2 == 0:  # when there is only one class in the group, entropy is 0
        return 0
    return entropy_func(c1, c1 + c2) + entropy_func(c2, c1 + c2)


# get the entropy of one big circle showing above
def entropy_of_one_division(division):
    """
    Returns entropy of a divided group of data
    Data may have multiple classes
    """
    s = 0
    n = len(division)
    classes = set(division)
    for c in classes:  # for each class, get entropy
        n_c = sum(division == c)
        e = n_c * 1.0 / n * entropy_cal(sum(division == c), sum(division != c))  # weighted avg
        s += e
    return s, n


# The whole entropy of two big circles combined
def get_entropy(y_predict, y_real):
    """
    Returns entropy of a split
    y_predict is the split decision, True/Fasle, and y_true can be multi class
    """
    if len(y_predict) != len(y_real):
        print('They have to be the same length')
        return None
    n = len(y_real)
    s_true, n_true = entropy_of_one_division(y_real[y_predict])  # left hand side entropy
    s_false, n_false = entropy_of_one_division(y_real[~y_predict])  # right hand side entropy
    s = n_true * 1.0 / n * s_true + n_false * 1.0 / n * s_false  # overall entropy, again weighted average
    return s


def find_best_split_of_all(x, y):
    col = None
    min_entropy = 1
    cutoff = None
    for i, c in enumerate(x.T):
        entropy, cur_cutoff = find_best_split(c, y)
        if entropy == 0:  # find the first perfect cutoff. Stop Iterating
            return i, cur_cutoff, entropy
        elif entropy <= min_entropy:
            min_entropy = entropy
            col = i
            cutoff = cur_cutoff
    return col, cutoff, min_entropy


def find_best_split(col, y):
    min_entropy = 10
    n = len(y)
    for value in set(col):
        y_predict = col < value
        my_entropy = get_entropy(y_predict, y)
        if my_entropy <= min_entropy:
            min_entropy = my_entropy
            cutoff = value
    return min_entropy, cutoff

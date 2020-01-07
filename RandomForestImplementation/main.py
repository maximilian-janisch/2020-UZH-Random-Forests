import numpy as np

from RandomForestImplementation.helpers import find_best_split_of_all


class Tree:
    def __init__(self, prediction, feature_index=None, feature_name=None, cutoff=None):
        self.feature_name = feature_name
        self.feature_index = feature_index
        self.cutoff = cutoff
        self.prediction = prediction

        self.left = None
        self.right = None
    # TODO: Repräsentierung des trees durch __repr__ ?


class DecisionTreeClassifier(object):
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def fit(self, x, y):
        depth = 0
        base = Tree(feature_name="base", prediction=np.round(np.mean(y)))
        queue = [(base, x, y, depth)]

        while 0 < len(queue):
            # print(len(queue))
            """
            Each node in the queue which are just terminal nodes get replaced by a split->prediction1/prediction2 construct.
            The old node gets removed from the queue and the new nodes get added to it
            """
            c, x, y, d = queue.pop(0)
            if len(y) == 0 or d + 1 >= self.max_depth or len(np.unique(y)) == 1: continue
            # Adding a new layer would exceed the maximum depth of the tree or we have no more data left
            # or further fitting to our data would not change the predictions (if all the y are the same)

            col, cutoff, entropy = find_best_split_of_all(x, y)  # find the best split for the data at the node
            filter_ = x[:, col] < cutoff
            x_left, y_left = x[filter_], y[filter_]
            x_right, y_right = x[np.logical_not(filter_)], y[np.logical_not(filter_)]
            # print(x_left, y_left, x_right, y_right, sep="\n")

            left_c, right_c = Tree(prediction=np.round(np.mean(y_left))), Tree(prediction=np.round(np.mean(y_right)))
            c.left, c.right = left_c, right_c
            c.cutoff = cutoff
            c.feature_index = col
            # TODO: feature_name !

            queue.extend([(left_c, x_left, y_left, d + 1), (right_c, x_right, y_right, d + 1)])
        return base

    def predict(self, x):
        # TODO: Testing !
        return np.apply_along_axis(self._get_prediction, axis=1, arr=x)

    def _get_prediction(self, row):
        # TODO: Get prediction
        # TODO: static method ?
        pass


if __name__ == '__main__':  # Test
    from sklearn.datasets import load_iris

    iris = load_iris()
    x = iris.data
    y = iris.target
    a = DecisionTreeClassifier(7)
    b = a.fit(x, y)

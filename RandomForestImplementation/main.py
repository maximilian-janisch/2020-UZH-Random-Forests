import numpy as np

from RandomForestImplementation.helpers import best_split


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
        n_classes = len(np.unique(y))
        n_features = x.shape[1]
        depth = 0
        base = Tree(feature_name="base", prediction=np.round(np.mean(y)))
        queue = [(base, x, y, depth)]

        while queue:
            """
            Each node in the queue which are just terminal nodes get replaced by a split->prediction1/prediction2 construct.
            The old node gets removed from the queue and the new nodes get added to it
            """
            c, x, y, d = queue.pop(0)
            if len(np.unique(y)) <= 1 or d + 1 >= self.max_depth or len(np.unique(y)) == 1:
                """
                Adding a new layer would exceed the maximum depth of the tree or we have no more data left
                or further fitting to our data would not change the predictions (this happens if all the y are the same)
                """
                continue

            # print("Trying to get split") TODO: remove
            col, cutoff = best_split(x, y, n_classes, n_features)  # find the best split for the data at the node
            # print("Got split") TODO: remove
            filter_ = x[:, col] < cutoff
            x_left, y_left = x[filter_], y[filter_]
            x_right, y_right = x[np.logical_not(filter_)], y[np.logical_not(filter_)]
            # print(x_left, y_left, x_right, y_right, sep="\n") TODO: remove

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
    from sklearn.datasets import load_digits

    digits = load_digits()
    x = digits.data
    y = digits.target
    a = DecisionTreeClassifier(5)
    classifier = a.fit(x, y)

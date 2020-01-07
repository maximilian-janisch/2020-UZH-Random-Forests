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

    def __repr__(self):
        if self.cutoff is not None:
            return f"Splitting node which splits {self.feature_name} (index {self.feature_index}) at cutoff {self.cutoff}"
        else:
            return f"Terminal node with prediction {int(self.prediction)}"


class DecisionTreeClassifier(object):
    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.ctree: Tree = None

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
        self.ctree = base

    def predict(self, x):
        return np.apply_along_axis(self.__prediction_for_row, axis=1, arr=x).astype(int)

    def __prediction_for_row(self, data_row):
        base = self.ctree
        if base is None:  # tree hasn't been fitted yet
            raise Exception("This tree hasn't been fitted yet")

        while base.feature_index is not None:
            if data_row[base.feature_index] < base.cutoff:
                base = base.left
            else:
                base = base.right

        return base.prediction


if __name__ == '__main__':  # Test
    from sklearn.datasets import load_digits

    digits = load_digits()
    x = digits.data
    y = digits.target
    classifier = DecisionTreeClassifier(10)
    classifier.fit(x, y)

    correct = np.sum(y == classifier.predict(x))

    print(f"Out of {len(y)} observations, the decision tree with depth {classifier.max_depth} classified {correct} correctly")
    # Output: Out of 1797 observations, the decision tree with depth 10 classified 1599 correctly
    # Attention: The above tree is probably overfitting the data

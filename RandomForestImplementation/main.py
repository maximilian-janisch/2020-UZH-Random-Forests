import numpy as np

from RandomForestImplementation.helpers import Splitter


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


class DecisionTreeClassifier:
    def __init__(self, max_depth: int, max_features: int="all"):
        self.max_depth = max_depth
        self.max_features = max_features

        self.ctree: Tree = None

    def fit(self, X, y):
        depth = 0
        base = Tree(feature_name="base", prediction=np.round(np.mean(y)))

        queue = [(base, X, y, depth)]
        splitter = Splitter(n_classes=len(np.unique(y)), max_features=self.max_features)

        while queue:
            """
            Each node in the queue which are just terminal nodes get replaced by a split->prediction1/prediction2 construct.
            The old node gets removed from the queue and the new nodes get added to it
            """
            c, X, y, d = queue.pop(0)
            if len(np.unique(y)) <= 1 or d + 1 >= self.max_depth or len(np.unique(y)) == 1:
                # Exceeding of the maximum depth of the tree or no more data left
                continue

            col, cutoff = splitter.best_split(X, y)  # find the best split for the data at the node
            filter_ = X[:, col] < cutoff
            x_left, y_left = X[filter_], y[filter_]
            x_right, y_right = X[np.logical_not(filter_)], y[np.logical_not(filter_)]

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
    X = digits.data
    y = digits.target
    classifier = DecisionTreeClassifier(7, 15)
    classifier.fit(X, y)

    correct = np.sum(y == classifier.predict(X))

    print(f"Out of {len(y)} observations, the decision tree with depth {classifier.max_depth}"
          f" and max_features {classifier.max_features} classified {correct} correctly.")

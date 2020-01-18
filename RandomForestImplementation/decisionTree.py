import numpy as np

from RandomForestImplementation.helpers import Splitter


class Tree:
    def __init__(self, class_counts: np.ndarray, feature_index=None, cutoff=None):
        """
        Initializes the tree class
        :param class_counts: Number of elements of each class falling in this node. For example [1, 4] means that
        class 0 occurs 1 time and class 1 occurs 4 times. If the tree is fully grown this list should contain all zeroes
        at all indices except for exactly one.
        :param feature_index: Index of the feature to split along
        :param cutoff: Threshold of the split
        """
        self.feature_index = feature_index
        self.cutoff = cutoff
        self.class_counts = class_counts

        self.left = None
        self.right = None

    def __repr__(self):
        if self.cutoff is not None:
            return f"Splitting node which splits the feature with index {self.feature_index} at cutoff {self.cutoff}"
        else:
            return f"Terminal node with class counts {self.class_counts}"


class DecisionTreeClassifier:
    def __init__(self, max_depth: int=np.Inf, max_features: int="all", min_samples: int=2):
        """
        Initializes the DecisionTreeClassifer class
        :param max_depth: Maximum height of the tree. Default is np.Inf, which fully expands the tree
        :param max_features: Size of the subset of features to randomly choose at each node while fitting
        :param min_samples: Minimum number of features in order to allow splitting of a node
        """
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples = min_samples

        self.ctree: Tree = None

    def fit(self, X, y):
        """Fits a single tree in order to predict y given X."""
        depth = 0
        num_classes = len(np.unique(y))
        base = Tree(class_counts=np.bincount(y))

        queue = [(base, X, y, depth)]
        splitter = Splitter(n_classes=num_classes, max_features=self.max_features)

        while queue:
            """
            Each node in the queue is a terminal node. These nodes get replaced by a split->prediction1/prediction2 
            construct. The original node gets removed from the queue and the new nodes get added to it
            """
            c, X, y, d = queue.pop(0)
            if len(np.unique(y)) <= 1 or len(y) <= self.min_samples or d + 1 >= self.max_depth:
                # Exceeding of the maximum depth of the tree or no more data left
                continue

            col, cutoff = splitter.best_split(X, y)  # find the best split for the data at the node
            if col is None:  # there is no split that reduces the Gini impurity
                continue
            filter_ = X[:, col] < cutoff
            x_left, y_left = X[filter_], y[filter_]
            x_right, y_right = X[np.logical_not(filter_)], y[np.logical_not(filter_)]

            left_c = Tree(class_counts=np.bincount(y_left, minlength=num_classes))
            right_c = Tree(class_counts=np.bincount(y_right, minlength=num_classes))
            c.left, c.right = left_c, right_c
            c.cutoff = cutoff
            c.feature_index = col

            queue.extend([(left_c, x_left, y_left, d + 1), (right_c, x_right, y_right, d + 1)])
        self.ctree = base

    def predict(self, X):
        """
        Gives the predicted classes for the data X
        :param X: n times p dimensional matrix (n is the number of observations and p the number of features)
        :returns: n dimensional vector where the ith element of that vector corresponds to the predicted class of the ith row of X
        """
        return np.apply_along_axis(np.argmax, axis=1, arr=self.predict_proba(X)).astype(int)

    def predict_proba(self, X):
        """
        Predict predictions of the random tree for the data X in terms of probabilities
        :param X: n times p dimensional matrix (n is the number of observations and p the number of features)
        :return: n times m dimensional matrix (m is the number of classes)
        """
        return np.apply_along_axis(self.__prediction_of_probability_for_row, axis=1, arr=X)

    def score(self, X, y):
        """
        Returns relative number of correctly predicted data rows.
        Note: This function is used for compatibility with sklearns cross_validation utilities
        :param X: Features to predict from
        :param y: Correct labels
        :return: Score between 0 (no rows correctly classified) and 1 (all labels correctly classified)
        """
        return (self.predict(X) == y).sum() / len(y)

    def get_params(self, deep=True):
        """
        Gets parameters of the classifier.
        Note: This function is required for compatibility with sklearns cross_validation utilities.
        """
        return {"max_depth": self.max_depth, "max_features": self.max_features, "min_samples": self.min_samples}

    def set_params(self, **params):
        """
        Sets parameters for this classifier
        Note: This function is required for compatibility with sklearns cross_validation utilities.
        """
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def __prediction_of_probability_for_row(self, x):
        """
        Returns the probabilities with which the decision tree predicts that the data row x is associated to each class
        :param x: p-dimensional vector, where p is the number of features
        :returns: m-dimensional vector, where m is the number of classes, such that the elements of that vector are non-negative and sum to one
        """
        base = self.ctree
        if base is None:  # tree hasn't been fitted yet
            raise Exception("This tree hasn't been fitted yet")

        while base.feature_index is not None:
            if x[base.feature_index] < base.cutoff:
                base = base.left
            else:
                base = base.right

        return base.class_counts / np.sum(base.class_counts)


if __name__ == '__main__':  # Test
    from sklearn.datasets import load_digits

    digits = load_digits()
    X = digits.data
    y = digits.target
    classifier = DecisionTreeClassifier(max_features=8)
    classifier.fit(X, y)

    correct = np.sum(y == classifier.predict(X))

    print(f"Out of {len(y)} observations, the decision tree with depth {classifier.max_depth}"
          f" and max_features {classifier.max_features} classified {correct} correctly.")

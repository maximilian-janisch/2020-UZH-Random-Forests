import numpy as np


class Splitter:
    def __init__(self, n_classes, max_features="all"):
        """
        Initializes the Splitter class
        :param n_classes: Number of distinct labels y_i
        :param max_features: Number of features to randomly choose at each node when performing a split. If set to "all"
        then all features are used. If max_features is larger than the number of available features, all features are
        used too.
        """
        self.n_classes = n_classes
        self.max_features = max_features

    def best_split(self, X, y):
        """
        For given data X (m times p matrix, where p is the number of features and n the number of samples)
        and given labels y (m-dimenstional vector), this function finds the best split for that data.
        split for a node.

        Best means that the average impurity of the two children, weighted by their
        population, is the smallest possible.
        To find this split, we compute the optimal split threshold for each feature
        and return the feature/threshold pair, which minimizes the Gini impurity.
        :param X: numpy array with shape (m, p)
        :param y: numpy array with shape (m, )
        :return: optimal split
        """

        m = y.size  # number of samples
        p = X.shape[1]  # number of features
        class_count = [np.sum(y == c) for c in range(self.n_classes)]  # class counts

        best_gini = 1.0 - sum((n / m) ** 2 for n in class_count)  # Gini impurity of current split
        best_idx, best_thr = None, None

        if isinstance(self.max_features, int) and self.max_features <= p:
            feature_subset = np.sort(
                np.random.choice(np.arange(p), size=self.max_features, replace=False)
            )
        else:
            feature_subset = np.arange(p)

        # Loop through all features of the subset
        for idx in feature_subset:
            # Sort data along selected feature
            threshholds, features = zip(*sorted(zip(X[:, idx], y)))

            # We could actually split the node according to each feature/threshold pair
            # and count the resulting population for each class in the children, but
            # instead we compute them in an iterative fashion, making this for loop
            # linear rather than quadratic.
            num_left = [0] * self.n_classes
            num_right = class_count.copy()
            for i in range(1, m):  # possible split positions
                c = features[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes)
                )

                # the Gini impurity of a split is the weighted average of the Gini impurity of the children.
                gini = (i * gini_left + (m - i) * gini_right) / m

                if threshholds[i] == threshholds[i - 1]:  # avoid points that are identical to split to different sides
                    continue

                if gini < best_gini:  # minimal gini found
                    best_gini = gini
                    best_idx = idx
                    best_thr = (threshholds[i] + threshholds[i - 1]) / 2  # midpoint

        return best_idx, best_thr

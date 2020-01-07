import numpy as np

from RandomForestImplementation.decisionTree import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(self, n_estimators, max_features: int="sqrt", max_depth=np.Inf, min_samples=2):
        """
        Initializes the RandomForestClassifier class
        :param n_estimators: Number of trees in the forest
        :param max_features: See DecisionTreeClassifier. Default is "sqrt" which leads to sqrt(n_features) features
        :param max_depth: See DecisionTreeClassifier
        :param min_samples: See DecisionTreeClassifier
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples = min_samples

        self.forest = []

    def fit(self, X, y):
        self.forest = []  # this allows for fitting multiple times without getting duplicate trees
        n_samples = y.size
        for _ in range(self.n_estimators):  # in every iteration of the loop, a new tree is planted
            sample = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
            X_bootstrap = X[sample, :]
            y_bootstrap = y[sample]

            tree = DecisionTreeClassifier(max_depth=self.max_depth, max_features=self.max_features, min_samples=self.min_samples)
            tree.fit(X_bootstrap, y_bootstrap)
            self.forest.append(tree)

    def predict(self, X):
        assert self.forest != []

        predictions = np.vstack([tree.predict(X) for tree in self.forest])
        return np.apply_along_axis(lambda column: np.bincount(column).argmax(), axis=0, arr=predictions)


if __name__ == '__main__':  # Test
    from sklearn.datasets import load_digits

    digits = load_digits()
    X = digits.data
    y = digits.target
    classifier = RandomForestClassifier(n_estimators=3)
    classifier.fit(X, y)

    correct = np.sum(y == classifier.predict(X))

    print(f"Out of {len(y)} observations, the random forest with {classifier.n_estimators} trees"
          f" classified {correct} correctly.")

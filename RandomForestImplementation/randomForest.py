import numpy as np

from RandomForestImplementation.decisionTree import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(self, n_estimators, max_features: int="sqrt", max_depth=np.Inf, min_samples=2, random_state=None):
        """
        Initializes the RandomForestClassifier class
        :param n_estimators: Number of trees in the forest
        :param max_features: See DecisionTreeClassifier. Default is "sqrt" which leads to sqrt(n_features) features
        :param max_depth: See DecisionTreeClassifier
        :param min_samples: See DecisionTreeClassifier
        :param random_state: Seed for the random number generators
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples = min_samples

        self.forest = []

        if random_state is not None:
            np.random.seed(random_state)

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
        """
        Gives the predicted classes for the data X
        :param X: n times p dimensional matrix (n is the number of observations and p the number of features)
        :returns: n dimensional vector where the ith element of that vector corresponds to the predicted class of the ith row of X
        """
        if self.forest is None:  # forest hasn't been fitted yet
            raise Exception("This forest hasn't been fitted yet")

        n, p = X.shape
        predictions = np.zeros(n)
        for k in range(n):
            row = X[[k], :]  # I am using [k] instead of k so that I get a 1 times k matrix instead of a vector
            prediction = np.vstack([np.reshape(tree.predict_proba(row), -1) for tree in self.forest])
            preferred_prediction = np.apply_along_axis(np.average, axis=0, arr=prediction).argmax()

            predictions[k] = preferred_prediction

        return predictions.astype(int)

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
        return {"n_estimators": self.n_estimators, "max_depth": self.max_depth, "max_features": self.max_features, "min_samples": self.min_samples}

    def set_params(self, **params):
        """
        Sets parameters for this classifier
        Note: This function is required for compatibility with sklearns cross_validation utilities.
        """
        for key, value in params.items():
            setattr(self, key, value)

        return self

if __name__ == '__main__':  # Test
    from sklearn.datasets import load_digits

    digits = load_digits()
    X = digits.data
    y = digits.target
    classifier = RandomForestClassifier(n_estimators=10)
    classifier.fit(X, y)

    correct = np.sum(y == classifier.predict(X))

    print(f"Out of {len(y)} observations, the random forest with {classifier.n_estimators} trees"
          f" classified {correct} correctly.")

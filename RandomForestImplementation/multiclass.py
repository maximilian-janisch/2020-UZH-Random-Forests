import numpy as np


class OneVsOneClassifier:  # the purpose of this class is similar to the one from sklearn
    def __init__(self, estimator, n_classes, random_state=None, *args, **kwargs):
        """
        Initializes the OvO classifier class
        :param estimator: Estimator to use (for example random forest)
        :param n_classes: Number of classes
        :param args: Arguments to pass to the init of estimator
        :param kwargs: Keyword arguments to pass to the init of estimator
        """
        self.estimator = estimator
        self.args = args
        self.kwargs = kwargs

        if random_state:
            np.random.seed(random_state)

        self.pairs = [(i, j) for i in range(n_classes) for j in range(i+1, n_classes)]

        self.classifiers = []
        self.relablers = []

    def fit(self, X, y):
        """Fits n*(n-1)/2 estimators on each of the classes in order to predict multilabel y from X."""
        for pair in self.pairs:
            relabler = self.get_relabel(y[np.isin(y, pair)])
            self.relablers.append(relabler)
            y_transformed = np.array([relabler[e] for e in y[np.isin(y, pair)]])  # transform the labels in order to avoid gaps in the numbering

            classifier = self.estimator(*self.args, **self.kwargs)
            classifier.fit(X[np.isin(y, pair), :], y_transformed)
            self.classifiers.append(classifier)

    def predict(self, X):
        """
        Gives the predicted classes for the data X
        :param X: n times p dimensional matrix (n is the number of observations and p the number of features)
        :returns: n dimensional vector where the ith element of that vector corresponds to the predicted class of the ith row of X
        """
        predictions = []

        for k, pair in enumerate(self.pairs):
            classifier = self.classifiers[k]
            relabler = self.relablers[k]
            inverse_relabler = {y: x for x, y in relabler.items()}

            prediction = classifier.predict(X)
            prediction = np.array([inverse_relabler[pred] for pred in prediction])
            predictions.append(prediction)
        predictions = np.vstack(predictions)
        return np.apply_along_axis(lambda slice: np.bincount(slice).argmax(), axis=0, arr=predictions)

    @staticmethod
    def get_relabel(iterable):
        # "Factorizes" the data in iterable. For example [0,1,5,3,5,0] gets a mapper to [0,1,2,3,2,0]
        dct = {}
        largest = 0
        for element in iterable:
            if element in dct:
                continue
            dct[element] = largest
            largest += 1
        return dct


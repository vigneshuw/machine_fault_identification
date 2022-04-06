from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.ensemble import IsolationForest


class MahalanobisDistanceClassifer(BaseEstimator, ClassifierMixin):

    def __init__(self, threshold_level, **kwargs):

        # Get the arguments
        self.kwargs = kwargs

        # Training threshold level
        self.threshold_level = threshold_level

        # Training parameters
        self.covariance = None
        self.inv_covariance = None
        self.mean = None
        self.trained_threshold = None

    def fit(self, X, y=None):

        # Compute the centroid of the distribution
        self.covariance = np.cov(X, rowvar=self.kwargs["rowvar"]) if "rowvar" in self.kwargs.keys() else np.cov(X)
        self.inv_covariance = np.linalg.inv(self.covariance)
        self.mean = np.mean(X, axis=0)

        # Determine the threshold
        training_distances = self.__distance_distribution(X, self.mean, self.inv_covariance)
        self.trained_threshold = self.__compute_threshold(training_distances, level=self.threshold_level)

    def predict(self, X, y=None, distance_type="mahalanobis"):

        # Compute the distance
        if distance_type == "mahalanobis":
            distances = self.__distance_distribution(X, self.mean, self.inv_covariance)
            # Make predictions
            distances = np.array(distances)
            predictions = np.where(distances < self.trained_threshold, 0, 1)

        else:
            raise Exception("Distance metric not implemented")

        return predictions

    @staticmethod
    def __compute_threshold(distances, level):
        # Compute mean and STD
        normal_distances_mean = np.mean(distances)
        normal_distances_std = np.std(distances)

        return normal_distances_mean + (level * normal_distances_std)

    @staticmethod
    def __distance_metric(X, mean, inv_cov, metric_type="mahalanobis"):

        assert len(X.shape) == 2, "The X for prediction must be an array, and not a vector"

        # Distance metric
        if metric_type == "mahalanobis":
            # difference
            difference = (X - mean).T

            return np.sqrt(difference.T.dot(inv_cov).dot(difference))

        else:
            raise Exception("Metric not defined")

    def __distance_distribution(self, X, mean, inv_cov, metric_type="mahalanobis"):

        dd = []
        for index, item in enumerate(X):
            distance = self.__distance_metric(item[np.newaxis, :], mean, inv_cov, metric_type=metric_type).squeeze()
            dd.append(distance.tolist())

        return dd

    def compute_distributions(self, X):

        # Compute the distances for different data
        distances = self.__distance_distribution(X, self.mean, self.inv_covariance)

        return distances


class KDEAnomalyDetector(KernelDensity):

    def __init__(self, quantile_threshold, **kwargs):
        super(KDEAnomalyDetector, self).__init__(**kwargs)

        # Thresholds
        self.trained_threshold = None
        self.quantile_threshold = quantile_threshold

    def fit(self, X, y=None, sample_weight=None):

        # Fit the super class to the data
        super(KDEAnomalyDetector, self).fit(X, y, sample_weight)

        # Get the scores for the trained case
        normal_scores = super(KDEAnomalyDetector, self).score_samples(X)
        # Compute threshold from normal scores
        self.trained_threshold = np.quantile(normal_scores, q=self.quantile_threshold)

    def predict(self, X):

        # Score the sample using the super class
        scores = super(KDEAnomalyDetector, self).score_samples(X)
        # Relative to threshold - make predictions
        predictions = np.where(scores < self.trained_threshold, 1, 0)

        return predictions
    

class IsolationForestClassifier(IsolationForest):           # noqa
    
    def __init__(self, **kwargs):
        # Initialize the super class
        super(IsolationForestClassifier, self).__init__(**kwargs)
          
    def predict(self, X):
        
        # Make predictions
        predictions = super(IsolationForestClassifier, self).predict(X)

        # Convert to 0-1
        predictions = np.where(predictions == 1, 0, 1)

        return predictions

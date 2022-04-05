import copy
import sys

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity

from lib import model_evaluations


class Models:

    def __init__(self, models_dict=None):

        # Initialize model dictionary
        if models_dict is None:
            self.model_dict = {}
        else:
            self.model_dict = copy.deepcopy()
        # After training
        self.trained_model_dict = {}
        # List of available models
        self.model_list = ["LogisticRegression", "DecisionTreeClassifier", "KNeighborsClassifier", "BaggingClassifier",
                           "RandomForestClassifier", "SVC", "MultinomialNB", "GridSearchCV"]

        # Scores
        self.scores = {}
        self.cv_scores = {}

        # Chosen models
        self.best_model = None
        self.best_model_name = None
        self.best_score = None
        # Hyperparameters optimization
        self.hyper_opt_models = {}
        self.hyper_opt_model_scores = {}
        self.hyper_opt_model_params = {}

        # Status flag
        self.is_trained = False
        self.is_initialized = False

    def load_models_dict(self, model_dict):

        self.model_dict = model_dict
        self.is_initialized = True

    def create_models(self, kwargs):

        # Ensure the keys are valid
        for key in kwargs.keys():
            assert key in self.model_list, "The model key is unknown"

        # Linear models
        # Logistic Regression
        if "LogisticRegression" in kwargs.keys():
            self.model_dict["LogisticRegression"] = LogisticRegression(**kwargs['LogisticRegression'])

        # Non-Linear models
        # DT
        if "DecisionTreeClassifier" in kwargs.keys():
            self.model_dict["DecisionTreeClassifier"] = DecisionTreeClassifier(**kwargs["DecisionTreeClassifier"])

        # KNN
        if "KNeighborsClassifier" in kwargs.keys():
            self.model_dict["KNeighborsClassifier"] = KNeighborsClassifier(**kwargs["KNeighborsClassifier"])

        # SVM
        if "SVC" in kwargs.keys():
            self.model_dict["SVC"] = SVC(**kwargs["SVC"])

        # Ensemble
        # Bagging Classifier
        if "BaggingClassifier" in kwargs.keys():
            self.model_dict["BaggingClassifier"] = BaggingClassifier(**kwargs["BaggingClassifier"])

        # Random Forest Classifier
        if "RandomForestClassifier" in kwargs.keys():
            self.model_dict["RandomForestClassifier"] = RandomForestClassifier(**kwargs["RandomForestClassifier"])

        # Multinomial NB
        if "MultinomialNB" in kwargs.keys():
            self.model_dict["MultinomialNB"] = MultinomialNB(**kwargs["MultinomialNB"])

        # Set the initialization flag
        self.is_initialized = True

    def train_models(self, X_train, y_train, verbose=0):
        """
        Train models and save them in a trained_model_dict

        :param X_train: The X data for the model. Not necessarily standardized
        :param y_train: The y data for the model. Not necessarily standardized
        :param verbose: If more details required on the models being trained
        :return: None
        """

        # train one after the other
        for model_name in self.model_dict.keys():

            # Fit the model
            clf = self.model_dict[model_name].fit(X_train, y_train)

            # Add a trained model to a dictionary
            self.trained_model_dict[model_name] = clf

            if verbose:
                sys.stdout.write(f"|| Trained - {model_name} ||\n")

        # Completed training
        self.is_trained = True

    @staticmethod
    def summarize_cv_results(cv_scores):
        # Summarize results
        cv_results_summary = {}
        for index, model_name in enumerate(cv_scores.keys()):
            data = cv_scores[model_name]

            # Get average
            average = data.mean(axis=1)
            # Get the std
            std = data.std(axis=1)
            # Get the min
            minimum = data.min(axis=1)
            # Get the max
            maximum = data.max(axis=1)
            # Count
            count = data.count(axis=1)

            # Get the dataframe
            summary_df = pd.concat([average, std, minimum, maximum, count], axis=1)
            summary_df.columns = ["average", "std", "min", "max", "count"]

            # Store
            cv_results_summary[model_name] = summary_df

        return cv_results_summary

    def train_models_cvfolds(self, X, y, kfolds=10, summarize_results=False, standardize=False):

        """
        Train the models using kfold CV, store the best model, and return the resulting metrics for the CV-folds

        :param X: Un-normalized X of the data
        :param y: The actual values for the corresponding X
        :param kfolds: Number of folds
        :param summarize_results: Flag to summarize the results if required and return them
        :param standardize: Flag to check if standardization is required
        :return: summarized or not-summarized results of the cv-folds
        """

        if self.is_trained:
            sys.stdout.write("WARN! Resetting the previously trained models\n")
            self.trained_model_dict = {}
            self.is_trained = False

        # Get the KFold
        kf = StratifiedKFold(n_splits=kfolds, shuffle=True)

        # Get a count of the labels in dataset
        unique_labels = np.unique(y)
        minority_labels = unique_labels[1:].tolist()
        # Evaluation arguments
        kwargs = {
            "accuracy_score": {},
            "balanced_accuracy_score": {},
            "f1_score": {"labels": minority_labels, "average": "micro"},
            "recall_score": {"labels": minority_labels, "average": "micro"},
            "precision_score": {"labels": minority_labels, "average": "micro"},

        }

        # Clone the original models list
        cloned_models = {}
        for model_name in self.model_dict.keys():
            cloned_models[model_name] = clone(self.model_dict[model_name])

        # Score to consider
        f1_score_check = {}
        for model_name in self.model_dict.keys():
            f1_score_check[model_name] = -1

        # Run the algorithm
        for fold, (train_indices, test_indices) in enumerate(kf.split(X, y)):

            # Get the test and train data
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            # Standardize the data if required
            if standardize:
                # Fit only on the train
                scaler = preprocessing.StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                # Apply the fitted on test
                X_test = scaler.transform(X_test)

            # Track the current WIP
            print(f"Fold - {fold}", end="\r")

            # Fit the model(s)
            for model_name in cloned_models.keys():

                # Fit the model
                clf = cloned_models[model_name].fit(X_train, y_train)

                # Predict
                y_predicted = clf.predict(X_test)

                # Evaluate
                model_eval = model_evaluations.ModelEval()
                computed_metrics = model_eval.compute_all_metrics(y_test, y_predicted, kwargs)
                if model_name not in self.cv_scores.keys():
                    self.cv_scores[model_name] = {}
                self.cv_scores[model_name][fold] = computed_metrics

                # Check to find the better performing model
                if f1_score_check[model_name] < computed_metrics["f1_score"]:
                    self.trained_model_dict[model_name] = clf

                # Reset the clone model before moving on
                cloned_models[model_name] = clone(self.model_dict[model_name])

        # Convert all to a dataframe for easy analysis
        for model_name in self.cv_scores.keys():
            self.cv_scores[model_name] = pd.DataFrame(self.cv_scores[model_name])

        if summarize_results:
            cv_results_summary = self.summarize_cv_results(self.cv_scores)
            return cv_results_summary

        # Set the training flag
        self.is_trained = True

        return self.cv_scores

    def train_predict_models(self, X_train, y_train, X_test):

        if self.is_trained:
            sys.stdout.write("WARN! Resetting the previously trained models\n")
            self.trained_model_dict = {}
            self.is_trained = False

        # Dictionary of predictions
        y_predict_dict = {}

        # Train and predict
        for model_name in self.model_dict.keys():

            # Fit the model
            clf = self.model_dict[model_name].fit(X_train, y_train)

            # Predict
            y_predict = clf.predict(X_test)
            y_predict_dict[model_name] = y_predict

            # Add to trained models
            self.trained_model_dict[model_name] = clf

        self.is_trained = True
        return y_predict_dict

    def compute_score(self, X_test, y_test):

        assert self.is_trained, "The model must be trained before this stage"

        for model_name in self.model_dict.keys():
            self.scores[model_name] = self.model_dict[model_name].score(X_test, y_test)

    def get_best_model(self, constraint="max"):

        assert self.is_trained and bool(self.scores), "The model must have been trained before this and " \
                                                      "scores must have been computed"

        # get all models
        model_names = list(self.trained_model_dict.keys())
        scores = [self.scores[x] for x in model_names]

        if constraint == "max":
            best_score = max(scores)
        else:
            best_score = min(scores)
        best_index = scores.index(best_score)

        # Get the best one
        self.best_model_name = model_names[best_index]
        self.best_model = self.trained_model_dict[self.best_model_name]
        self.best_score = best_score

    def optimize_hyperparameters(self, hyperparameters, X_train, y_train, scoring=None, standardize=False):

        if scoring is None:
            scoring = "f1_micro"

        if standardize:
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)

        for model_name in hyperparameters.keys():

            # Fit the model
            clf = GridSearchCV(self.model_dict[model_name], hyperparameters[model_name], cv=10, scoring=scoring)
            clf.fit(X_train, y_train)

            # Ideal best model
            self.hyper_opt_models[model_name] = clf.best_estimator_
            self.hyper_opt_model_scores[model_name] = clf.best_score_
            self.hyper_opt_model_params[model_name] = clf.best_params_

            sys.stdout.write(f"The optimization process completed for - {model_name}\n")


########################################################################################################################
# Anomaly detection models
########################################################################################################################
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


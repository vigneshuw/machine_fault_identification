from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, log_loss, \
    recall_score


class ModelEval:

    def __init__(self):
        self.computed_metrics = {}
        # Metrics record
        self.metrics = [
            self.compute_accuracy_score,
            self.compute_balanced_accuracy_score,
            self.compute_f1_score,
            self.compute_recall_score,
            self.compute_precision_score
        ]
        self.metrics_name = [
            "accuracy_score",
            "balanced_accuracy_score",
            "f1_score",
            "recall_score",
            "precision_score"
        ]

        # Flags
        self.is_completed = False

    @staticmethod
    def compute_accuracy_score(y_true, y_pred, kwargs):

        return accuracy_score(y_true, y_pred, **kwargs)

    @staticmethod
    def compute_balanced_accuracy_score(y_true, y_pred, kwargs):

        return balanced_accuracy_score(y_true, y_pred, **kwargs)

    @staticmethod
    def compute_f1_score(y_true, y_pred, kwargs):

        return f1_score(y_true, y_pred, **kwargs)

    @staticmethod
    def compute_recall_score(y_true, y_pred, kwargs):

        return recall_score(y_true, y_pred, **kwargs)

    @staticmethod
    def compute_precision_score(y_true, y_pred, kwargs):

        return precision_score(y_true, y_pred, **kwargs)

    def compute_all_metrics(self, y_true, y_pred, kwargs):

        assert len(self.metrics) == len(kwargs.keys()), "The Arguments does not match the metric list"

        for metric, metric_name in zip(self.metrics, self.metrics_name):

            self.computed_metrics[metric_name] = metric(y_true, y_pred, kwargs[metric_name])

        # Set the flags
        self.is_completed = True
        return self.computed_metrics


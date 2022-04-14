import json
import os
import joblib
# Needs to be here fot joblib to load custom models
import anomaly_detection_models
import flask
import numpy as np
import pandas as pd
import feature_extraction


prefix = "/opt/ml"
# Get the models path
models_path = os.path.join(prefix, "models")


class PredictionService:

    # List of models - categorized
    models_dict = None

    @classmethod
    def get_models(cls):

        # Loading the model if not already loaded
        if cls.models_dict is None:

            # Model types
            model_types = ["multi_class", "anomaly_detection"]
            # Instantiate
            cls.models_dict = {
                "multi_class": {},
                "anomaly_detection": {}
            }
            for model_type in model_types:
                load_path = os.path.join(models_path, model_type)
                for model_name in os.listdir(load_path):
                    # get the model path
                    model_path = os.path.join(load_path, model_name)
                    # Open and load model
                    with open(model_path, "rb") as file_handle:
                        cls.models_dict[model_type][model_name.split(".")[0]] = joblib.load(file_handle)

        return cls.models_dict

    @classmethod
    def predict(cls, input_data):

        # Dictionary of trained classifier pipelines
        clf_dict = cls.get_models()

        # Generate predictions
        predictions = {
            "multi_class": {},
            "anomaly_detection": {}
        }
        for model_type_name, model_type in clf_dict.items():
            for model_name in model_type.keys():
                predictions[model_type_name][model_name] = str(clf_dict[model_type_name][model_name].predict(
                    input_data)[0])

        return predictions


# The app to serve the predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():

    # Container is healthy if it can load all models
    health = PredictionService.get_models() is not None

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():

    # Parse the input JSON type data
    if flask.request.content_type == "application/json":
        # With power components as the key
        request_json = flask.request.get_json()
        data = {}

        # Convert csv in dict-values to a list
        # Get the energy features present in the data - Ensures ordering is maintained
        energy_features = request_json["energyFeatures"].split(",")
        energy_features = [x.strip() for x in energy_features]
        for power_prop_name in energy_features:
            data[power_prop_name] = [float(x) for x in request_json["data"][power_prop_name].split(",")]

        # Convert to numpy array
        df = pd.DataFrame(data)
        # Convert back to an array and ensure ordering is maintained
        df = df[energy_features]
        data = df.to_numpy()

    else:
        temp_response = {"error": "This predictor only supports JSON data"}
        return flask.Response(response=json.dumps(temp_response), status=415,
                              mimetype="application/json")

    # Print the results
    print(f"Invoked with {data.shape} records")

    # TODO: Combine Feature Extraction in pipeline
    freq_args = [{"axis": 0}, {"axis": 0}, {"axis": 0, "nperseg": 15}]
    freq_time_args = [{"wavelet": "db1"}, {"wavelet": "db1"}, {"wavelet": "db1"}]
    # Apply col by col
    computed_features = []
    for col_index in range(data.shape[1]):
        computed_features += feature_extraction.compute_all_features(data[:, col_index], freq_args, freq_time_args)
    # Convert from list to numpy array
    data = np.array(computed_features)[np.newaxis, :]

    # Prediction on the data by all models
    predictions = PredictionService.predict(data)

    # Return the predictions
    return flask.Response(response=json.dumps(predictions), status=200, mimetype="application/json")

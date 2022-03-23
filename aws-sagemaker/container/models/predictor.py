import json
import os
import joblib
import pickle
import flask
import numpy as np
import pandas as pd
import feature_extraction


prefix = "/opt/ml"
# Get the models path
models_path = os.path.join(prefix, "models")
# Get the preprocessor path
preprocessors_path = os.path.join(prefix, "preprocessors")


class PredictionService:

    # List of models
    models_dict = None

    @classmethod
    def get_models(cls):
        # Loading the model if not already loaded
        if cls.models_dict is None:
            cls.models_dict = {}
            for model_name in os.listdir(models_path):
                # get the model path
                model_path = os.path.join(models_path, model_name)

                with open(model_path, "rb") as file_handle:
                    cls.models_dict[model_name.split(".")[0]] = joblib.load(file_handle)

        return cls.models_dict

    @classmethod
    def predict(cls, input_data):

        # Dictionary of trained classifiers
        clf_dict = cls.get_models()

        # Generate predictions
        predictions = {}
        for model_name in clf_dict.keys():
            predictions[model_name] = str(clf_dict[model_name].predict(input_data)[0])

        return predictions


# The app to serve the predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():

    # Container is healthy if it can load models
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

    # TODO: Maybe automate if required
    # Prediction process
    # Feature extraction
    freq_args = [{"axis": 0}, {"axis": 0}, {"axis": 0, "nperseg": 30}]
    freq_time_args = [{"wavelet": "db1"}, {"wavelet": "db1"}, {"wavelet": "db1"}]
    # Apply col by col
    computed_features = []
    for col_index in range(data.shape[1]):
        computed_features += feature_extraction.compute_all_features(data[:, col_index], freq_args, freq_time_args)
    # Convert from list to numpy array
    data = np.array(computed_features)[np.newaxis, :]

    # Normalize the data
    with open(os.path.join(preprocessors_path, "standardization_standard-scaler.pkl"), "rb") as file_handle:
        preprocessor_data = pickle.load(file_handle)
    data = preprocessor_data["object"].transform(data)

    # Make prediction
    predictions = PredictionService.predict(data)

    if request_json["model"] == "all":
        return flask.Response(response=json.dumps(predictions), status=200, mimetype="application/json")
    else:
        try:
            temp_response = {"inference": predictions[request_json["model"]]}
            return flask.Response(response=json.dumps(temp_response), status=200, mimetype="application/json")
        except KeyError:
            temp_response = {"error": "Unknown Model name used"}
            return flask.Response(response=json.dumps(temp_response), status=400, mimetype="application/json")


import sys
import os
import argparse
import yaml
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from lib import data_prep, feature_extraction, models
from sklearn.utils import shuffle
from joblib import dump
import boto3


if __name__ == "__main__":

    # Parsing the arguments
    parser = argparse.ArgumentParser(allow_abbrev=False,
                                     description="Training and uploading model(s) to S3")
    # model params yaml file - required positional argument
    parser.add_argument("Yaml", metavar="yaml", help="YAML file specifying model parameters")
    # train data location
    parser.add_argument("-d", "--data_location", default="DATA", help="The directory of the data files")
    # standardize requirement
    parser.add_argument("--standardize", action="store_true")
    # model save location
    parser.add_argument("-sl", "--save_location", default="trained_models",
                        help="Location to save the trained model")
    # aws s3 bucket name
    parser.add_argument("-sb", "--bucket_name", help="Bucket name to save model in AWS S3 bucket")

    # Parse the arguments
    args = parser.parse_args()

    # Load the yaml file containing all params
    with open(args.Yaml, "r") as file_handle:
        yaml_file_params = yaml.load(file_handle, Loader=yaml.Loader)

    ##################################################
    # Loading the data
    ##################################################

    # Base directory location
    data_loc = args.data_location
    # Segmentation
    segment_secs = yaml_file_params["segment_window"]
    # Energy params cols
    chosen_cols = yaml_file_params["energy_params"]
    # Get the file names
    file_names = []
    for val in yaml_file_params["class_file_association"].values():
        file_names += val

    # Segmentation
    segmented_data = {}
    for file_name in file_names:
        path = os.path.join(data_loc, file_name)
        temp = data_prep.segment_data(file_name=path, col_names=chosen_cols, segment_secs=segment_secs)
        # Remove the sample_time col
        temp = temp[:, 1:, :]
        segmented_data[file_name] = temp

    # Print the files loaded
    sys.stdout.write("="*80 + "\n")
    sys.stdout.write("Files loaded and segmented are: \n")
    count = 0
    for file_name in segmented_data.keys():
        sys.stdout.write(f"\tFor the file-{file_name} the shape-{segmented_data[file_name].shape}\n")
        count += 1
    sys.stdout.write(f"Total files loaded are {count}\n")

    ##################################################
    # Determine classes
    ##################################################

    # class-file association
    class_file_association = yaml_file_params["class_file_association"]

    # Associate the classes
    class_segmented_data = {}
    for class_instance in class_file_association.keys():
        for index, file_name in enumerate(class_file_association[class_instance]):

            if index == 0:
                class_segmented_data[class_instance] = segmented_data[file_name]
            else:
                class_segmented_data[class_instance] = np.append(class_segmented_data[class_instance],
                                                                 segmented_data[file_name], axis=-1)

    # Reshape the data appropriately
    for class_instance in class_segmented_data.keys():
        class_segmented_data[class_instance] = np.transpose(class_segmented_data[class_instance], (2, 1, 0))

    # Print to ensure that the files have been loaded correctly
    sys.stdout.write("="*80 + "\n")
    sys.stdout.write("Shape of each class: \n")
    for class_instance in class_segmented_data.keys():
        sys.stdout.write(f"\tThe class-{class_instance} has the shape-{class_segmented_data[class_instance].shape}\n")

    ##################################################
    # Feature extraction
    ##################################################
    class_dataset_features = {}
    for class_instance in class_segmented_data.keys():
        dataset_features = []
        for row in class_segmented_data[class_instance]:
            computed_features = []
            for col in row:
                freq_args = [{"axis": 0}, {"axis": 0}, {"axis": 0, "nperseg": 15}]
                freq_time_args = [{"wavelet": "db1"}, {"wavelet": "db1"}, {"wavelet": "db1"}]

                computed_features += feature_extraction.compute_all_features(col, freq_args=freq_args,
                                                                             freq_time_args=freq_time_args)

            # Append to a list
            dataset_features.append(computed_features)

        # Add to class instance
        class_dataset_features[class_instance] = np.array(dataset_features)

    # print the results for verification
    sys.stdout.write("=" * 80 + "\n")
    sys.stdout.write("After feature extraction process\n")
    for class_instance in class_dataset_features.keys():
        sys.stdout.write(f'\tFor the class-{class_instance} , the extracted features has the '
                         f'shape={class_dataset_features[class_instance].shape}\n')

    ##################################################
    # Generate training data
    ##################################################
    class_label_associations = yaml_file_params["class_label_associations"]
    for index, class_instance in enumerate(class_dataset_features.keys()):

        temp_X = class_dataset_features[class_instance]
        temp_y = np.repeat(class_label_associations[class_instance], temp_X.shape[0])[:, np.newaxis]

        if index == 0:
            X = temp_X
            y = temp_y
        else:
            X = np.append(X, temp_X, axis=0)
            y = np.append(y, temp_y, axis=0)

    # Shuffle the dataset
    X, y = shuffle(X, y, random_state=42)
    # To a vector format
    y = np.squeeze(y)

    # Standardize the data if required
    sys.stdout.write("=" * 80 + "\n")
    if args.standardize:
        scaler = preprocessing.StandardScaler()
        scaler = scaler.fit(X)  # Fit
        X = scaler.transform(X)
        sys.stdout.write(f"The training data has been scaled\n")

        # Saving the metrics
        metrics_info = {
            "name": "StandardScaler",
            "params": scaler.get_params(deep=True),
            "object": scaler,
            "mean_": scaler.mean_,
            "var": scaler.var_,
            "n_features_in_": scaler.n_features_in_,
            "n_samples_seen_": scaler.n_samples_seen_
        }
        metrics_save_name = "standardization_standard-scaler.pkl"

    sys.stdout.write(f"The final combined shape-{X.shape}\n")

    ##################################################
    # Model development
    ##################################################
    # No hyper-parameter optimization here
    model_params = yaml_file_params["models"]

    # CV-fold training
    # Create repo of models
    models_repo = models.Models()
    # Initialize the models
    models_repo.create_models(model_params)

    # Train the model for the entirety of the data
    sys.stdout.write("=" * 80 + "\n")
    models_repo.train_models(X, y, verbose=1)

    sys.stdout.write("Training Complete!\n")

    ##################################################
    # Saving the models
    ##################################################
    # Save the trained models
    save_location = args.save_location
    if not os.path.isdir(save_location):
        os.makedirs(save_location)
    for model_name in models_repo.trained_model_dict.keys():
        model_save_fname = os.path.join(save_location, model_name + ".joblib")
        dump(models_repo.trained_model_dict[model_name], model_save_fname)

    sys.stdout.write("=" * 80 + "\n")
    sys.stdout.write("Trained Models Saved!\n")

    # Save the preprocessing metrics as well
    if args.standardize:
        with open(os.path.join(save_location, metrics_save_name), "wb") as file_handle:
            pickle.dump(metrics_info, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
        sys.stdout.write("The preprocessing information saved!\n")

    # Uploading the files to S3 bucket
    sys.stdout.write("=" * 80 + "\n")
    bucket_name = args.bucket_name
    if bucket_name is not None:
        # Get the client
        s3 = boto3.resource("s3")
        # Upload model files
        for model_name in models_repo.trained_model_dict.keys():
            model_saved_location = os.path.join(save_location, model_name + ".joblib")
            # Get the s3 object
            s3_object = s3.Object(bucket_name, f"energy_monitoring/{model_name}/{model_name}" + ".joblib")
            result = s3_object.put(Body=open(model_saved_location, "rb"))
            res = result.get("ResponseMetadata")

            # Check the upload status
            if res.get('HTTPStatusCode') == 200:
                sys.stdout.write(f"File - {model_name} uploaded successfully!\n")
            else:
                sys.stdout.write(f"File - {model_name} upload failed!\n")

        # Save the preprocessing information as well
        s3_object = s3.Object(bucket_name, f"energy_monitoring/{metrics_save_name}")
        result = s3_object.put(Body=open(os.path.join(save_location, metrics_save_name), "rb"))
        res = result.get("ResponseMetadata")
        # Check for success
        if res.get('HTTPStatusCode') == 200:
            sys.stdout.write(f"File - {metrics_save_name} uploaded successfully!\n")
        else:
            sys.stdout.write(f"File - {metrics_save_name} upload failed!\n")


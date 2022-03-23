import sys
import os
import boto3
from boto3.dynamodb.conditions import Key
import pandas as pd
import matplotlib.pyplot as plt
import time


class DynamoDBDataLoader:

    def __init__(self, table_name, region, dynamodb=None):

        # Initialization
        self.table_name = table_name
        self.region = region
        if dynamodb is None:
            # Configure the required resource
            self.dynamodb = boto3.resource("dynamodb", region_name="us-east-2")
        else:
            self.dynamodb = dynamodb
        # Select the table to query
        self.table = self.dynamodb.Table(self.table_name)

        # Dataset
        self.data_state = "unavailable"
        self.data_type = "unavailable"
        self.data = []
        self.query_counts = 0

    def query_data(self, sample_time_range):

        assert isinstance(sample_time_range, tuple), "The sample time range should be a tuple"

        # Check of the data is already queried
        if self.data_state == "queried":
            sys.stdout.write("The data has already been queried")
            return 0

        # Query the data - The first query

        response = self.table.query(
            KeyConditionExpression=Key('device_id').eq(1) & Key('sample_time').between(sample_time_range[0],
                                                                                       sample_time_range[1]))
        # Extract data
        self.data.append(response["Items"])
        self.query_counts += response["Count"]
        # Get the last evaluated key
        if 'LastEvaluatedKey' in response.keys():
            last_evaluated_key = response['LastEvaluatedKey']
        else:
            last_evaluated_key = False
        # Wait until LastEvaluatedKey is empty
        num_queries = 1
        while last_evaluated_key:

            # Query again
            try:
                response = self.table.query(
                    KeyConditionExpression=Key('device_id').eq(1) & Key('sample_time').between(sample_time_range[0],
                                                                                               sample_time_range[1]),
                    ExclusiveStartKey=last_evaluated_key
                )
            except Exception as e:
                time.sleep(30)
                continue
            # Append data
            self.data.append(response["Items"])
            self.query_counts += response["Count"]
            # Get the last evaluated key
            if "LastEvaluatedKey" in response.keys():
                last_evaluated_key = response["LastEvaluatedKey"]
            else:
                last_evaluated_key = False

            # Print
            num_queries += 1
            print(f"Num of queries completed: {num_queries}", end="\r")

        self.data_state = "queried"
        self.data_type = "list"

    def explore_single_query(self, sample_time_range):

        assert isinstance(sample_time_range, tuple), "The sample time range should be a tuple"

        # Identify the table
        table = self.dynamodb.Table(self.table_name)

        # Query the data - The first query
        response = table.query(
            KeyConditionExpression=Key('device_id').eq(1) & Key('sample_time').between(sample_time_range[0],
                                                                                       sample_time_range[1])
        )

        return response

    def get_dataframe(self):

        assert self.data_state == "queried", "The data must be queried before this step"

        # Convert to DataFrame
        temp = []
        for page_number in range(len(self.data)):
            temp += self.data[page_number]
        temp = pd.DataFrame(temp)

        # Expand the data column
        temp_data_df = pd.DataFrame(temp["data"].tolist())
        temp[list(temp_data_df.columns)] = temp_data_df
        temp.drop(labels="data", axis="columns", inplace=True)

        self.data = temp.astype('float64')
        self.data_type = "dataframe"

    def save_dataframe(self, save_path, save_file_name):

        assert self.data_type == "dataframe", "The dataframe should have been constructed for this function to run"

        # Check for directories
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        # Modify save path to include file name
        save_path = os.path.join(save_path, save_file_name)

        # Check the path for 'DATA'
        if 'DATA' not in save_path.split(os.sep):
            sys.stdout.write("The path should contain a folder named 'DATA")
            return 1

        # Save the data
        self.data.to_csv(save_path, sep=",", header=True, index=True, index_label="no")

    def plot_data(self, col_name, axs=None, time_range=None):

        assert self.data_type == "dataframe", "The dataframe should have been constructed for this function to run"

        # Select the time range
        if time_range is None:
            time_range = range(0, 5*60)

        # Adding the axes
        if axs is None:
            fig = plt.figure(figsize=(18, 7))
            axs = fig.add_axes([0, 0, 1, 1])

        # Select the range to plot
        selected_df = self.data.iloc[time_range]
        selected_df[col_name].plot(kind="line", ax=axs, use_index=True, title=f"Plot of {col_name}", xlabel="Time (s)",
                                   ylabel=col_name)

    def query_save_data(self, sample_time_range, save_path, save_file_name):

        assert self.data_state != "queried", 'The data has already been queried'

        # Get the data
        self.query_data(sample_time_range)

        # Save the data
        self.save_dataframe(save_path, save_file_name)

    def get_queried_counts(self, verbose=0):

        assert self.data_state == "queried", "The data must be queried beforehand"

        if verbose:
            sys.stdout.write(f"The total Queried counts is {self.query_counts}\n")

    def reset_last_query(self):

        # Resetting items
        self.data_state = "unavailable"
        self.data_type = "unavailable"

        # Empty query counts
        self.query_counts = 0
        # Empty data list
        self.data = []



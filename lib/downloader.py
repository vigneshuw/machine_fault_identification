import boto3
import sys
import os
from datetime import datetime
from lib import data_loader_dynamodb


# NOTE:- Don't combine multiple queries -> One per file name
def load_save_dynamodbdata(dynamodb, table_name, file_names, time_intervals, save_dir):

    # Create only once
    data_loader = data_loader_dynamodb.DynamoDBDataLoader(table_name, "us-east-1", dynamodb=dynamodb)

    # Go through one-by-one and save the data
    for file_name, sample_time_range in zip(file_names, time_intervals):
        # Get the save path
        if len(file_name.split(".")) > 1:
            save_loc = os.path.join(save_dir, file_name)
        else:
            save_loc = os.path.join(save_dir, file_name + ".csv")

        query_count = 0

        # Query data
        data_loader.query_data(sample_time_range)
        # Convert to dataframe
        data_loader.get_dataframe()
        # Get the data
        data = data_loader.data
        # get the current query count
        query_count += data_loader.query_counts

        # Save the data
        data.to_csv(save_loc, sep=",", header=True, index=True, index_label="no")
        # Reset the state after a save
        data_loader.reset_last_query()

        # Print some results
        sys.stdout.write(f"For the file named {file_name}, the total query count is {query_count}\n")

    sys.stdout.write(f'The files are saved in the location {save_dir}\n')


if __name__ == "__main__":

    # Load every single file; Based on the experiments conducted
    # A list of save file names
    file_names = [
        "machine_OFF_no-error",  # When the machine was turned off at the controller
        "machine_ON_no-ref_start-error_1",  # Machine turned ON, and the parameter switch enable error - file1
        "machine_ON_no-ref_start-error_2",  # Machine turned ON, and the parameter switch enable error - file2
        "machine_ON_ref_no-error",  # Machine ON referenced and no-error idling
        "machine_ON_ref_overtravel-error_x_neg",  # Machine ON referenced and Overtravel for X negative
        "machine_ON_ref_overtravel-error_x_pos",  # Machine ON referenced and Overtravel for X positive
        "machine_ON_ref_overtravel-error_y_neg",  # Machine ON referenced and Overtravel for Y negative
        "machine_ON_ref_overtravel-error_y_pos",  # Machine ON referenced and Overtravel for Y positive
        "machine_ON_ref_overtravel-error_z_neg",  # Machine ON referenced and Overtravel for Z negative
    ]
    # Time intervals to query for the appropriate file names
    time_intervals = [
        # "machine_OFF_no-error"
        (
            int(datetime(2022, 2, 25, 11, 15).timestamp()), int(datetime(2022, 2, 25, 13, 30).timestamp())
        ),
        # "machine_ON_no-ref_start-error_1"
        (
            int(datetime(2022, 2, 25, 13, 45).timestamp()), int(datetime(2022, 2, 25, 14, 8).timestamp())
        ),
        # "machine_ON_no-ref_start-error_2"
        (
            int(datetime(2022, 2, 25, 14, 25).timestamp()), int(datetime(2022, 2, 25, 16, 30).timestamp())
        ),
        # "machine_ON_ref_no-error"
        (
            int(datetime(2022, 2, 25, 17, 00).timestamp()), int(datetime(2022, 2, 25, 21, 45).timestamp())
        ),
        # "machine_ON_ref_overtravel-error_x_neg"
        (
            int(datetime(2022, 2, 25, 22, 15).timestamp()), int(datetime(2022, 2, 26, 11, 45).timestamp())
        ),
        # "machine_ON_ref_overtravel-error_x_pos"
        (
            int(datetime(2022, 2, 26, 12, 15).timestamp()), int(datetime(2022, 2, 26, 18, 00).timestamp())
        ),
        # "machine_ON_ref_overtravel-error_y_neg"
        (
            int(datetime(2022, 2, 26, 18, 45).timestamp()), int(datetime(2022, 2, 27, 12, 00).timestamp())
        ),
        # "machine_ON_ref_overtravel-error_y_pos"
        (
            int(datetime(2022, 2, 27, 12, 30).timestamp()), int(datetime(2022, 2, 27, 18, 00).timestamp())
        ),
        # "machine_ON_ref_overtravel-error_z_neg"
        (
            int(datetime(2022, 2, 27, 20, 30).timestamp()), int(datetime(2022, 2, 28, 8, 30).timestamp())
        )
    ]

    # Dynamodb configuration
    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    # Save directory location
    save_dir = os.path.join(os.path.dirname(os.getcwd()), "DATA")

    load_save_dynamodbdata(dynamodb=dynamodb, table_name="robonano1_energy_wn", file_names=file_names,
                           time_intervals=time_intervals, save_dir=save_dir)

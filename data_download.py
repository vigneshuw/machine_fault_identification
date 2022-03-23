import boto3
import time
import argparse
from lib import downloader


if __name__ == "__main__":

    # Parsing the arguments
    parser = argparse.ArgumentParser(allow_abbrev=False, description="Extract data from DynamoDB for a specified time range")
    # Start time
    parser.add_argument("-st", "--start_time", type=str, nargs="*",
                        required=True, help="Start time in the format Y/m/d-H:M:S")
    # End time
    parser.add_argument("-et", "--end_time", type=str, nargs="*",
                        required=True, help="End time in the format Y/m/d-H:M:S")
    # Save File name
    parser.add_argument("-fn", "--save_file_name", type=str, nargs="*",
                        required=True, help="Name for the file to be saved")
    # Save file location
    parser.add_argument("-sl", "--save_location", type=str, nargs=1,
                        required=True, help="Location to save the data")
    # Parse arguments
    args = parser.parse_args()
    # Simple validator
    assert len(args.start_time) == len(args.end_time) == len(args.save_file_name),  "Incorrect number of arguments " \
                                                                                    "between items"

    # Save file name
    file_names = args.save_file_name
    # Save directory
    save_dir = args.save_location[0]

    # Saved file's time interval
    time_intervals = []
    for (st, et) in zip(args.start_time, args.end_time):

        # Get the time interval
        temp_time_interval = (
            (int(time.mktime(time.strptime(st, '%Y/%m/%d-%H:%M:%S'))),
             int(time.mktime(time.strptime(et, '%Y/%m/%d-%H:%M:%S'))))
        )

        time_intervals.append(temp_time_interval)

    # Dynamodb configuration
    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")

    downloader.load_save_dynamodbdata(dynamodb, "robonano1_energy_wn", file_names, time_intervals, save_dir)

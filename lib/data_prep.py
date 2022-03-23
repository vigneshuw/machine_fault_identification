import numpy as np
import pandas as pd


def segment_data(file_name, col_names, segment_secs):

    assert isinstance(col_names, list), "The col_names should be a tuple"

    # Read the data
    cols_to_read = ["no", "sample_time"] + col_names
    df = pd.read_csv(file_name, sep=",", header="infer", index_col="no", usecols=cols_to_read)

    # Get the split indices and configuration
    df_length_secs = df.count()[0]
    split_indices = np.arange(segment_secs, df_length_secs, segment_secs)

    # Split the array
    df_array = df.to_numpy()
    split_arrays = np.split(df_array, split_indices, axis=0)
    # Remove the last segment if unequal
    if not split_arrays[-1].shape[0] == segment_secs:
        split_arrays.pop(-1)
    # Stack the arrays back
    segmented_data = np.stack(split_arrays, axis=2)

    return segmented_data






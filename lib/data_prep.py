import numpy as np
import pandas as pd


def segment_data(file_name, col_names, segment_secs, overlap_rate=0.0):

    assert isinstance(col_names, list), "The col_names should be a tuple"
    assert overlap_rate < 1.0, "The overlap rate cannot be 1.0 or greater than 1.0"

    # Read the data
    cols_to_read = ["no", "sample_time"] + col_names
    # !!! This does not arrange columns for you
    df = pd.read_csv(file_name, sep=",", header="infer", index_col="no", usecols=cols_to_read)
    # Ensure the column ordering - very important
    # sample time is included as they will be removed later
    df = df[["sample_time"] + col_names]

    # Get the split indices and configuration
    df_length_secs = df.count()[0]
    increments = int((1 - overlap_rate) * segment_secs)
    seg_starting_indices = np.arange(0, df_length_secs, increments)

    # Split the array
    df_array = df.to_numpy()
    split_arrays = [df_array[np.s_[start:start+segment_secs]] for start in seg_starting_indices
                    if start + segment_secs < df_length_secs] # One less
    # Stack the arrays back
    segmented_data = np.stack(split_arrays, axis=2)

    return segmented_data







import os
import shutil
import numpy as np
import csv
import glob
import pandas as pd
from scipy import signal
from scipy import interpolate


def Downsampled(root_directory, old_freq, new_freq):
    
    original_sampling_rate = old_freq   # Hz
    desired_sampling_rate = new_freq    # Hz
    # Desired downsampling factor
    downsampling_factor = int(original_sampling_rate / desired_sampling_rate)
    
    # Iterate through all subfolders and CSV files
    for current_folder, subfolders, files in os.walk(root_directory):
        for input_file in files:
            if input_file.endswith('.csv'):
                file_path = os.path.join(current_folder, input_file)
        
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path)
                
                columns = df.columns[1:]
                data = df.iloc[:, 1:].values.astype(float)
                time = df.iloc[:, 0].values
                
                # Downsampling process
                # Calculate the new time interval
                old_time_interval = 1 / original_sampling_rate
                new_time_interval = 1 / desired_sampling_rate
                
                # Perform downsampling
                downsampled_data = data[::downsampling_factor]
                downsampled_time = time[::downsampling_factor]
                
                # Create a new DataFrame with downsampled data and time
                downsampled_columns = columns.insert(0, 'time')
                combined_downsampled_data = np.hstack((downsampled_time.reshape(-1, 1), downsampled_data))
                df_downsampled = pd.DataFrame(combined_downsampled_data, columns=downsampled_columns)
                
                # Get the file name without extension
                file_name = os.path.splitext(input_file)[0]
                
                # Create the destination path for the new CSV file
                downsampled_file_path = os.path.join(current_folder, file_name + '.csv')
                
                # Save the downsampled data into a new CSV file
                df_downsampled.to_csv(downsampled_file_path, index=False)
                
                return

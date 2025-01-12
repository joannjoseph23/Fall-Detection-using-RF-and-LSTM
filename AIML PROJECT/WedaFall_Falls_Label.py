

import os
import pandas as pd
import numpy as np

# Root directory where the folders are located
root_directory = '/Users/nityareddy/Desktop/AIML PROJECT/processed'  # <- Change this path if needed

# List of activity names of interest
activities_of_interest = ['Activity13', 'Activity14', 'Activity15', 'Activity16', 
                          'Activity17', 'Activity18', 'Activity19', 'Activity20']

# Index of the column to modify (assuming the first column has index 0)
column_to_modify = "Tag"

# Function to detect abrupt changes
def detect_abrupt_changes(data, threshold):
    changes = []
    for i in range(1, len(data)):
        diff = abs(data[i] - data[i - 1])
        if diff > threshold:
            changes.append(i)
    return changes

def find_max_change_index(data, changes):
    max_index = max(changes, key=lambda i: abs(data[i] - data[i - 1]))
    return max_index

# Function to find the start and end indices of the window
def find_window_indices(data, threshold=0.5, window_duration=3.0):
    accel_x = data['Accelerometer: x-axis (g)']
    accel_y = data['Accelerometer: y-axis (g)']
    accel_z = data['Accelerometer: z-axis (g)']

    changes_x = detect_abrupt_changes(accel_x, threshold)
    changes_y = detect_abrupt_changes(accel_y, threshold)
    changes_z = detect_abrupt_changes(accel_z, threshold)

    start_x = find_max_change_index(accel_x, changes_x)
    start_y = find_max_change_index(accel_y, changes_y)
    start_z = find_max_change_index(accel_z, changes_z)

    start_window = int(np.mean([start_x, start_y, start_z])) if changes_x and changes_y and changes_z else 0

    # Define the time of the largest change
    time_of_max_change = data['TimeStamp'][start_window]
    # Define window start time 1 second before the largest change
    window_start_time = time_of_max_change - 1
    # Define window end time 1.5 seconds after the largest change
    window_end_time = time_of_max_change + 1.5

    start_idx = data.index[data['TimeStamp'] >= window_start_time]
    end_idx = data.index[data['TimeStamp'] >= window_end_time]

    # Handle cases where start or end indices are not found
    start_idx = start_idx[0] if len(start_idx) > 0 else 0
    end_idx = end_idx[0] if len(end_idx) > 0 else len(data) - 1

    return start_idx, end_idx

# Function to change values in the specified column within the window to 1
def modify_window_values(data, start_idx, end_idx, column):
    modified_data = data.copy()
    modified_data.loc[start_idx:end_idx, column] = "1"
    return modified_data

# Function to search for CSV files in the directories of interest
def search_csv_files(directory):
    csv_files = []
    for current_folder, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                for activity in activities_of_interest:
                    if activity in file:
                        full_path = os.path.join(current_folder, file)
                        csv_files.append(full_path)
                        break
    return csv_files

# Search for CSV files in the specified root directory
csv_files = search_csv_files(root_directory)

# Process each CSV file one by one
for csv_file in csv_files:
    try:
        # Load data into a DataFrame
        data = pd.read_csv(csv_file)

        if not data.empty:
            if any(activity in csv_file for activity in activities_of_interest):
                # Find the start and end indices of the window
                start_idx, end_idx = find_window_indices(data)

                # Modify the values in the specified column within the window
                modified_data = modify_window_values(data, start_idx, end_idx, column_to_modify)

                # Save the modified data to the original file
                modified_data.to_csv(csv_file, index=False)
            else:
                print(f"Activity not of interest: {csv_file}")
        else:
            print(f"Empty DataFrame in file: {csv_file}")

    except FileNotFoundError:
        print(f"File not found: {csv_file}")
    except Exception as e:
        print(f"Error processing file {csv_file}: {str(e)}")

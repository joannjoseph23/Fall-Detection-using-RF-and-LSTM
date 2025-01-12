

import os
import csv
from WedaFall_Downsampling import Downsampled

def process_WEDA(current_folder, new_folder):
    
    # Dictionary to store subjects, trials, and "accel" and "gyro" files
    subject_data = {}

    # Iterate through all subfolders in the current folder
    for activity in os.listdir(current_folder):
        activity_folder = os.path.join(current_folder, activity)
        
        activity_id = None
        activity_name = activity
        
        if activity_name == 'D01':
            activity_id = 1
        elif activity_name == 'D02':
            activity_id = 2
        elif activity_name == 'D03':
            activity_id = 3
        elif activity_name == 'D04':
            activity_id = 4
        elif activity_name == 'D05':
            activity_id = 5
        elif activity_name == 'D06':
            activity_id = 6
        elif activity_name == 'D07':
            activity_id = 7
        elif activity_name == 'D08':
            activity_id = 8
        elif activity_name == 'D09':
            activity_id = 9
        elif activity_name == 'D10':
            activity_id = 10
        elif activity_name == 'D11':
            activity_id = 11
        elif activity_name == 'D12':
            activity_id = 12
        elif activity_name == 'F01':
            activity_id = 13
        elif activity_name == 'F02':
            activity_id = 14
        elif activity_name == 'F03':
            activity_id = 15
        elif activity_name == 'F04':
            activity_id = 16
        elif activity_name == 'F05':
            activity_id = 17
        elif activity_name == 'F06':
            activity_id = 18
        elif activity_name == 'F07':
            activity_id = 19
        elif activity_name == 'F08':
            activity_id = 20
        
        # Check if it's a folder
        if os.path.isdir(activity_folder):
            # Create a new folder to store grouped files
            os.makedirs(new_folder, exist_ok=True)
        
            # Iterate through all CSV files in the current subfolder
            for file_name in os.listdir(activity_folder):
                if file_name.endswith('.csv') and 'accel' in file_name:
                    # Extract information from the file name
                    subject = file_name.split('_')[0][1:].lstrip('0')
                    trial = file_name.split('_')[1][2]
            
                    # Check if there is a corresponding gyroscope file
                    gyro_file_name = file_name.replace('accel', 'gyro')
                    if gyro_file_name in os.listdir(activity_folder):
                        subject = file_name.split('_')[0][1:].lstrip('0')
                        trial = file_name.split('_')[1][2]
                        
                        # Define the name for the combined file
                        new_name = f'WEDAFALL_Subject{subject}Activity{activity_id}Trial{trial}.csv'
                        # Create folder structure in the new folder
                        new_subject_folder = os.path.join(new_folder, f'Subject{subject}')
                        new_activity_folder = os.path.join(new_subject_folder, f'Activity{activity_id}')
                        new_trial_folder = os.path.join(new_activity_folder, f'Trial{trial}')
                        os.makedirs(new_trial_folder, exist_ok=True)

                        new_path = os.path.join(new_trial_folder, new_name)

                        with open(os.path.join(activity_folder, file_name), 'r') as accel_file, \
                                open(os.path.join(activity_folder, gyro_file_name), 'r') as gyro_file, \
                                open(new_path, 'w', newline='') as combined_file:
            
                            accel_reader = csv.reader(accel_file)
                            gyro_reader = csv.reader(gyro_file)
                            combined_writer = csv.writer(combined_file)
            
                            # Get headers from the acceleration and gyroscope files
                            accel_header = next(accel_reader)
                            gyro_header = next(gyro_reader)
            
                            # Identify columns of interest (excluding the first column)
                            accel_columns = accel_header[1:]
                            gyro_columns = gyro_header[1:]
            
                            # Write the header to the combined file
                            combined_header = ['Value'] + accel_columns + gyro_columns
                            combined_writer.writerow(combined_header)
            
                            # Read data from the acceleration and gyroscope files
                            accel_data = list(accel_reader)
                            gyro_data = list(gyro_reader)
            
                            # Check the data size
                            data_length = min(len(accel_data), len(gyro_data))
            
                            # Compare values and write combined rows to the combined file
                            for i in range(data_length):
                                accel_row = accel_data[i]
                                gyro_row = gyro_data[i]
            
                                # Get values from the first columns
                                accel_value = float(accel_row[0])
                                gyro_value = float(gyro_row[0])
            
                                # Determine the column order in the combined file
                                if accel_value >= gyro_value:
                                    combined_row = [accel_value] + accel_row[1:] + gyro_row[1:]
                                else:
                                    combined_row = [gyro_value] + accel_row[1:] + gyro_row[1:]
            
                                combined_writer.writerow(combined_row)
         
    Downsampled(new_folder, 50, 18)
    print("Downsampling complete")
    
    process_to_up(new_folder)
    print("Processing completed")


def process_to_up(output_folder):
    
    # Root directory where the CSV files are located
    root_directory = output_folder
    
    # Iterate through all subfolders and CSV files
    for current_folder, subfolders, files in os.walk(root_directory):
        for input_file in files:
            if input_file.endswith('.csv'):
                file_path = os.path.join(current_folder, input_file)
                file_name = os.path.splitext(input_file)[0]
                
                # Extract data from the file name
                parts = file_name.split("WEDAFALL_Subject")[1].split("Activity")
                subject = int(parts[0])
                activity, trial = map(int, parts[1].split("Trial"))    
    
                # Extract tag based on activity
                if 12 <= activity <= 19:
                    # To be done later; initialize with "0" since a normal activity precedes a fall
                    tag = "0"                       
                else:
                    tag = "0"
                    
                # Define the output file path for the selected file
                output_file = os.path.join(current_folder, file_name + ".csv")
                
                # Initialize the list to store selected rows
                selected_rows = []
                
                # Read the CSV file into a list of rows
                with open(file_path, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    rows = list(csv_reader)
            
                    # Create a new header with existing columns plus "Subject" and "Trial" columns
                    new_header = ['TimeStamp', 'Accelerometer: x-axis (g)', 'Accelerometer: y-axis (g)', 
                                  'Accelerometer: z-axis (g)', 'Gyroscope: x-axis (rad/s)', 'Gyroscope: y-axis (rad/s)',
                                  'Gyroscope: z-axis (rad/s)', 'Subject', 'Activity', 'Trial', 'Tag']
                    
                    for original_row in rows[1:]:
                        row = original_row.copy()  # Copy the original row to avoid modifying the original list
                        # Convert values from the second to fourth columns from m/s^2 to g
                        for i in range(1, 4):
                            value_m_s2 = float(row[i])
                            value_g = value_m_s2 / 9.81
                            row[i] = value_g
                        
                        # Add "Subject", "Activity", "Trial", and "Tag" columns to the row
                        row = row[:7] + [subject, activity, trial, tag] 
                        
                        # Add the modified row to the list of selected rows
                        selected_rows.append(row)
                    
                # Save the selected rows into a new CSV file
                with open(output_file, 'w', newline='') as output_csv:
                    csv_writer = csv.writer(output_csv)
                    # Write the new header
                    csv_writer.writerow(new_header)
                    csv_writer.writerows(selected_rows)
             

def main():      
    
    # Path to the directory where the WEDA Fall dataset is located
    input_folder = '/Users/nityareddy/Desktop/AIML PROJECT/50Hz copy'
   
    # Path to the directory where we want to save the datasets
    output_folder = '/Users/nityareddy/Desktop/AIML PROJECT/processed'
    process_WEDA(input_folder, output_folder)

if __name__ == "__main__":
    main()

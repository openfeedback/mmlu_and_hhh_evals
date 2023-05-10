import csv
import os
import glob
import argparse

def modify_header(input_file, output_file):
    with open(input_file, mode='r', newline='') as infile:
        reader = csv.reader(infile)

        # Extract the header row and modify it
        header = next(reader)
        header[6] = f"{header[6].split('/')[-2]}/{header[6].split('/')[-1]}"
        header[7] = f"{header[6].split('/')[-2]}/{header[6].split('/')[-1]}"
        header[8] = f"{header[6].split('/')[-2]}/{header[6].split('/')[-1]}"
        header[9] = f"{header[6].split('/')[-2]}/{header[6].split('/')[-1]}"
        # header[1] = 'NewColumnName2'

        # Store the rest of the rows
        rows = [row for row in reader]

    # Write the modified contents to a new CSV file or overwrite the original file
    with open(output_file, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)

        # Write the modified header and the rest of the rows
        writer.writerow(header)
        writer.writerows(rows)

# Create the argument parser
parser = argparse.ArgumentParser(description="Modify the header of all CSV files in a directory")
parser.add_argument("--directory", help="Path to the directory containing the CSV files")

# Parse the command-line arguments
args = parser.parse_args()

# Get the directory from the command-line arguments
directory = args.directory


# Find all CSV files in the directory
csv_files = glob.glob(os.path.join(directory, '*.csv'))

# Process each CSV file
for input_file in csv_files:
    # Create an output file name based on the input file name
    # output_file = os.path.splitext(input_file)[0] + '_modified.csv'

    # Modify the header of the input CSV file and save it as a new output file
    modify_header(input_file, input_file)


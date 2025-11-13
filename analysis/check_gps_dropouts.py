import glob
import pynmea2 

def is_bad_gps_value(gps_file):
    """
    Check if a NMEA sentence has bad GPS values.
    You can define 'bad' based on specific criteria, e.g., checksum failures, missing data, etc.
    """
    bad_sentences = 0
    total_sentences = 0
    with open(gps_file, encoding="utf8", errors='ignore') as file:
        for line in file:
            if "GPGGA" in line:
                total_sentences += 1
                #check to see if we have lost GPS fix
                try:
                    gpgga = pynmea2.parse(line)   #grab gpgga sentence and parse
                    if gpgga.gps_qual < 1:
                        bad_sentences += 1
                except:
                    continue
    return total_sentences, bad_sentences

def read_file_with_fallback(file_path):
    """
    Try reading a file with different encodings and return lines if successful.
    """
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'ascii']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as file:
                return file.readlines()
        except (UnicodeDecodeError, ValueError):
            continue
    # If all encodings fail, return an empty list or handle accordingly
    return []

def calculate_bad_gps_percentage(folder_path):
    """
    Calculate the percentage of bad GPS values in all NMEA text files in the specified folder and its subfolders.
    :param folder_path: Path to the folder containing NMEA text files.
    :return: Percentage of bad GPS values.
    """
    total_sentences = 0
    bad_sentences = 0

    # Loop through each directory to access data from mission
     # Define microSWIFT num
    mission_list = glob.glob(folder_path + 'mission*')
    progress_counter = 0
    total_missions = len(mission_list)
    for mission in mission_list:
        microSWIFT_dir_list = glob.glob(mission + '/microSWIFT_*')
        for microSWIFT_dir in microSWIFT_dir_list:

            # Check if there are both IMU and GPS files in the microSWIFT directory
            gps_file_list = glob.glob(microSWIFT_dir + '/*GPS*.dat')
            if (len(gps_file_list) > 0):
                for file in gps_file_list:
                    total_sentences_in_file, bad_sentences_in_file = is_bad_gps_value(file)
                    total_sentences += total_sentences_in_file
                    bad_sentences += bad_sentences_in_file

        progress_counter+=1
        print(f'{progress_counter/total_missions*100:.2f} % complete')
    if total_sentences == 0:
        return 0.0
    bad_percentage = (bad_sentences / total_sentences) * 100
    return bad_percentage


# Example usage:
folder_path = "/Volumes/EJR-DATA/DUNEX/microSWIFT_data/"
bad_gps_percentage = calculate_bad_gps_percentage(folder_path)
print(f"Total percentage of bad GPS values: {bad_gps_percentage:.2f}%")
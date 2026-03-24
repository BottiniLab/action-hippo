# script to retrieve eye data from csv file

# copies eye data from saved location to BIDS folder
# loads eye data, cleans and downsamples and saves with the suffix _cleaned

# =============================================================================



import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from copy import deepcopy

# define the path to the folder containing the eye tracking data
path = '/data/pt_02747_hippo/1_day_one/2_eyetracker/eye_data/*'

# define the path to the folder where we want to store the eye tracking data in BIDS format
path_bids = '/data/pt_02747/action_hippo/data/derivatives/eyetracker/'

# define the path to the folder containing the behavioural data
path_beh = '/data/pt_02747_hippo/1_day_one/2_eyetracker/responses/trials/'

# define the path to the folder containing the response data
path_resp = '/data/pt_02747_hippo/1_day_one/2_eyetracker/responses/probe_trials/'



# get all subject names from the original data
subjects = [i for i in os.listdir(path[:-1]) if i.startswith('sub')]


moved_subjects = [i for i in os.listdir(path_bids) if i.startswith('sub')]

# get all subject names that have not yet been moved
subjects_new = [i for i in subjects if i not in moved_subjects]


subjects_new = []

# loop through all subjects
for subject in subjects_new:

    print("copying " + subject)

    # create a new folder for each subject in the BIDS folder
    os.mkdir(path_bids + subject)
    # copy all files from the original data to the new folder
    os.system('cp -r ' + path + subject + '/* ' + path_bids + subject + '/')


    # unlike with fMRI runs, we care about absolute time, not relative time
    # BUT this makes things easier as we don't need to split things into runs

    # behavioural data can be copied from the original data to the new folder
    # behavioural data is located in /data/pt_02747_hippo/1_day_one/2_eyetracker/responses/trials/

    # correct for '0' at start of sub names in eye data
    if subject.split('-')[1][0] == '0':
        subject_beh = subject.split('-')[0] + '-' + subject.split('-')[1][-1]
    else:
        subject_beh = subject

    # copy the folder structure from the original data to the new folder
    file_beh = [i for i in os.listdir(path_beh) if i.startswith(subject_beh+'_')]
    print(file_beh)

    # copy the behavioural data to the new folder
    os.system('cp -r ' + path_beh + file_beh[0] + ' ' + path_bids + subject + '/')


    file_beh = [i for i in os.listdir(path_resp) if i.startswith(subject_beh+'_')]

    os.system('cp -r ' + path_resp + file_beh[0] + ' ' + path_bids + subject + '/')

    



def get_clean_eye_data(file, path_bids_subj, buffer=100, round_dp = 3): #x is how much buffer around bad data

    # load in the eye data
    eye_data = pd.read_csv(path_bids_subj+file)

    # clean data

    # find data which is BAD
    eye_data_status_list = list(eye_data['status'])
    eye_data_status_list_shifted_past = eye_data_status_list[buffer:] + [0]*buffer
    eye_data_status_list_shifted_future = [0]*buffer + eye_data_status_list[:-buffer]

    # add a buffer of x samples around bad data
    eye_data[f'status_buffer_{buffer}'] =  [i+j+k for i,j,k in zip(eye_data_status_list, eye_data_status_list_shifted_past, eye_data_status_list_shifted_future)]

    # select only data where status_buffer is 0
    eye_data_cleaned = deepcopy(eye_data[eye_data[f'status_buffer_{buffer}'] == 0])

    # create a new column with the rounded time
    eye_data_cleaned['time_rounded'] = np.nan

    # use iloc to assign the rounded time to the new column
    eye_data_cleaned['time_rounded'] = eye_data_cleaned['time'].apply(lambda x: round(x,round_dp))

    eye_data_cleaned = eye_data_cleaned.groupby('time_rounded').mean().reset_index()

    # get gaze position for both eyes in x and y
    eye_data_cleaned['mean_gaze_x'] = (eye_data_cleaned['left_gaze_x'] + eye_data_cleaned['right_gaze_x'])/2
    eye_data_cleaned['mean_gaze_y'] = (eye_data_cleaned['left_gaze_y'] + eye_data_cleaned['right_gaze_y'])/2

    # plot and save a figure of the cleaned eye data to path_bids_subj
    plt.figure(figsize=(20,10))
    plt.plot(eye_data_cleaned['time_rounded'], eye_data_cleaned['mean_gaze_x'], label='x')
    plt.plot(eye_data_cleaned['time_rounded'], eye_data_cleaned['mean_gaze_y'], label='y')
    plt.legend()
    plt.savefig(path_bids_subj + 'eye_data_cleaned.png')


    return eye_data_cleaned


# loop through all subjects



#subjects = ['sub-23', 'sub-29', 'sub-30'] # also 11, 47, 23, 29, 30, 

for subject in subjects:

    print("cleaning " + subject)

    eye_file = [i for i in os.listdir(path_bids + subject + '/') if 'eye_data_2' in i]
    
    if len(eye_file) > 1 and len([i for i in eye_file if 'cleaned' in i]) > 0:
        print(f"{subject} has more than one eye file and a cleaned eye file already exists. Skipping.")
        continue
    
    # exclude any files with 'cleaned' in the name
    eye_file = [i for i in eye_file if 'cleaned' not in i]

    if len(eye_file) ==1:
        print(f"loading {eye_file} for {subject}")

        # get the cleaned eye data
        eye_data_cleaned = get_clean_eye_data(eye_file[0], path_bids_subj=path_bids + subject + '/', buffer=100, round_dp = 3)

        # save the cleaned eye data
        eye_data_cleaned.to_csv(path_bids + subject + '/' + eye_file[0][:-4] + '_cleaned.csv', index=False)

    else:
        print(f"{subject} has no unique eye file. Skipping.")
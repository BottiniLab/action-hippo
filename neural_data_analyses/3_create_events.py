# for preprocessed subjects, create a an events file named 'events.tsv' in the BIDS directory, of format <sub-01>_task-probe_<run-01>_events.tsv
# folder for event files is: /data/pt_02747/action_hippo/data/derivatives/sub-07/func/
# folder to get the raw data from is: /data/pt_02747_hippo/2_day_two/4_scanner/responses/trials/
# trial data saved in the format sub-7_004_trials_20230621_0804.csv

import pandas as pd
import numpy as np
import os
import glob
import json
from copy import deepcopy


dir_proj    = '/data/pt_02747/action_hippo/'
dir_data    = dir_proj+ 'data/'
dir_deriv   = dir_data + 'derivatives/'

dir_responses = '/data/pt_02747_hippo/2_day_two/4_scanner/responses/trials/'
dir_responses_probes = '/data/pt_02747_hippo/2_day_two/4_scanner/responses/probe_trials/'


subs = [os.path.basename(f) for f in glob.glob(dir_data + 'sub*')]
# only folders
subs = [sub for sub in subs if '.' not in sub]
print(subs)


print(subs)

def get_sub_number(sub_):
    return sub_.split('-')[1]


def get_response(sub_number, path = dir_responses):
    responses = [os.path.basename(f) for f in glob.glob(path + 'sub*')]
    if sub_number[0] == '0':
        return [response for response in responses if 'sub-'+sub_number[-1]+'_'  in response][0]
    else:
        return [response for response in responses if 'sub-'+sub_number+'_'  in response][0]



for sub in subs:

    print(f'Creating events file for {sub}')

    sub_i = get_sub_number(sub)
    response_file = get_response(sub_i)
    df_responses = pd.read_csv(dir_responses + response_file, sep = ',')

    response_file_probe = get_response(sub_i, path = dir_responses_probes)
    df_responses_probes = pd.read_csv(dir_responses_probes + response_file_probe, sep = ',')


    runs = df_responses['run'].unique()

    for run in runs:

        print(f'Creating events file for run {run}')


        # =============================================================================

        # get the df_responses for this run
        df_responses_run = df_responses[df_responses['run'] == run]

        # get the probe trial responses for this run
        df_responses_probes_run = df_responses_probes[df_responses_probes['run'] == run]

        # extract start time of run
        start_time_ = df_responses_run['unseen'].iloc[0] # because this is where I saved it 

        # remove first row from df_responses_run
        df_responses_run = df_responses_run.iloc[1:]


        # =============================================================================
        # add number trials to events file
        # =============================================================================

        # create a new column called 'onset' which is the time of the trial minus the start time of the run
        df_responses_run['onset'] = df_responses_run['start_time'] - start_time_

        # create a new column called 'condition' which is 'number' is len(position) == 1, and 'probe' if len(position) == 3
        df_responses_run['trial_type'] = np.where(df_responses_run['position'].str.len() == 1, 'number', 'probe')

        # create a new column called 'duration' which is 3 seconds for all 'number' condition trials
        df_responses_run['duration'] = np.where(df_responses_run['trial_type'] == 'number', 2, 0)


        # =============================================================================
        # add position to events file
        # =============================================================================


        # create a copy of df_responses_run to add in position
        df_responses_run_position = deepcopy(df_responses_run)

        # select only the 'number' condition trials
        df_responses_run_position = df_responses_run_position[df_responses_run_position['trial_type'] == 'number']

        # replace the 'trial_type' column with 'position'
        df_responses_run_position['trial_type'] = df_responses_run_position['position']

        # replace the 'duration' column with '3'
        df_responses_run_position['duration'] = 2

        
        # =============================================================================
        # add number value to events file
        # =============================================================================

        # create a copy of df_responses_run to add in number value
        df_responses_run_number_value = deepcopy(df_responses_run)

        # select only the 'number' condition trials
        df_responses_run_number_value = df_responses_run_number_value[df_responses_run_number_value['trial_type'] == 'number']

        # replace the 'trial_type' column with 'stimulus'
        df_responses_run_number_value['trial_type'] = 'stimulus'

        # replace the 'duration' column with '2'
        df_responses_run_number_value['duration'] = 2

        # =============================================================================
        # add gen condition to events file
        # =============================================================================

        # create a copy of df_responses_run to add in gen condition
        df_responses_run_gen = deepcopy(df_responses_run)

        # select only the 'number' condition trials
        df_responses_run_gen = df_responses_run_gen[df_responses_run_gen['trial_type'] == 'number']

        # convert gen column to str
        df_responses_run_gen['gen'] = df_responses_run_gen['unseen'].astype(str)

        # replace the 'trial_type' column with 'gen' if gen is 1, and 'nogen' if gen is 0
        df_responses_run_gen['trial_type'] = np.where(df_responses_run_gen['gen'] == '1', 'gen', 'nogen')

        # replace the 'duration' column with '2'
        df_responses_run_gen['duration'] = 2

        

        # =============================================================================
        # add button press info
        # =============================================================================

        # add column called 'duration' to df_responses_probes_run which is 0
        df_responses_probes_run['duration'] = 0

        # for every trial, button press onset is the time of the trial plus the rt column
        working_df = deepcopy(df_responses_probes_run)

        # change the response column to string
        working_df['response'] = working_df['response'].astype(str)

        # filter for trials where response is 1 or 2
        working_df = working_df[working_df['response'].isin(['1', '2', 1, 2, 'missed'])]

        # for these trials, button press onset is the onset time of the trial minus start_time plus the rt column
        working_df['onset'] = working_df['start_time'] - start_time_ + working_df['rt']

        # button press duration is 0 seconds
        working_df['duration'] = 0

        # rename the response column to 'trial_type'
        working_df = working_df.rename(columns = {'response': 'trial_type'})

        # change responses to 'left' and 'right'
        working_df['trial_type'] = np.where(working_df['trial_type'] == '1', 'left', working_df['trial_type'])
        working_df['trial_type'] = np.where(working_df['trial_type'] == '2', 'right', working_df['trial_type'])

        # add position column as 'NA'
        working_df['position'] = 'NA'



        # =============================================================================
        # add screen change to events file
        # =============================================================================
        
        # for every button press, screen change onset is the time of button press
        # screen change duration is the total potential time of the trial minus the rt column
        # total potential time is 3 - rt
        # trial_type is 'new_screen_left' where response is '1' and 'new_screen_right' where response is '2'
        # trial_type is 'new_screen_missed' where response is 'missed'
        # position is 'NA'

        # copy working_df to get a new df which will be for screen change
        working_df_screen_change = deepcopy(working_df)

        # change trial_type left to new_screen_left
        working_df_screen_change['trial_type'] = np.where(working_df_screen_change['trial_type'] == 'left', 'new_screen_left', working_df_screen_change['trial_type'])
        # change trial_type right to new_screen_right
        working_df_screen_change['trial_type'] = np.where(working_df_screen_change['trial_type'] == 'right', 'new_screen_right', working_df_screen_change['trial_type'])
        # change trial_type missed to new_screen_missed
        working_df_screen_change['trial_type'] = np.where(working_df_screen_change['trial_type'] == 'missed', 'new_screen_missed', working_df_screen_change['trial_type'])

        # change duration to 3 - rt
        working_df_screen_change['duration'] = 4.4 - working_df_screen_change['rt'] #total wait uncludes 1.4s max for probe!

        # add position column as 'NA'
        working_df_screen_change['position'] = 'NA'



        # =============================================================================
        # filter working_df to only include button presses where trial_type is 'left' or 'right'
        working_df = working_df[working_df['trial_type'].isin(['left', 'right'])]

        # concatenate working_df and working_df_screen_change
        working_df = pd.concat([working_df, working_df_screen_change], ignore_index = True)

        # select subset of columns
        working_df = working_df[['onset', 'duration', 'trial_type', 'position']]

        # =============================================================================



        # change position to NA for all 'probe' condition trials
        df_responses_run['position'] = np.where(df_responses_run['trial_type'] == 'probe', 'NA', df_responses_run['position'])


        # =============================================================================
        # add fixations to events file
        # =============================================================================

        # for every trial, fixation onset is the time of the trial minus the start time of the iti_before column
        # fixation duration is the iti_before column
        # trial_type is 'fixation'
        # position is 'NA'
        # response is 'NA'

        # copy and filter df_responses_run to get only number trials
        df_fixations = df_responses_run[df_responses_run['trial_type'] == 'number']
        onset_times_number = df_fixations['onset'].tolist()
        # get the iti_before column
        fixation_duration = df_fixations['iti_before'].tolist()
        # fixation onset is the time of the trial minus the iti_before column
        onset_times_fixation = [float(onset_times_number[i]) - float(fixation_duration[i]) for i in range(len(onset_times_number))]

        # create a dataframe with fixation info
        onset_times_fixation_array = [[onset_times_fixation[i], fixation_duration[i], 'fixation', 'NA'] for i in range(len(onset_times_fixation))]
        
        # add in baseline fixation
        onset_times_fixation_array.append([0, 9, 'baseline', 'NA'])
        
        df_fixations_new = pd.DataFrame(onset_times_fixation_array, columns = ['onset', 'duration', 'trial_type', 'position'])



        # take subset of df_responses_run for columns 'onset', 'duration', 'trial_type', 'position', 'response'
        df_responses_run = df_responses_run[['onset', 'duration', 'trial_type', 'position']]

        df_fixations_before_probe = df_responses_run[df_responses_run['trial_type'] == 'probe']
        onset_times_probe = [float(i)-1 for i in df_fixations_before_probe['onset'].tolist()]

        # create a dataframe with fixation info
        onset_times_probe_fixation_array = [[onset_times_probe[i], 1, 'fixation', 'NA'] for i in range(len(onset_times_probe))]
        df_fixations_probe = pd.DataFrame(onset_times_probe_fixation_array, columns = ['onset', 'duration', 'trial_type', 'position'])


        # =============================================================================
        # from df_responses_probes_run, get the onset times and durations of the probe trials
        # =============================================================================

        # column onset is the start times of the probe trials minus the start time of the run
        df_responses_probes_run['onset'] = df_responses_probes_run['start_time'] - start_time_

        # column duration is the rt column
        df_responses_probes_run['duration'] = df_responses_probes_run['rt']

        # column trial_type is 'probe'
        df_responses_probes_run['trial_type'] = 'probe'

        # column position is 'NA'
        df_responses_probes_run['position'] = 'NA'


        # filter to only include 'number' condition trials
        df_responses_run = df_responses_run[df_responses_run['trial_type'] == 'number']

        # concatenate dataframes
        df_responses_run = pd.concat([df_responses_run_position, # position code
                                      df_responses_run_number_value, # number value
                                      df_responses_run_gen, # gen condition
                                      df_responses_probes_run, # probe trials
                                      df_responses_run, # number trials
                                      df_fixations_new, # fixations before number trials
                                      df_fixations_probe, # fixations before probe trials
                                      working_df], # button presses
                                      ignore_index = True)
        print(df_responses_run)
        df_responses_run = df_responses_run[['onset', 'duration', 'trial_type']]



        # sort df_responses_run by onset
        df_responses_run = df_responses_run.sort_values(by = ['onset'])



        # select subset of rows based on value in 'trial_type' column

        # select only rows where 'trial_type' is '1' or '2'
        df_responses_run = df_responses_run[df_responses_run['trial_type'].isin(['1', '2', '3', '4', 'probe', 'left', 'right', 'new_screen_left', 'new_screen_right', 'new_screen_missed'])]

        # for stick function, change duration of 1,2,3,4 to 0
        df_responses_run['duration'] = np.where(df_responses_run['trial_type'].isin(['1', '2', '3', '4']), 0, df_responses_run['duration'])

        # to include condition-general regressor, duplicate rows where 'trial_type' is ['1', '2', '3', '4'] and change 'trial_type' to 'stimulus' with duration 2
        df_responses_run_all = df_responses_run[df_responses_run['trial_type'].isin(['1', '2', '3', '4'])].copy()
        df_responses_run_all['trial_type'] = 'stimulus'
        df_responses_run_all['duration'] = 2

        # concatenate df_responses_run and df_responses_run_all
        df_responses_run = pd.concat([df_responses_run, df_responses_run_all], ignore_index = True)

        # sort df_responses_run by onset
        df_responses_run = df_responses_run.sort_values(by = ['onset'])


        # save the df_responses_run as a tsv file in the BIDS directory

        df_responses_run.to_csv(dir_data + sub + '/func/' + sub + '_task-probe_run-0' + str(run) + '_events.tsv', sep = '\t', index = False)
        df_responses_run.to_csv(dir_deriv + sub + '/func/' + sub + '_task-probe_run-0' + str(run) + '_events.tsv', sep = '\t', index = False)







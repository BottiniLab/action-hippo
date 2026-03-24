smoothing_ = 5


import sys
sub = sys.argv[1] # e.g. '01'
contrast_ = sys.argv[2] # e.g. '1_2_3_4'

append_str = '_stickfunction0vis'

# #############################################################################


# =============================================================================
# 1. First level analysis
# =============================================================================


import os
import json
from glob import glob
import pandas as pd
import numpy as np

from nilearn.glm.first_level import first_level_from_bids
from nilearn.interfaces.bids import save_glm_to_bids
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import norm
from nilearn import plotting, image
from nilearn.plotting import plot_design_matrix


dataset_dir = '/data/pt_02747/action_hippo/data/'
data_dir = '/data/pt_02747/action_hippo/data/derivatives/'
output_dir_first = '/data/pt_02747/action_hippo/data/derivatives/first_level/'
output_dir_second = '/data/pt_02747/action_hippo/data/derivatives/second_level/'
working_dir = '/data/pt_02747/action_hippo/data/derivatives/working_dir/'
mask_dir = f'/data/pt_02747/action_hippo/data/derivatives/sub-{sub}/anat/'


# split contrast_ and convert to list of int
contrasts_ = contrast_.split('_')

# =============================================================================
# make confounds match image filter, because nilearn sucks
# for anyone in the future: you might not have to do this if nilearn recognises your confounds files from fmriprep. 
# =============================================================================

# get all relevant confoudns and save in a new file

confounds_files = [os.path.abspath(f) for f in glob(data_dir + f'sub-{sub}/func/*_task-probe_*_desc-confounds_timeseries.tsv')]
if len(confounds_files) > 0:
    for fil in confounds_files:

        # select subset of cofounds to use 
        confounds_df = pd.read_csv(fil, sep='\t')

        # select subset of columns from confounds_df: trans_x, trans_y, trans_z, rot_x, rot_y, rot_z, csf, white_matter, global_signal
        confounds_reduced_df = confounds_df[['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf', 'white_matter']] #not use global signal

        # save reduced confounds to file
        # add "_reduced" to filename before extension
        # add image filter to filename
        confounds_reduced_filename_divided = fil.split('.') # take out other stuff so descriptor is just 'confounds'
        confounds_reduced_filename = confounds_reduced_filename_divided[0] + '_reduced.' + confounds_reduced_filename_divided[1]

        # save file if it doesn't already exist
        if not os.path.exists(confounds_reduced_filename):
            confounds_reduced_df.to_csv(confounds_reduced_filename, sep='\t')



def get_confounds_file(run_file):
    # get confounds file
    confounds_file = run_file.replace('_space-T1w_desc-preproc_bold.nii.gz', '_desc-confounds_timeseries_reduced.tsv')
    return confounds_file



# =============================================================================
# run first level model for the subject specified using the bash file
# this allows us to do magic slurmy parallelisation
# =============================================================================


# =============================================================================
# get subject specific brain mask
# =============================================================================

# get brain mask from fmriprep in anat folder
orig_mask_nii = nib.load(f'{mask_dir}sub-{sub}_desc-brain_mask.nii.gz')

# resample mask to match functional data

# take a sample functional image
image_file = f'/data/pt_02747/action_hippo/data/derivatives/sub-{sub}/func/sub-{sub}_task-probe_run-08_space-T1w_desc-preproc_bold.nii.gz'

image_nii = nib.load(image_file)

# resample mask to match functional data
mask_nii = image.resample_to_img(orig_mask_nii, image_nii, interpolation='nearest')


# check if there's a subject-specific drive and if not, make one
if not os.path.exists(f'{output_dir_first}nilearn_glm_runwise/sub-{sub}'):
    os.makedirs(f'{output_dir_first}nilearn_glm_runwise/sub-{sub}')

sub_dir = f'{output_dir_first}nilearn_glm_runwise/sub-{sub}/'

for run in ['01', '02', '03', '04', '05', '06', '07', '08']:

    # create run folder; not strictly necessary but makes things neater
    if not os.path.exists(f'{sub_dir}run-{run}'):
        os.makedirs(f'{sub_dir}run-{run}')

    run_dir = f'{sub_dir}run-{run}/'

    (
    models,
    models_run_imgs,
    models_events,
    models_confounds,
    ) = first_level_from_bids(dataset_path = dataset_dir,
                                        img_filters = [('space', 'T1w'), ('desc', 'preproc'), ('run', run)],
                                    task_label = 'probe',
                                    sub_labels=[sub],
                                    t_r = 1,
                                    slice_time_ref = None,
                                    hrf_model = 'spm',
                                    high_pass = .01,
                                    smoothing_fwhm = smoothing_,
                                    signal_scaling = 0,
                                    minimize_memory=False,
                                    mask_img=mask_nii) # ('run', run[-2:]), ('desc', 'preproc'), 
    

    # Create decoding output directory
    output_dir_decoding = f'/data/pt_02747/action_hippo/data/derivatives/first_level/glm_decoding/sub-{sub}/'
    os.makedirs(output_dir_decoding, exist_ok=True)

    # Modify events: assign a unique regressor per trial for cond1–4
    for i_run, events_run in enumerate(models_events[0]):
        updated_events = []
        for idx, row in events_run.iterrows():
            row = row.copy()
            if str(row["trial_type"]) in ["1", "2", "3", "4"]:
                row["trial_type"] = f"cond{row['trial_type']}_trial_{idx:03d}"
                row["duration"] = 1
                orig_onset = row["onset"]
                row["onset"] = orig_onset + 0.899-0.07145
            updated_events.append(row)
        models_events[0][i_run] = pd.DataFrame(updated_events)



    print(models_confounds)
    print('models are ', models)

    # special code to remove run 7 from sub-10 if we ask for left or right button press

    print('this many models, ', len(models_run_imgs))

    if sub == '10' and ('left' in contrast_ or 'right' in contrast_):
        models_run_imgs = [models_run_imgs[0][:6] + models_run_imgs[0][7:]]
        models_events = [models_events[0][:6] + models_events[0][7:]]

        try:
            models_confounds = [models_confounds[0][:6] + models_confounds[0][7:]]
        except:
            print('no confounds to remove')
            pass



    models_confounds = []

    for sub_ in models_run_imgs:
        sub_confounds = []
        for run_ in sub_:
            confounds_file = get_confounds_file(run_)
            sub_confounds.append(confounds_file)
        models_confounds.append(sub_confounds)

    print(models_confounds)


    #############################################################################
    # Quick sanity check on fit arguments
    # -----------------------------------
    # Additional checks or information extraction from pre-processed data can
    # be made here.

    ############################################################################
    # We just expect 8 run_imgs per subject.

    print([os.path.basename(run) for run in models_run_imgs[0]])


    ############################################################################
    print(models_events[0][0]["trial_type"].value_counts())
    ############################################################################
    # First level model estimation
    # ----------------------------
    # Now we simply fit each first level model and plot for each subject the contrast

    model_and_args = zip(models, models_run_imgs, models_events, models_confounds)

    print(model_and_args)

    for midx, (model, imgs, events, confounds) in enumerate(model_and_args):
        # fit the GLM
        model.fit(imgs, events, confounds)
        print(f'fitting model {midx}')

        # if I wanted to, I could apply a run-wise brain mask here

        print('print first design matrix')
        design_matrix = model.design_matrices_[0]
        
        plot_design_matrix(design_matrix)
        
        if len(model.design_matrices_) >1:
            print('print second design matrix')
            design_matrix = model.design_matrices_[1]
            plot_design_matrix(design_matrix)

    # save model

    for model in models:
        for reg in model.design_matrices_[0].columns:
            if reg.startswith("cond"):
                beta = model.compute_contrast(reg, output_type="effect_size")
                fname = f"sub-{sub}_run-{run}_{reg}_beta.nii.gz"
                nib.save(beta, os.path.join(output_dir_decoding, fname))




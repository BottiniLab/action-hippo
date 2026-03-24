smoothing_ = 5


import sys
sub = sys.argv[1] # e.g. '01'
contrast_ = sys.argv[2] # e.g. 'left+right-baseline'


#sub = '10'
#contrast_ = 'modulation_affordance-modulation_behav_affordance'

contrasts_ = contrast_.split('-')

append_str = ''

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
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import norm
from nilearn import plotting, image


dataset_dir = '/data/pt_02747/action_hippo/data/'
data_dir = '/data/pt_02747/action_hippo/data/derivatives/'
output_dir_first = '/data/pt_02747/action_hippo/data/derivatives/first_level/'
output_dir_second = '/data/pt_02747/action_hippo/data/derivatives/second_level/'
working_dir = '/data/pt_02747/action_hippo/data/derivatives/working_dir/'
mask_dir = f'/data/pt_02747/action_hippo/data/derivatives/sub-{sub}/anat/'


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
# get subject specific GM mask
# =============================================================================

# get mask from fmriprep in anat folder
prob_mask_nii = nib.load(f'{mask_dir}sub-{sub}_label-GM_probseg.nii.gz')

# threshold prob mask at a certain value
threshold_ = 0.3
str_threshold = str(threshold_).replace('.', '')

# resample mask to match functional data

# take a sample functional image
image_file = f'/data/pt_02747/action_hippo/data/derivatives/sub-{sub}/func/sub-{sub}_task-probe_run-08_space-T1w_desc-preproc_bold.nii.gz'
image_nii = nib.load(image_file)
shape_func = image_nii.shape

# resample mask to match functional data
mask_nii = image.resample_to_img(prob_mask_nii, image_nii, interpolation='continuous')
bin_mask_nii = image.get_data(mask_nii)


bin_mask_nii[bin_mask_nii < threshold_] = 0
bin_mask_nii[bin_mask_nii >= threshold_] = 1

# save binary mask
prob_mask_binary_nii = nib.Nifti1Image(bin_mask_nii, mask_nii.affine, mask_nii.header)
nib.save(prob_mask_binary_nii, f'{mask_dir}sub-{sub}_space-T1w_label-GM_probseg_binary_threshold-{str_threshold}.nii.gz')


(
models,
models_run_imgs,
models_events,
models_confounds,
) = first_level_from_bids(dataset_path = dataset_dir,
                                    img_filters = [('space', 'T1w'), ('desc', 'preproc')],
                                task_label = 'probe',
                                sub_labels=[sub],
                                t_r = 1,
                                slice_time_ref = None,
                                hrf_model = 'spm',
                                high_pass = .01,
                                smoothing_fwhm = smoothing_,
                                signal_scaling = 0,
                                minimize_memory=False,
                                mask_img=prob_mask_binary_nii) # ('run', run[-2:]), ('desc', 'preproc'), 

print(models_confounds)
print('models are ', models)

# special code to remove run 7 from sub-10 if we ask for left or right button press

print('this many models, ', len(models_run_imgs))

if sub == '10' and ('left' in contrasts_ or 'right' in contrasts_):
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
    for run in sub_:
        confounds_file = get_confounds_file(run)
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
import os

print([os.path.basename(run) for run in models_run_imgs[0]])


############################################################################
print(models_events[0][0]["trial_type"].value_counts())
############################################################################
# First level model estimation
# ----------------------------
# Now we simply fit each first level model and plot for each subject the contrast

############################################################################
# Set the threshold as the z-variate with an uncorrected p-value of 0.001.
from scipy.stats import norm

p001_unc = norm.isf(0.001)

############################################################################
# Prepare figure for concurrent plot of individual maps.
import matplotlib.pyplot as plt

from nilearn import plotting

model_and_args = zip(models, models_run_imgs, models_events, models_confounds)

print(model_and_args)

for midx, (model, imgs, events, confounds) in enumerate(model_and_args):
    # fit the GLM
    model.fit(imgs, events, confounds)
    print(f'fitting model {midx}')

    # NOTE: if I wanted to, I could apply a run-wise brain mask here

    print('print first design matrix')
    design_matrix = model.design_matrices_[0]
    from nilearn.plotting import plot_design_matrix
    plot_design_matrix(design_matrix)
    #plt.show()
    
    if len(model.design_matrices_) >1:
        print('print second design matrix')
        design_matrix = model.design_matrices_[1]
        from nilearn.plotting import plot_design_matrix
        plot_design_matrix(design_matrix)
        #plt.show()



from nilearn.interfaces.bids import save_glm_to_bids
# save model

# check if there's a subject-specific drive and if not, make one
if not os.path.exists(f'{output_dir_first}nilearn_glm/sub-{sub}'):
    os.makedirs(f'{output_dir_first}nilearn_glm/sub-{sub}')

sub_dir = f'{output_dir_first}nilearn_glm/sub-{sub}/'


for model in models:
    save_glm_to_bids(
        model,
        contrasts=contrasts_,
        out_dir=sub_dir,
        prefix=f"sub-{sub}_smoothing-{smoothing_}_space-T1w{append_str}",
    )




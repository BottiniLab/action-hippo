
import sys
sub_i = sys.argv[1] # e.g. '01'
append_str = sys.argv[2] # '_stickfunction5vis'
thr_str = '_thr3'
wholebrain=True


from datetime import datetime
startTime = datetime.now()

# =============================================================================
# 1. import packages
# =============================================================================


sys.path.append('/data/u_eperon_software/python/debian-bullseye-amd64/lib/python3.9/site-packages') # in case the environment did not load properly

import os
import rsatoolbox
import rsatoolbox.data as rsd # abbreviation to deal with dataset
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import nilearn
import nibabel as nib
from nilearn.masking import apply_mask
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs_crossvalidated, evaluate_models_searchlight
from nilearn import plotting
from rsatoolbox.inference import eval_fixed
from rsatoolbox.model import ModelFixed
from rsatoolbox.rdm import RDMs
from nilearn.image import resample_to_img

def upper_tri(RDM):
    """upper_tri returns the upper triangular index of an RDM

    Args:
        RDM 2Darray: squareform RDM

    Returns:
        1D array: upper triangular vector of the RDM
    """
    # returns the upper triangle
    m = RDM.shape[0]
    r, c = np.triu_indices(m, 1)
    return RDM[r, c]


# =============================================================================
# 2. Set up parameters and paths
# =============================================================================
sub = 'sub-' + sub_i
runs = ['01', '02', '03', '04', '05', '06', '07', '08']
conditions = ['1', '2', '3', '4']
#drives
glm_results_dir = '/data/pt_02747/action_hippo/data/derivatives/first_level/nilearn_glm_runwise/'
mask_dir = f'/data/pt_02747/action_hippo/data/derivatives/{sub}/anat/'
# files
GM_mask_1mm = f'{sub}_space-T1w_label-GM_probseg_binary_threshold-03.nii.gz'

if wholebrain:
    thr_str = thr_str + 'wholebrain' 
    GM_mask_1mm = f'{sub}_desc-brain_mask.nii.gz'

func_file = f'/data/pt_02747/action_hippo/data/derivatives/{sub}/func/{sub}_task-probe_run-02_space-T1w_desc-preproc_bold.nii.gz'

# =============================================================================
# 3. Get searchlight centres and neighbours
# =============================================================================

# resample GM mask to the dimensions of the func file, please
mask_nii = resample_to_img(mask_dir + GM_mask_1mm, func_file, interpolation='nearest')
# get mask data from mask
mask = mask_nii.get_fdata()
# get searchlight centres and neighbours
centers, neighbors = get_volume_searchlight(mask, radius=3, threshold=0.3) # low threshold as we use a gm mask: means more corrections, but we reduce the number already with a gm mask
# previous results done with threshold of 0.3, trying 0.1 in this script to see if we get more voxels to work with

# =============================================================================
# 4. Get data
# =============================================================================

n_rep = 8 # number of repetitions of each condition, i.e. runs
conds = np.array(['cond_' + x for x in conditions]) # conditions
conds = np.repeat(conds, n_rep) # repeat each condition n_rep times
print('conditions: ', conds)
sessions = np.tile(np.arange(n_rep), len(conditions)) # sessions
sessions = np.array(['session_' + str(x+1) for x in sessions])
print('sessions: ', sessions)

des = {'subj': sub} # design matrix
obs_des = {'conds': conds, 'sessions': sessions} # observations: i.e. conditions and sessions

# load the glm results for each condition
# condition_data_dict = {}
measurements = []
for condition in conditions:
    for run in runs:
        run_drive = glm_results_dir + sub + '/run-' + run + '/'
        beta_file = run_drive + f'{sub}_run-{run}_space-T1w{append_str}_contrast-{condition}_stat-effect_statmap.nii.gz'
        beta = nib.load(beta_file)
        beta = beta.get_fdata()
        # append the masked beta map to the measurements array, giving us a 2D array of size n_observations x n_voxels (n_channels)
        measurements.append(beta)

measurements = np.array(measurements)
data_2d = measurements.reshape([measurements.shape[0], -1])
data_2d = np.nan_to_num(data_2d)

# get residuals too!
residuals = []
for run in runs:
    print('getting residuals for run ', run)
    run_drive = glm_results_dir + sub + '/run-' + run + '/'
    res_file = f'{sub}_run-{run}_space-T1w{append_str}_residuals.nii.gz'
    res = nib.load(run_drive+res_file)
    res = res.get_fdata()
    # convert into array
    res = np.array(res)

    # make 2D array of voxels*measurements, so we can slice it up all neat like 
    residuals_2d = res.reshape([res.shape[-1], -1])
    #residuals_2d = measurements.reshape([measurements.shape[0], -1])
    residuals.append(residuals_2d)

# create big residuals array of size (8, n_voxels, 415) (runs, voxels, measurements)
# residuals should be size (n_residuals, n_channels) - i.e. (415, n_voxels)
residuals = np.array(residuals)
print('created residuals array of size ', residuals.shape)

# =============================================================================
# 5. Get RDMs
# =============================================================================


output_path = '/data/pt_02747/action_hippo/data/derivatives/first_level/searchlight/'
# check if we've already done this
if not os.path.exists(output_path + sub + f'/{sub}_space-T1w{append_str}_searchlight_RDMs_crossnobis{thr_str}.pkl'):
    SL_RDMs = get_searchlight_RDMs_crossvalidated(data_2d=data_2d, 
                                                centers=centers, 
                                                neighbors=neighbors, 
                                                obs_descriptor=obs_des,
                                            method='crossnobis', 
                                            n_conds = 4,
                                            n_sessions = 8,
                                            verbose=True,
                                            method_cov='shrinkage_eye',
                                            residuals=residuals)
    # save as a pickle :)

    # make sub drive if not there
    if not os.path.exists(output_path + sub):
        os.makedirs(output_path + sub)

    SL_RDMs.save(output_path + sub + f'/{sub}_space-T1w{append_str}_searchlight_RDMs_crossnobis{thr_str}.pkl')


# otherwise just load in the file, which isn't actually a pickle embarrassingly - future people running this script should correct this in the filename, probably :)
else:
    SL_RDMs = rsatoolbox.rdm.rdms.load_rdm(output_path + sub + f'/{sub}_space-T1w{append_str}_searchlight_RDMs_crossnobis{thr_str}.pkl', file_type='hdf5')

# =============================================================================

print('time elapsed: ', datetime.now() - startTime)
time_elapsed = datetime.now() - startTime

# =============================================================================

# compare to rdms - for time, let's just write it in here
# note that this is a sub-optimal method: in future we need to edit the compare_rdms function to take in a list of models and do partial correlation


import json
models = ['affordance']
# model directory
model_dir = '/data/pt_02747/action_hippo/data/derivatives/first_level/models/'
# model file
model_file = f"{sub}_model_dictionaries.json"
# load model dictionary
with open(model_dir + model_file) as f:
    model_dict = json.load(f)
        
# iterate through models to compare
for model_name in models:
    # load model
    model = model_dict[model_name]
    # get model RDM
    model_diss = rsatoolbox.rdm.RDMs(np.array([[[model['11'], model['12'], model['13'], model['14']],
                                                    [model['12'], model['22'], model['23'], model['24']],
                                                    [model['13'], model['23'], model['33'], model['34']],
                                                    [model['14'], model['24'], model['34'], model['44']]]]),
                                dissimilarity_measure='count'
                            )
    # create model as fixed model for evaluation
    rdm_model = rsatoolbox.model.ModelFixed('affordance', model_diss)
    rdm_model.n_rdm = 1

    # evaluate model
    if model_name == 'behav_affordance':
        eval_results = evaluate_models_searchlight(SL_RDMs, rdm_model, eval_fixed, method='corr', n_jobs=3)
        comp_str = 'corr'
    else:
        eval_results = evaluate_models_searchlight(SL_RDMs, rdm_model, eval_fixed, method='rho-a', n_jobs=3)
        comp_str = 'rho-a'

    # map to brain
    # method from rsatoolbox demo - a bit hacky, but it works. At some point I will go back and make it more pythonic

    # get the evaulation score for each voxel
    # We only have one model, but evaluations returns a list. By using float we just grab the value within that list
    eval_score = [float(e.evaluations) for e in eval_results]

    # Create an 3D array, with the size of mask, and use this to plot our searchlight scores
    x, y, z = mask.shape
    RDM_brain = np.zeros([x*y*z])
    RDM_brain[list(SL_RDMs.rdm_descriptors['voxel_index'])] = eval_score
    RDM_brain = RDM_brain.reshape([x, y, z])

    # turn into nifti
    RDM_brain_nii = nib.Nifti1Image(RDM_brain, affine=mask_nii.affine)


    # save nifti
    RDM_brain_nii.to_filename(output_path + sub + f'/{sub}_space-T1w{append_str}_searchlight_model-{model_name}_eval-{comp_str}{thr_str}.nii.gz')

    # plot and save distribution of evail scores
    sns.distplot(eval_score)
    plt.title(f'Distributions of correlations for {model_name}', size=18)
    plt.ylabel('Occurance', size=18)
    plt.xlabel('Spearman rho correlation', size=18)
    # save plot
    plt.savefig(output_path + sub + f'/{sub}_space-T1w{append_str}_searchlight_model-{model_name}_eval-{comp_str}{thr_str}.png')


'''
# =============================================================================

model_rdms = rsatoolbox.rdm.RDMs(np.array([[[0,1,1,0],
                                    [1,0,2,1],
                                    [1,2,0,1],
                                    [0,1,1,0]]]),
                            dissimilarity_measure='Count'
                           )

model_aff = rsatoolbox.model.ModelFixed('affordance', model_rdms)

eval_results = evaluate_models_searchlight(SL_RDMs, model_aff, eval_fixed, method='rho-a', n_jobs=3)

# get the evaulation score for each voxel
# We only have one model, but evaluations returns a list. By using float we just grab the value within that list
eval_score = [float(e.evaluations) for e in eval_results]

sns.distplot(eval_score)
plt.title('Distributions of correlations', size=18)
plt.ylabel('Occurance', size=18)
plt.xlabel('Spearman rho correlation', size=18)
sns.despine()
plt.show()

# =============================================================================
# Create an 3D array, with the size of mask, and
x, y, z = mask.shape
RDM_brain = np.zeros([x*y*z])
RDM_brain[list(SL_RDMs.rdm_descriptors['voxel_index'])] = eval_score
RDM_brain = RDM_brain.reshape([x, y, z])

# turn into nifti
RDM_brain_nii = nib.Nifti1Image(RDM_brain, affine=mask_nii.affine)

# save nifti
RDM_brain_nii.to_filename(output_path + sub + f'/{sub}_space-T1w{append_str}_searchlight_model-affordance_eval-rho-a.nii.gz')


# =============================================================================

# lets plot the voxels above the 0th percentile
threshold = np.percentile(eval_score, 0)
from nilearn.image import new_img_like
plot_img = new_img_like(func_file, RDM_brain)


coords = range(-20, 40, 5)
fig = plt.figure(figsize=(12, 3))

display = plotting.plot_stat_map(
        plot_img, colorbar=True, cut_coords=coords,threshold=threshold,
        display_mode='z', draw_cross=False, figure=fig,
        title=f'Affordance model evaluation',
        black_bg=False, annotate=False)
plt.show()

# =============================================================================
'''
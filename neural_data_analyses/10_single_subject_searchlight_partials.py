
import sys
sub_i = sys.argv[1] # e.g. '01'
append_str = sys.argv[2] # '_stickfunction5vis'


#sub_i = '01'
#append_str = '_stickfunction5vis'


thr_str = '_thr3wholebrain'

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

# =============================================================================
# function to run partial correlation using rho-a
# =============================================================================

def compare_rho_a(vector1, vector2):
    """calculates the spearman rank correlations between
    two RDMs objects without tie correction

    Args:
        v1, v2 (numpy.ndarray): RDMs to be compared
    Returns:
        numpy.ndarray: dist:
            rank correlations between the two RDMs

    """

    # check input is np array and if not, make it one
    if type(vector1) is not np.ndarray:
        vector1 = np.array(vector1)
    if type(vector2) is not np.ndarray:
        vector2 = np.array(vector2)

    # check input is 2D and if not, make it one
    if len(vector1.shape) != 2:
        vector1 = np.atleast_2d(vector1)
    if len(vector2.shape) != 2:
        vector2 = np.atleast_2d(vector2)


    # check input is horizontal and if not, make it one
    if vector1.shape[0] > vector1.shape[1]:
        vector1 = vector1.T
    if vector2.shape[0] > vector2.shape[1]:
        vector2 = vector2.T
        
    vector1 = np.apply_along_axis(scipy.stats.rankdata, 1, vector1)
    vector2 = np.apply_along_axis(scipy.stats.rankdata, 1, vector2)
    vector1 = vector1 - np.mean(vector1, 1, keepdims=True)
    vector2 = vector2 - np.mean(vector2, 1, keepdims=True)
    n = vector1.shape[1]
    sim = np.einsum('ij,kj->ik', vector1, vector2) / (n ** 3 - n) * 12
    return sim[0][0]

import scipy

def partial_corr_one_step(x, y, covar):
    """
    Compute the partial correlation coefficient between two variables (x and y)
    while controlling for the effects of other variables (covar) using rho-a correlation.

    Parameters:
    - x, y: variables as list
    - covar: list of covariates (list of lists).

    Returns:
    - r: Partial correlation coefficient between x and y controlling for covar.
    """

    # Compute rho-a correlation between x and y
    r_xy = compare_rho_a(x, y)

    # Compute rho-a correlations between x, y, and covariates
    r_x_covar = [compare_rho_a(x, cov) for cov in covar]
    r_y_covar = [compare_rho_a(y, cov) for cov in covar]

    # Compute the partial correlation coefficient
    numerator = r_xy - np.dot(r_x_covar, r_y_covar)
    denominator = np.sqrt((1 - np.sum(np.square(r_x_covar))) * (1 - np.sum(np.square(r_y_covar))))
    r = numerator / denominator

    return r


def partial_corr_two_step(x_, y_, covar_):

    # covar should be a list of two column names :)

    part_1 = partial_corr_one_step(x_, y_, [covar_[0]]) # 1 and 2, given 3
    #print(part_1)
    part_2 = partial_corr_one_step(x_, covar_[1], [covar_[0]]) # 1 and 4, given 3
    #print(part_2)
    part_3 = partial_corr_one_step(y_, covar_[1], [covar_[0]]) # 2 and 4, given 3
    #print(part_3)

    # see http://wzimm.weebly.com/uploads/1/3/5/2/13522665/partial_correlation_intro_1.pdf for more info here

    denominator = np.sqrt((1 - (part_2 ** 2)) * (1 - (part_3 ** 2)))

    # Avoid division by zero
    if np.isclose(denominator, 0):
        raise ValueError("Denominator is close to zero, unable to compute partial correlation coefficient.")

    r = (part_1 - (part_2 * part_3)) / denominator

    return r

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

if 'wholebrain' in thr_str:
    GM_mask_1mm = f'{sub}_desc-brain_mask.nii.gz'

func_file = f'/data/pt_02747/action_hippo/data/derivatives/{sub}/func/{sub}_task-probe_run-02_space-T1w_desc-preproc_bold.nii.gz'


# resample GM mask to the dimensions of the func file, please
mask_nii = resample_to_img(mask_dir + GM_mask_1mm, func_file, interpolation='nearest')
# get mask data from mask
mask = mask_nii.get_fdata()

# =============================================================================
# 5. Get RDMs
# =============================================================================


output_path = '/data/pt_02747/action_hippo/data/derivatives/first_level/searchlight/'

SL_RDMs = rsatoolbox.rdm.rdms.load_rdm(output_path + sub + f'/{sub}_space-T1w{append_str}_searchlight_RDMs_crossnobis{thr_str}.pkl', file_type='hdf5') 
# let' s pretend we didn't save all our data with the wrong file extension


# =============================================================================

# compare to rdms - for time, let's just write it in here
# note that this is a sub-optimal method: in future we need to edit the compare_rdms function to take in a list of models and do partial correlation


import json
# model directory
model_dir = '/data/pt_02747/action_hippo/data/derivatives/first_level/models/'
# model file
model_file = f"{sub}_model_dictionaries_rt_test.json"
# load model dictionary
with open(model_dir + model_file) as f:
    models = json.load(f)

# =============================================================================


# iterate through RDMs to compare
models_=['affordance_magnitude', 'rt', 'link_distance']
model_deets = {}
for model_ in models_:
    model_diss_tmp = rsatoolbox.rdm.RDMs(np.array([[[models[model_]['11'], models[model_]['12'], models[model_]['13'], models[model_]['14']],
                                                [models[model_]['12'], models[model_]['22'], models[model_]['23'], models[model_]['24']],
                                                [models[model_]['13'], models[model_]['23'], models[model_]['33'], models[model_]['34']],
                                                [models[model_]['14'], models[model_]['24'], models[model_]['34'], models[model_]['44']]]]),
                            dissimilarity_measure='count',
                            descriptors={'n_rdm': 1}
                        )
    model_deets['model_diss_' + model_] = model_diss_tmp.dissimilarities[0]

print(model_deets)

all_corrs = []
indie_SLMs = 0
for RDM in SL_RDMs:
    # print progress
    print(f'Processing voxel {RDM.rdm_descriptors["voxel_index"][0]}')
    # contextualise out of total number of voxels
    print(f'Voxel {indie_SLMs} out of {len(SL_RDMs.rdm_descriptors["voxel_index"])}')

    neural_diss_tmp = RDM.dissimilarities[0]

    print('dissimilarityies are: ')
    print(neural_diss_tmp)

    vox_ind = RDM.rdm_descriptors['voxel_index'][0]
    vox_corrs = []

    for model_name in models_[:2]: # SO WE ONLY GET ONE
        # get model dissimilarity

        models_to_exclude = [mod for mod in models_ if mod != model_name]
        # create list from dictionary
        model_diss_exc_tmp = [model_deets['model_diss_' + mod] for mod in models_to_exclude]

        model_diss_inc_tmp = model_deets['model_diss_' + model_name]

        if len(model_diss_exc_tmp) == 1:
            eval_corr = partial_corr_one_step(neural_diss_tmp, model_diss_inc_tmp, model_diss_exc_tmp)
        elif len(model_diss_exc_tmp) == 2:
            eval_corr = partial_corr_two_step(neural_diss_tmp, model_diss_inc_tmp, model_diss_exc_tmp)
        model_diss = model_deets['model_diss_' + model_name]
        
        # if 1 or -1, replace with 0.999 or -0.999
        if eval_corr == 1:
            eval_corr = 0.999
        elif eval_corr == -1:
            eval_corr = -0.999

        eval_score = np.arctanh(eval_corr) # z-score
        vox_corrs.append(eval_score)

    all_corrs.append(vox_corrs)
    indie_SLMs += 1


indies = 0
for model_name in models_[:2]:
    # get the corrs for that specific model, which is the index of the model in the list of models
    corr_list = [vox[indies] for vox in all_corrs]

    # Create an 3D array, with the size of mask, and use this to plot our searchlight scores
    x, y, z = mask.shape
    RDM_brain = np.zeros([x*y*z])
    RDM_brain[list(SL_RDMs.rdm_descriptors['voxel_index'])] = corr_list
    RDM_brain = RDM_brain.reshape([x, y, z])

    # turn into nifti
    RDM_brain_nii = nib.Nifti1Image(RDM_brain, affine=mask_nii.affine)

    # save nifti
    RDM_brain_nii.to_filename(output_path + sub + f'/{sub}_space-T1w{append_str}_searchlight_model-{model_name}2_eval-rho-a{thr_str}_partial.nii.gz')

    # tell us something
    print('saved searchlight model evaluation for ' + model_name + ' in ' + sub + f'/{sub}_space-T1w{append_str}_searchlight_model-{model_name}_eval-rho-a{thr_str}_partial.nii.gz')
    indies += 1

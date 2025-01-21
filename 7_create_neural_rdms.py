
# for a single subject, calculate and save a neural dissimilarity matrix for a given ROI, and save with an appropriate name 
# this will be useful for parallelisation
# run using python 3.9 and act_hippo_rsa environment using rsatoolbox 1.3.0


import sys
sub_i = sys.argv[1] # e.g. '01'
region = sys.argv[2] # e.g. 'Entorhinal'
hemisphere = sys.argv[3] # e.g. 'right'

append_str = '_stickfunction5vis'

#sub_i = '01' # for debugging
#region = 'EC_L'

# =============================================================================
# 1. import packages
# =============================================================================


sys.path.append('/data/u_eperon_software/python/debian-bullseye-amd64/lib/python3.9/site-packages') # in case the environment did not load properly

import rsatoolbox
import rsatoolbox.data as rsd # abbreviation to deal with dataset
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import nilearn
import nibabel as nib
from nilearn.masking import apply_mask


# =============================================================================
# 2. Set up parameters and paths
# =============================================================================
sub = 'sub-' + sub_i
runs = ['01', '02', '03', '04', '05', '06', '07', '08']
conditions = ['1', '2', '3', '4']

glm_results_dir = '/data/pt_02747/action_hippo/data/derivatives/first_level/nilearn_glm_runwise/'
mask_dir = f'/data/pt_02747/action_hippo/data/derivatives/{sub}/anat/roi/'


if 'rel5' in region:
    mask_file = f'{region}-{hemisphere}/{sub}_mask-{region}_{hemisphere}{append_str}.nii.gz'
elif 'rel2' in region:
    mask_file = f'{region}-{hemisphere}/{sub}_mask-{region}_{hemisphere}{append_str}.nii.gz'
else:
    mask_file = f'{region}-{hemisphere}/{sub}_mask-{region}_{hemisphere}.nii.gz'



# =============================================================================
# 3. Set up dataset object for the RDM
# =============================================================================

# load the mask 
mask = nib.load(mask_dir + mask_file)

n_rep = 8 # number of repetitions of each condition, i.e. runs
conds = np.array(['cond_' + x for x in conditions]) # conditions
conds = np.repeat(conds, n_rep) # repeat each condition n_rep times

print('conditions: ', conds)

sessions = np.tile(np.arange(n_rep), len(conditions)) # sessions

sessions = np.array(['session_' + str(x+1) for x in sessions])

print('sessions: ', sessions)


des = {'subj': sub} # design matrix

# load the glm results for each condition
# condition_data_dict = {}
measurements = []
for condition in conditions:
    for run in runs:
        run_drive = glm_results_dir + sub + '/run-' + run + '/'
        beta_file = run_drive + f'{sub}_run-{run}_space-T1w{append_str}_contrast-{condition}_stat-effect_statmap.nii.gz'
        beta = nib.load(beta_file)

        # apply the mask to the beta map, giving us a 1D array of size n_voxels (n_channels)
        masked_beta = apply_mask(beta, mask)

        # save the masked beta map to a dictionary
        # condition_data_dict[condition] = masked_beta

        # append the masked beta map to the measurements array, giving us a 2D array of size n_observations (runs) x n_voxels (n_channels)
        measurements.append(masked_beta)

# use size of mask as number of voxels
n_voxels = mask.get_fdata().sum()
print('number of voxels: ', n_voxels)

measurements = np.array(measurements)

# design matrix
chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(n_voxels)])} # channels: i.e. voxels in ROI

obs_des = {'conds': conds, 'sessions': sessions} # observations: i.e. conditions and sessions

#measurements is a 2D array of size n_observations x n_channels, e.g. [[voxel, voxel],[voxel, voxel]]
print('dimensions of measurements should be n_observations (sessions*conditions) x n_channels (voxels): ')
print(np.shape(measurements))

# create a data matrix
data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des)
print(data)


# =============================================================================
# 4. Calculate and add residuals
# =============================================================================

residuals = {} # create dict of 8 residuals, one for each run

ind_noise = 0
# let's see if the residuals are not singular, so we can actually use them!
noise_singular = False
for run in runs:
    run_drive = glm_results_dir + sub + '/run-' + run + '/'
    res_file = f'{sub}_run-{run}_space-T1w{append_str}_residuals.nii.gz'
    res = nib.load(run_drive+res_file)
    # apply the mask to the residuals
    masked_res = apply_mask(res, mask)
    print('residuals for run ', run, ' are: ', masked_res)
    # make into np matrix
    masked_res = np.array(masked_res)
    # save as csv in /data/pt_02747/action_hippo/
    #np.savetxt(f'/data/pt_02747/action_hippo/data/derivatives/first_level/nilearn_glm_runwise/{sub}/run-{run}/{sub}_run-{run}_space-T1w_residuals.csv', masked_res, delimiter=',')
    # print dimensions
    print('dimensions of masked residuals: ', np.shape(masked_res)) 
    # get the noise for the run and shrink
    noise_pres_res = rsatoolbox.data.noise.prec_from_residuals(masked_res, method='shrinkage_eye') # or shrinkage_diag - get an estimate of noise, dof is n. regressors in glm-1 (but we can ignore this safely here)
    # not working for EC right now as result is a singular matrix for some subjects - not sure why
    # append the masked beta map to the measurements array
    residuals[ind_noise] = noise_pres_res
    ind_noise += 1
    print(f'residuals for run {run} shrunk are:', noise_pres_res)
    # dims should be dataset.n_channel x dataset.n_channel
    print('dimensions of noise_pres_res (noise matrix): ', np.shape(noise_pres_res))
    # check if the noise is all just nans
    if np.isnan(noise_pres_res).all():
        noise_singular = True
# =============================================================================
# 5. Add the residuals to the dataset and calculate neural RDM
# =============================================================================
rdm_cv = rsatoolbox.rdm.calc_rdm(data, method='crossnobis', descriptor='conds', cv_descriptor='sessions', noise=residuals) #residuals

# =============================================================================
# 6. Save the RDM
# =============================================================================

save_dir = glm_results_dir + sub + '/'
save_file = save_dir + f'{sub}{append_str}_{region}-{hemisphere}_rdm_crossnobis.pkl'

rdm_cv.save(save_file, file_type='pkl', overwrite=False)

print('============================================================')
print(f'saved RDM  for subject {sub} in region {region} to: ', save_file)
print('============================================================')

# =============================================================================
# FIN

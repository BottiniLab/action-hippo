import sys
import os
import numpy as np
import nibabel as nib


#sub = 'sub-01'
sub = 'sub-'+sys.argv[1]

append_str = '_stickfunction5vis' #_stickfunction5 for example, or empty


# =============================================================================

def get_euclidean_distances(data):
    """Get the squared euclidean distances between each fold of the data.
    
    Parameters
    ----------
    data : array_like
        An array of shape (n, p) where n is the number of runs and p is
        the number of conditions.
        
    Returns
    -------
    distances : a list of Euclidean distance for each split in the data
    """

    distances = []
    fold_count = 0
    for fold in range(0, data.shape[0]):
        # get the data for this fold
        fold_data = data[fold,:]
        #print('number of conditions in fold is ', len(fold_data))

        # get the data for the other folds
        other_folds = np.delete(data, fold, axis=0)

        # average other folds together
        other_folds = np.mean(other_folds, axis=0)

        #print('number of conditions in train data is ', len(other_folds))
        # compute the euclidean distance between the fold and the other folds
        euclidean_distance = np.linalg.norm(fold_data - other_folds, axis=0)

        distances.append(euclidean_distance)

        fold_count += 1

    #print(f'number of folds is {fold_count}')

    return distances



def permutation_distances(data, n_permutations=10000):
    """Get the squared euclidean distances between each fold of the data using shuffled data to create a null distribution
    
    Parameters
    ----------
    data : array_like
        An array of shape (n, p) where n is the number of runs and p is
        the number of conditions.
        
    Returns
    -------
    distances : a list of Euclidean distance for each split in the data
    """

    # initialise two rngs with the same seed. This lets us check if the permutations are the same
    rng = np.random.default_rng(seed=42)
    rng2 = np.random.default_rng(seed=42)


    # create duplicated data where each entry is a unique integerfier

    duplicated_data = np.copy(data)
    for i in range(0, data.shape[1]):
        duplicated_data[:, i] = i

    # print the duplicated data
    #print(duplicated_data)
    
    full_distances = []
    already_tested = []
    count_already_tested = 0
    for i in range(n_permutations):
        # shuffle all values in the data
        #print(data)
        shuffled_data = rng.permuted(data, axis=1)
        #print(shuffled_data)
        #print(duplicated_data)
        shuffles_repeat = rng2.permuted(duplicated_data, axis=1) 
        #print(shuffles_repeat)
        check_list = np.copy(shuffles_repeat)
        check_list = list(check_list.flatten())
        # check if we've already tested this permutation using 
        if check_list in already_tested:
            #print('already tested')
            count_already_tested += 1
            
        else:

            distances = get_euclidean_distances(shuffled_data)

            full_distances.append(np.mean(distances))

        # flatten shuffled_data into a list
        shuffles_repeat_list = list(shuffles_repeat.flatten())
        already_tested.append(shuffles_repeat_list)

    print(f'already tested {count_already_tested} out of {n_permutations} permutations')
    return full_distances


# =============================================================================


base_dir = '/data/pt_02747/action_hippo/data/derivatives/first_level/nilearn_glm_runwise/'
roi_dir =  f'/data/pt_02747/action_hippo/data/derivatives/{sub}/anat/roi/'

runs = ['01', '02', '03', '04', '05', '06', '07', '08']
conditions = ['1', '2', '3', '4']


# get beta maps for each run and condition 

beta_maps = []

sub_dir = base_dir + sub + '/'
for run in runs:
    run_dir = sub_dir + 'run-' + run + '/'
    
    condition_betas = []

    for condition in conditions:
        beta_file = f'{sub}_run-{run}_space-T1w{append_str}_contrast-{condition}_stat-effect_statmap.nii.gz'
        beta = nib.load(run_dir + beta_file)

        # get data for this beta map so it is easier to handle
        beta_data = beta.get_fdata()

        condition_betas.append(beta_data)

    beta_maps.append(condition_betas)

beta_maps = np.array(beta_maps)

print(beta_maps.shape)

# for each voxel, run euclidean distance analysis

# new empty array to store euclidean distances, same dimensions as a beta map

euclidean_distances = np.zeros(beta_maps[0][0].shape)

pvals = np.zeros(beta_maps[0][0].shape)

# loop through each voxel

for i in range(euclidean_distances.shape[0]):
    for j in range(euclidean_distances.shape[1]):
        for k in range(euclidean_distances.shape[2]):

            # if every value is the same, let's assign distance 0 and p 1 and skip
            if len(np.unique(beta_maps[:, :, i, j, k])) == 1:
                euclidean_distances[i, j, k] = 0
                pvals[i, j, k] = 1

            else:
                # get the beta maps for this voxel
                voxel_betas = beta_maps[:, :, i, j, k]
                print('shape should be 8,4, actual shape is: ', voxel_betas.shape) 

                # let's do a shuffle of our voxel betas to see if our pvals are meaningful
                # this was a sanity check, which worked; pvals were around .5
                #rng = np.random.default_rng(seed=41)
                #voxel_betas_new = rng.permuted(voxel_betas, axis=0)
                #voxel_euclidean_distances = get_euclidean_distances(voxel_betas_new)

                # get the euclidean distances for this voxel
                voxel_euclidean_distances = get_euclidean_distances(voxel_betas)
                # average the euclidean distances for this voxel
                voxel_euclidean_distance = np.mean(voxel_euclidean_distances)
                # save the euclidean distance for this voxel
                euclidean_distances[i, j, k] = voxel_euclidean_distance

                # get the pval for this distance using a permutation method to create a null distribution

                null_distribution = permutation_distances(voxel_betas, n_permutations=1000)

                # get the pval for this voxel: i.e. percentage of null distribution that is less than the actual distance
                pval = sum(null_distribution <= voxel_euclidean_distance) / len(null_distribution)

                pvals[i, j, k] = pval

                print('voxel: ', i, j, k)
                print('euclidean distance: ', voxel_euclidean_distance)
                print('pval: ', pval)



# save the euclidean distances to a nifti file
euclidean_distances_img = nib.Nifti1Image(euclidean_distances, beta.affine, beta.header)

# save the pvals to a nifti file
pvals_img = nib.Nifti1Image(pvals, beta.affine, beta.header)

# create new folder in roi drive called 'voxel_reliability'

if not os.path.exists(roi_dir + 'voxel_reliability'):
    os.makedirs(roi_dir + 'voxel_reliability')

# save the euclidean distances image to a file

euclidean_distances_file = roi_dir + 'voxel_reliability/' + f'{sub}_space-T1w{append_str}_euclidean_distances.nii.gz'
nib.save(euclidean_distances_img, euclidean_distances_file)

# save the pvals image to a file

pvals_file = roi_dir + 'voxel_reliability/' + f'{sub}_space-T1w{append_str}_pvals_euclidean_reliability.nii.gz'
nib.save(pvals_img, pvals_file)


# plot and save null distribution
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(null_distribution, kde=True)

# save
plt.savefig(roi_dir + 'voxel_reliability/' + f'{sub}_space-T1w{append_str}_null_distribution_euclidean_reliability.png')

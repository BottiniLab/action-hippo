
import sys
#prepend_str = sys.argv[1]
prepend_str = ''

#contrast_s = ['modulationState', 'modulationAffordance','modulationAffordanceMagnitude','modulationNumericalDifference','modulationMagnitude','stimulus',
#              'visSame', 'postProbe']

append_str = ''
smoothing_ = '5'
#prepend_str = '1A_'

contrast_s = ['scalarcode']

for contrast_ in contrast_s:

    print('='*50)
    print(f'Running contrast {contrast_}')
    print('='*50)
    # TO DO: convert files to MNI space before running this!!!

    # =============================================================================
    import os
    import json
    from glob import glob
    import pandas as pd
    import numpy as np

    from nilearn.glm.second_level import SecondLevelModel
    import nibabel as nib
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from nilearn import plotting, image


    dataset_dir = '/data/pt_02747/action_hippo/data/'
    data_dir = '/data/pt_02747/action_hippo/data/derivatives/'
    output_dir_first = '/data/pt_02747/action_hippo/data/derivatives/first_level/nilearn_glm/'
    output_dir_second = '/data/pt_02747/action_hippo/data/derivatives/second_level/nilearn_glm/'
    working_dir = '/data/pt_02747/action_hippo/data/derivatives/working_dir/'

    # get all subjects
    subs = [os.path.basename(f) for f in glob(data_dir + 'sub*')]
    subs = [sub for sub in subs if '.' not in sub]

    # get run labels
    runs = ['run-0' + str(i) for i in range(1,9)]


    mask_gm = '/data/pt_02747/tpl-MNI152NLin6Asym_res-25_label-GM_03.nii.gz'



    from nilearn.glm.second_level import non_parametric_inference


    models = []
    count = 0
    for sub in subs:
        # get filename
        display_orient = 'z'
        file_ = output_dir_first + f'{sub}/{sub}_smoothing-{smoothing_}_space-MNI152NLin6Asym{append_str}_contrast-{contrast_}_stat-t_statmap.nii.gz'
        # extract nifti file from filenames
        try:
            nifti_ = nib.load(file_)
        except:
            print(f'file {file_} not found')
            continue
        # append to list
        models.append(nifti_)
        count += 1
    #print('models are: ',models)


    # =============================================================================

    second_level_input = models
    design_matrix = pd.DataFrame(
        [1] * len(second_level_input),
        columns=[contrast_],
    )

    # =============================================================================
    # create a mask for small volume correction
    # =============================================================================

    # to add a mask in we add mask_img=mask_img to SecondLevelModel()
    # in this case let's use a random HC one and see how it looks

    # really we need to map our MPM map from T1w to MNI space 
    # and then use that as a mask
    # in this case we can just use the MNI152NLin6Asym brain

    # load juelich atlas in MNI space
    '''juelich_atlas_1mm_file = '/data/pt_02747/action_hippo/juelich_atlas/MPMs_FSL/JulichBrainAtlas_3.0_areas_MPM_b_N10_MNI152NLin6Asym.nii.gz'
    # load mask
    juelich_atlas_1mm = nib.load(juelich_atlas_1mm_file)
    # downsample to 2mm
    from nilearn.image import resample_to_img
    # take one of our maps as a reference
    ref_img = second_level_input[0]
    # resample to 2.5mm
    juelich_atlas_2mm = resample_to_img(juelich_atlas_1mm_file, ref_img, interpolation='nearest')
    # for MTL atlas, take voxels with values of ... *drumroll* 23, 24, 25, 26, 27 28 and 1023 to 1028, inclusive
    # get mask
    mtl_values = [23, 24, 25, 26, 27, 28, 1023, 1024, 1025, 1026, 1027, 1028] # HPC and EC
    #mtl_values = [25, 1025] # EC
    #mtl_values = [74, 75, 1074, 1075] # mPFC
    mtl_values = [1134, 1135, 1136]
    # check if voxel values are in mtl_values, create binary map
    mask = np.isin(juelich_atlas_2mm.get_fdata(), mtl_values)
    # turn True and False into 1 and 0
    mask = mask.astype(int)
    # convert to nifti
    mask_nii = nib.Nifti1Image(mask, affine=juelich_atlas_2mm.affine, header=juelich_atlas_2mm.header)
    # plot
    plotting.plot_roi(mask_nii, title='Mask')'''
    # =============================================================================
    # use parametric inference
    # =============================================================================


    from nilearn.glm.second_level import SecondLevelModel

    second_level_model = SecondLevelModel(
        #mask_img = mask_nii
        mask_img = mask_gm
        )
    second_level_model = second_level_model.fit(
        second_level_input,
        design_matrix=design_matrix,
    )

    # apply correction using bonferroni correction
    from nilearn.image import get_data, math_img

    zmap = second_level_model.compute_contrast(output_type="z_score")
    nib.save(zmap, output_dir_second + f'{prepend_str}zmap_{contrast_}_twosided.nii.gz')


    p_val = second_level_model.compute_contrast(output_type="p_value")



    # =============================================================================
    # get accurate GM mask, as template includes voxels which are excluded in subject
    # specific GM masks
    # =============================================================================

    # non-paametric inference function is strange
    # take the mask only where the pval map is not 0.5, as this accounts for our GM mask
    # looking a bit different in T1w space and MNI space
    # (not quite sure why the MNI space one includes the brainstem)

    # intersect mask with pval map
    mask_mult = math_img('img1 * img2', img1=mask_gm, img2=p_val)

    # filter to take value which are not 0.5
    mask_mult_data = mask_mult.get_fdata()
    mask_mult_data[mask_mult_data == 0.5] = 0
    mask_mult_data[mask_mult_data != 0] = 1

    # turn into nifti image
    mask_mult_nii = nib.Nifti1Image(mask_mult_data, affine=mask_mult.affine, header=mask_mult.header)

    # plot
    plotting.plot_roi(mask_mult_nii, title='Mask')

    mask_gm = mask_mult_nii

    # =============================================================================

    #n_voxels = np.sum(get_data(second_level_model.masker_.mask_img_))
    n_voxels = np.sum(get_data(mask_gm))

    # plot second_level_model.masker_.mask_img_
    plotting.plot_roi(second_level_model.masker_.mask_img_, title='Mask')



    # save pval map
    nib.save(p_val, output_dir_second + f'{prepend_str}pval_{contrast_}_onesided.nii.gz')

    # Correcting the p-values for multiple testing and taking negative logarithm
    neg_log_pval = math_img(
        f"-np.log10(np.minimum(1, img * {str(n_voxels)}))",
        img=p_val,
    )
    print('this many voxels: ' , n_voxels)

    # save z_map
    nib.save(neg_log_pval, output_dir_second + f'{prepend_str}logp_bonferroni_{contrast_}_onesided.nii.gz')

    print('corrected logp map saved')

    # =============================================================================
    # use non-parametric inference
    # =============================================================================

    print('Running non-parametric inference')

    out_dict = non_parametric_inference(
        second_level_input,
        design_matrix=design_matrix,
        mask=mask_gm,
        model_intercept=True,
        n_perm=10000,  # 500 for the sake of time. Ideally, this should be 10,000.
        two_sided_test=True,
        n_jobs=-1,
        verbose=1,
        tfce=True
    )

    # =============================================================================
    # save output

    # create subfolder

    tfce = out_dict['tfce']
    #t = out_dict['t']
    logp_max_tfce = out_dict['logp_max_tfce']
    #logp_max_t = out_dict['logp_max_t']

    # save tfce
    nib.save(tfce, output_dir_second + f'{prepend_str}tfce_{contrast_}_twosided.nii.gz')

    print('tfce saved')

    # save t
    #nib.save(t, output_dir_second + f'{prepend_str}t_{contrast_}_onesided.nii.gz')

    #print('t saved')

    # save logp_max_tfce
    nib.save(logp_max_tfce, output_dir_second + f'{prepend_str}logp_max_tfce_{contrast_}_twosided.nii.gz')

    print('logp_max_tfce saved')

    # save logp_max_t
    #nib.save(logp_max_t, output_dir_second + f'{prepend_str}logp_max_t_{contrast_}_onesided.nii.gz')

    #print('logp_max_t saved')

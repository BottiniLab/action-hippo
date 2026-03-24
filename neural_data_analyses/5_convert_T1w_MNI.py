# code to create a version of the GLM results for each subject in MNE space for group-level analyses
# file to run from within the nipype environment with ANTS enabled; from the MPI cbs system this is a mess - getserver -sL / ANTSENV / 
# conda activate nipype / python3 /data/pt_02747/action_hippo/code/5_convert_T1w_MNI.py modulation_affordance (phew!)
# using ANTS 2.3.5 via a nipype wrapper

from nipype.interfaces.ants import ApplyTransforms
from os.path import join as opj
import numpy as np
import nibabel as nib
import sys
import os
from shutil import copyfile

contrast = sys.argv[1] # e.g. 'modulationAffordance'
#contrast = 'modulation_affordance'

# Set paths
glm_output_dir = '/data/pt_02747/action_hippo/data/derivatives/first_level/nilearn_glm/'
data_path = '/data/pt_02747/action_hippo/data/derivatives/'

#mni_reference_file = '/data/pt_02747/tpl-MNI152NLin2009aAsym_res-1_T1w.nii.gz/'

# ======================================================================================================================
# get subject list and juelich atlas
# ======================================================================================================================

# list data directory
subs = os.listdir(data_path)
# only take folders with 'sub-' at start
subs = [x for x in subs if x.startswith('sub-')]
# exclude anything with '.' in
subs = [x for x in subs if '.' not in x]

# sort subs
subs.sort()

# get glm betas for modulation effect


mni_reference_file = '/data/pt_02747/tpl-MNI152NLin6Asym_res-25_T1w.nii.gz' # any run is fine
# sub-60_task-probe_run-06_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz

print(subs)

# ======================================================================================================================
# loop through subjects and create T1w image of Juelich atlas
# ======================================================================================================================

for sub in subs:
    # we need to go into the /anat/ folder
    sub_path = opj(data_path, sub, 'anat')
    # get transformation matrix
    transform_file = sub_path + f'/{sub}_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5'
    if sub == 'sub-23':
        transform_file = sub_path + f'/sub-23_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5' # let's see if it works ...

    # get mni reference image - use func file as it will be the correct size etc

    # get list of files in glm output directory
    sub_glm_path = f'{glm_output_dir}/{sub}/'
    glm_output_files = os.listdir(sub_glm_path)

    files_to_convert = [x for x in glm_output_files if f'_smoothing-5_space-T1w_contrast-{contrast}_stat-t' in x and 'nii.gz' in x]


    # loop through glm results files
    for file in files_to_convert:
        input_file = sub_glm_path + file

        # create output file name
        # replace 'nlin2ICBM152asym2009c' with 'T1w'
        output_name = file.replace('T1w', 'MNI152NLin6Asym')

        output_file = sub_glm_path+output_name

        # apply transformation
        at = ApplyTransforms()
        at.inputs.input_image = input_file
        at.inputs.reference_image = mni_reference_file
        at.inputs.transforms = transform_file
        at.inputs.interpolation = 'Linear'
        at.inputs.output_image = output_file
        at.cmdline
        print(at.cmdline)
        print('='*50)
        print('Running ANTS')
        print('='*50)
        at.run()
    print('='*50)
    print(f'Done ANTS for {sub}')

    


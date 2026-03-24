# code to create a version of the Juelich atlas for each subject in their own native T1w space. By using NN interpolation we can retain the original label values.
# file to run from within the nipype environment with ANTS enabled; from the MPI cbs system this is a mess - getserver -sL / ANTSENV / conda activate nipype / python3 script.py (phew!)
# using ANTS 2.3.5 via a nipype wrapper

from nipype.interfaces.ants import ApplyTransforms
from os.path import join as opj
import numpy as np
import nibabel as nib
import sys
import os
from shutil import copyfile

# Set paths
juelich_path = '/data/pt_02747/action_hippo/juelich_atlas/MPMs/'
data_path = '/data/pt_02747/action_hippo/data/derivatives/'

# ======================================================================================================================
# get subject list and juelich atlas
# ======================================================================================================================

# list data directory
subs = os.listdir(data_path)
# only take folders with 'sub-' at start
subs = [x for x in subs if x.startswith('sub-')]
# exclude anything with '.' in
subs = [x for x in subs if '.' not in x]

# get juelich atlas
juelich_files = os.listdir(juelich_path)
# we're only interested in the .nii.gz files
juelich_files_atlas = [x for x in juelich_files if x.endswith('.nii.gz')]
juelich_files_xml = [x for x in juelich_files if x.endswith('.xml')]

print(subs)

# ======================================================================================================================
# loop through subjects and create T1w image of Juelich atlas
# ======================================================================================================================

for sub in subs:
    # we need to go into the /anat/ folder
    sub_path = opj(data_path, sub, 'anat')
    # get T1w image, in name format <sub>_desc-preproc_T1w.nii.gz
    anat_file = sub_path + f'/{sub}_desc-preproc_T1w.nii.gz' # this is our reference image for the transformation

    # get transformation matrix
    transform_file = sub_path + f'/{sub}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5'

    # create a new folder for juelich
    juelich_path_sub = opj(sub_path, 'roi', 'juelich')
    # if it doesn't exist, create it
    if not os.path.exists(juelich_path_sub):
        os.makedirs(juelich_path_sub)

    # loop through atlas files
    for juelich in juelich_files_atlas:
        input_file = juelich_path + juelich

        # create output file name
        # replace 'nlin2ICBM152asym2009c' with 'T1w'
        juelich_output = juelich.replace('nlin2ICBM152asym2009c', 'T1w')

        output_file = juelich_path_sub + f'/{sub}_{juelich_output}'

        # apply transformation
        at = ApplyTransforms()
        at.inputs.input_image = juelich_path + juelich
        at.inputs.reference_image = anat_file
        at.inputs.transforms = transform_file
        at.inputs.interpolation = 'NearestNeighbor'
        at.inputs.output_image = output_file
        at.cmdline
        print(at.cmdline)
        print('='*50)
        print('Running ANTS')
        print('='*50)
        at.run()
    print('='*50)
    print(f'Done ANTS for {sub}')

    # loop through xml files
    for juelich_label in juelich_files_xml:
        input_file = juelich_path + juelich_label

        # create output file name
        # replace 'nlin2ICBM152asym2009c' with 'T1w'
        juelich_output = juelich_label.replace('nlin2ICBM152asym2009c', 'T1w')

        output_file = juelich_path_sub + f'/{sub}_{juelich_output}'

        # copy input file to output file
        copyfile(input_file, output_file)
    


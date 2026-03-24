# create from act_hippo environment

import sys
atlas = sys.argv[1] # form hippocampus_entorhinal etc.
hemisphere = sys.argv[2]

# e.g. 
'''

python3 /data/pt_02747/action_hippo/code/extras/create_rois.py 4a_4p bilateral
python3 /data/pt_02747/action_hippo/code/extras/create_rois.py 4a_4p left
python3 /data/pt_02747/action_hippo/code/extras/create_rois.py 4a_4p right

python3 /data/pt_02747/action_hippo/code/extras/create_rois.py PhG bilateral
python3 /data/pt_02747/action_hippo/code/extras/create_rois.py PhG left
python3 /data/pt_02747/action_hippo/code/extras/create_rois.py PhG right

'''


#atlas = 'hOc1'
#hemisphere = 'bilateral'


# if name has ' ' in it, change to an underscore for saving
if ' ' in atlas:
    atlas_savename = atlas.replace(' ', '_')
else:
    atlas_savename = atlas

import os
import pandas as pd
import numpy as np
import nilearn
import nibabel as nib
from os.path import join as opj
import xml.etree.ElementTree as ET
from nilearn.image import resample_to_img
import json


data_drive = '/data/pt_02747/action_hippo/data/derivatives/'

subs = os.listdir(data_drive)
subs = [sub for sub in subs if 'sub' in sub]
# only take folders
subs = [sub for sub in subs if os.path.isdir(os.path.join(data_drive, sub))]

subs.sort()

mask_voxels = []

for sub in subs:

    atlas_drive = opj(data_drive, sub, 'anat', 'roi')
    juelich_drive = opj(atlas_drive, 'juelich')
    juelich_b_file = opj(juelich_drive, f'{sub}_JulichBrainAtlas_3.0_areas_MPM_b_N10_T1w.nii.gz')
    juelich_b_labels = opj(juelich_drive, f'{sub}_JulichBrainAtlas_3.0_areas_MPM_b_N10_T1w.xml')

    gm_prob_mask = opj(data_drive, sub, 'anat', f'{sub}_label-GM_probseg.nii.gz')

    # resize to functional run size - actual run doesn't matter, but we can use run 5 for fun
    func_run_file = opj(data_drive, sub, 'func', f'{sub}_task-probe_run-05_space-T1w_desc-preproc_bold.nii.gz')


    # load xml file and look inside
    tree = ET.parse(juelich_b_labels)
    root = tree.getroot()

    # get all labels
    labels = {}
    for child in root:
        for subchild in child:
            #print(subchild.text)
            #print(subchild.attrib)
            labels[subchild.text] = [subchild.attrib['leftgrayvalue'], subchild.attrib['rightgrayvalue']]

    # get all labels as a list; makes it easier to search
    label_list = list(labels.keys())

    atlas_names = atlas.split('_')  # split atlas name into parts

    correct_labels = []
    for atlas_name in atlas_names:
        # get all labels that contain the atlas name
        correct_labels += [label for label in label_list if atlas_name.lower() in label.lower()]

    # find which codes to filter for in our atlas
    if hemisphere.lower() == 'left':
        roi_codes = [labels[label][0] for label in correct_labels]
    elif hemisphere.lower() == 'right':
        roi_codes = [labels[label][1] for label in correct_labels]
    elif hemisphere.lower() == 'bilateral':
        roi_codes = [labels[label][0] for label in correct_labels]
        roi_codes.extend([labels[label][1] for label in correct_labels])
    else:
        raise ValueError('hemisphere must be left, right, or bilateral')


    # filter atlas for correct codes

    # load atlas
    atlas_img = nib.load(juelich_b_file)
    atlas_data = atlas_img.get_fdata()

    # filter using roi_codes (feels like there is a better way to do this?)
    atlas_data_filtered = np.zeros(atlas_data.shape)
    for roi_code in roi_codes:
        atlas_data_filtered[atlas_data == int(roi_code)] = 1

    # turn atlas back into a nifti image
    atlas_img_filtered = nib.Nifti1Image(atlas_data_filtered, atlas_img.affine, atlas_img.header)



    resized_atlas = resample_to_img(atlas_img_filtered, func_run_file, interpolation='nearest')
    resized_gm = resample_to_img(gm_prob_mask, func_run_file, interpolation='continuous')

    # threshold both at 0.5
    resized_atlas_data = resized_atlas.get_fdata()
    resized_gm_data = resized_gm.get_fdata()
    resized_gm_data[resized_gm_data < 0.5] = 0
    resized_gm_data[resized_gm_data >= 0.5] = 1

    # combine both masks by multiplying them
    combined_mask = resized_atlas_data * resized_gm_data

    # turn into nifti image
    combined_mask_img = nib.Nifti1Image(combined_mask, resized_atlas.affine, resized_atlas.header)

    # check number of voxels 
    mask_voxels.append(np.sum(combined_mask))



    # save mask to file
    out_path = opj(data_drive, sub, 'anat', 'roi', f'{atlas_savename}-{hemisphere}')

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    out_file = opj(out_path, f'{sub}_mask-{atlas_savename}_{hemisphere}.nii.gz')
    nib.save(combined_mask_img, out_file)

    print(f'saved {out_file}')


# save mask voxels to file
out_file = f'/data/pt_02747/action_hippo/rois/voxels_mask-{atlas_savename}_{hemisphere}.csv'
with open(out_file, 'w') as f:
    for voxel_sum in mask_voxels:
        f.write(f'{voxel_sum}\n')

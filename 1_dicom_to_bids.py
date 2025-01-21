#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# =============================================================================
# ----------------------- DICOM to BIDS Conversion  ---------------------------
#
# Authors: Theo Schaefer, Marco Badwal, Felix Deilmann, Alex Nitsch
# Date: April 16 2021
# Contact: tschaefer@cbs.mpg.de

dcm2bids                  2.1.9              pyhd8ed1ab_0    conda-forge
dcm2niix                  1.0.20230411         h00ab1b0_0    conda-forge

# code modified June 2023 by Alex Eperon - key changes: create own environment for dcm2bids

# =============================================================================
"""

# =============================================================================
# Setup
# =============================================================================

import os
import json
import pandas as pd
from glob import glob
import shutil


# Settings
bids_convert = True


#fMRI Raw Data should be copied into the following dir_root and labeld with your subject IDs
dir_bids = '/data/pt_02747/BIDS_data'
dir_source =  dir_bids + '/sourcedata/'
dir_json = dir_source
dir_work = '/data/pt_02747/BIDS_data/output'

dir_final = '/data/pt_02747/action_hippo/data/'

dir_backup = '/data/p_02747/raw/bids_data/'
dir_backup_dicom = '/data/p_02747/raw/dicom_data/'

# Define tasks (function needed for more complex procedures)
task = 'probe'


# Which subs are already measured?
subs_source = [os.path.basename(f) for f in glob(dir_backup_dicom + 'sub*')]
subs_source.sort()

subs_source = [sub[:6] for sub in subs_source]

# Which subs already in BIDS format
subs_bids = [os.path.basename(f) for f in glob(dir_final + 'sub*')]
# Check which subs to convert to BIDS
subs = [sub for sub in subs_source if sub not in subs_bids]

#subs = ['sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06']

print("subs are: ", subs)

# oddSubject1 = 'sub-04'
# subs = ['sub-02', 'sub-03', 'sub-04']



# =============================================================================
# Functions
# =============================================================================


def cfg_stack(list_scans):
    """ Reads the scan protocol out of a provided list of scans (based on Scan.txt) 
        and runs the fitting configuration function which creates a dictionary
        based on the scan modality. These dictionaries are stacked together 
        while looping through the list of scans.
        
        Args:
            list_scans: (list) sorted list containing scan labels
            
        Returns: 
            (dict) nested dictionary containing scan descriptions ready to 
            be converted to a json configuration file.
    """
    scan_list = []
    for scan in list_scans:
        if 'bold' in scan: scan_list.append(cfg_func(scan))
        elif 'norm' in scan: scan_list.append(cfg_fmap(scan))
        elif 'invpol' in scan: scan_list.append(cfg_fmap(scan))
        elif 'gre_fieldmap' in scan: scan_list.append('gre_field')
        elif 'MPRAGE' in scan and 'ND' not in scan: scan_list.append(cfg_anat(scan))
        elif 'mp2rage' in scan and "ND" not in scan: scan_list.append(cfg_anat_mp2(scan))

    return {'descriptions':scan_list}


def cfg_func(scan):
    """ Creates a dictionary containing descriptions of a bold scan.
    
    Args:
        scan: (character) scan label
        
    Returns: 
        (dict) nested dictionary containing bold scan descriptions 
    """
    func_dict = {
      "dataType": "func",
      "modalityLabel": "bold",
      "customLabels": f"task-{task}_run-{scan[-1].zfill(2)}",
      "criteria": {
        "SeriesDescription": scan
      },
      "sidecarChanges": {
        "TaskName": task
      }
    }
    return func_dict


def cfg_fmap(scan):
    """ Creates a dictionary containing descriptions of a PEPOLAR fieldmap scan.
    
    Args:
        scan: (character) scan label
        
    Returns: 
        (dict) nested dictionary containing fieldmap scan descriptions 
    """
    fmap_dict = {
      "dataType": "fmap",
      "modalityLabel": "epi",
      "customLabels": "",
      "IntendedFor": [int(scan[-1])-1],
      "criteria": {
        "SeriesDescription": scan
      }
    }

    if "norm" in scan:
        fmap_dict["customLabels"] = f"dir-norm_run-{scan[-1].zfill(2)}"
    elif "invpol" in scan:
        fmap_dict["customLabels"] = f"dir-invpol_run-{scan[-1].zfill(2)}"

    return fmap_dict








def cfg_anat(scan):
    """ Creates a dictionary containing descriptions of MPRAGE anatomical scan.
    
    Args:
        scan: (character) scan label
        
    Returns: 
        (dict) nested dictionary containing MPRAGE scan descriptions 
    """
    anat_dict = {
      "dataType": "anat",
      "modalityLabel": "T1w",
      "suffix": "",
      "criteria": {
        "SeriesDescription": scan
      }
    }
    return anat_dict


def cfg_anat_mp2(scan):
    """ Creates a dictionary containing descriptions of MP2RAGE anatomical scan.
        Depending on the exact scan label it changes the modality description.
    
    Args:
        scan: (character) scan label
        
    Returns: 
        (dict) nested dictionary containing MP2RAGE scan descriptions 
    """
    anat_dict = {
      "dataType": "anat",
      "modalityLabel": "",
      "suffix": "",
      "criteria": {
        "SeriesDescription": scan
      }
    }
    if 'INV1' in scan:
        anat_dict['modalityLabel'] = 'T1inv1'
        anat_dict['suffix'] = "T1inv1"
    elif 'INV2' in scan:
        anat_dict['modalityLabel'] = 'T1inv2'
        anat_dict['suffix'] = "T1inv2"
    elif 'T1_Images' in scan:
        anat_dict['modalityLabel'] = 'T1map'
        anat_dict['suffix'] = "T1map"
    elif 'UNI' in scan:
        anat_dict['modalityLabel'] = 'T1w'
        anat_dict['suffix'] = "T1w"  #suffix ggf. nicht notwendig

    return anat_dict






def assign_tasks(subject, session):
    """ Assigns task label to bold runs of a particular subject and session.
        Has to be costumized if task-run assignment is more complex.

    Args:
        subject: full subject id (e.g. sub-01)
        sesssion: full session id (e.g. ses-01)
        
    Returns: 
        (str) task label  
    """
    
    return 'probe'


# =============================================================================
# Configuration files and bids conversion
# =============================================================================


# Create working directory (only needed for bids conversion)
if bids_convert:
    if not os.path.exists(dir_work): os.mkdir(dir_work)


# Loop through subjects and sessions
for sub in subs:

    print(f'Processing {sub}...')
    print('---------------------')
    print('copying dicom files to source directory')
    # copy sub from dir_backup_dicom to dir_source using os
    if not os.path.exists(os.path.join(dir_source, sub)):
        shutil.copytree(os.path.join(dir_backup_dicom, sub), os.path.join(dir_source, sub))



    # Get sessions
    task = 'probe'
    # global task
    
    # Path of Scans.txt file for subject
    fn_scans = glob(os.path.join(dir_source, f'{sub}/', '*.txt'))

    # List of scans based on txt file
    list_scans = pd.read_table(fn_scans[0], header=None).values.tolist()
    list_scans = [scan[0][5:] for scan in list_scans]

    # Sort scan list
    list_scans.sort()

    # Loop through scan list and write respective json chuncks in a dictionary
    cfg = cfg_stack(list_scans)

    # Save dictionary as json file
    fn_cfg = os.path.join(dir_json, sub, f'{sub}_bids-config.json')
    with open(fn_cfg,'w') as outfile:
        json.dump(cfg, outfile, indent='\t')


    # ----------------------------------------------------------------------
    # bids conversion (only runs if set to True)
    if bids_convert:
        
        # create subject/session directory
        dir_inp = os.path.join(dir_source, sub, 'DICOM')

        # Apply dcm2bids function with inputs (no need to load conda environment if you start from there - life hack :)) life hack was lie
        os.system(f"""
        dcm2bids -d {dir_inp} -p {sub[4:]} -c {fn_cfg} -o {dir_bids} --forceDcm2niix
        """)

        # Remove tmp directory
        try: 
            if os.path.exists(glob(dir_bids + '/tmp*')[0]):
                
                shutil.rmtree(glob(dir_bids + '/tmp*')[0])  
        except: print('tmp folder does not exist')

        # Create bidsignore file
        if not os.path.isfile(f'{dir_bids}/.bidsignore'):
            bidsignore = open(f'{dir_bids}/.bidsignore','w+')
            bidsignore.write('sourcedata/*\ncode/*'); bidsignore.close()



        print('===========================================')
        print(f'   finished conversion: {sub}     ')
        print('===========================================')



    sub_i = sub[4:]

    # create folder for subject in dir_bids

    if not os.path.exists(dir_bids+'/sub-'+sub_i):
        os.mkdir(dir_bids+'/sub-'+sub_i)
        
            # Write bash script for mp2rage FSL
    with open (dir_bids+'/sub-'+sub_i+'/mp2rage_denoise_'+sub_i+'.sh', 'w') as mp2rage_denoise:
        mp2rage_denoise.write(f'''\
    #!/usr/bin/env bash 
    #FSL
    #cd {dir_bids} 
    root_dir={dir_bids} 
    mkdir $root_dir/sub-{sub_i}/anat/tmp
    echo $root_dir 
    # 
    #  
    echo "Processing subject:" {sub_i} "..."

    #move everything to tmp
    mv $root_dir/sub-{sub_i}/anat/sub-{sub_i}_T1w.nii.gz $root_dir/sub-{sub_i}/anat/tmp
    mv $root_dir/sub-{sub_i}/anat/sub-{sub_i}_T1w.json $root_dir/sub-{sub_i}/anat/tmp
    mv $root_dir/sub-{sub_i}/anat/sub-{sub_i}_T1map.nii.gz $root_dir/sub-{sub_i}/anat/tmp
    mv $root_dir/sub-{sub_i}/anat/sub-{sub_i}_T1map.json $root_dir/sub-{sub_i}/anat/tmp
    mv $root_dir/sub-{sub_i}/anat/sub-{sub_i}_T1inv2.nii.gz $root_dir/sub-{sub_i}/anat/tmp
    mv $root_dir/sub-{sub_i}/anat/sub-{sub_i}_T1inv2.json $root_dir/sub-{sub_i}/anat/tmp
    mv $root_dir/sub-{sub_i}/anat/sub-{sub_i}_T1inv1.nii.gz $root_dir/sub-{sub_i}/anat/tmp
    mv $root_dir/sub-{sub_i}/anat/sub-{sub_i}_T1inv1.json $root_dir/sub-{sub_i}/anat/tmp
    #
    #rename files first 
    echo "removing surrounding noise in T1w for " {sub_i} "..." 
    cp $root_dir/sub-{sub_i}/anat/tmp/sub-{sub_i}_T1w.nii.gz $root_dir/sub-{sub_i}/anat/tmp/sub-{sub_i}_T1w_noise.nii.gz 
    cp $root_dir/sub-{sub_i}/anat/tmp/sub-{sub_i}_T1w.json $root_dir/sub-{sub_i}/anat/tmp/sub-{sub_i}_T1w_noise.json 
    #
    #step 1: add inv2 and gamma 
    fslmaths $root_dir/sub-{sub_i}/anat/tmp/sub-{sub_i}_T1inv2.nii.gz -add 100 $root_dir/sub-{sub_i}/anat/tmp/sub-{sub_i}_T1inv2_gamma.nii.gz 
    # 
    #step2: divide inv2/(inv2+gamma) 
    fslmaths $root_dir/sub-{sub_i}/anat/tmp/sub-{sub_i}_T1inv2.nii.gz -div $root_dir/sub-{sub_i}/anat/tmp/sub-{sub_i}_T1inv2_gamma.nii.gz $root_dir/sub-{sub_i}/anat/tmp/sub-{sub_i}_T1inv2_div_inv2_gamma.nii.gz 
    #
    #step 3: multiply t1w * (inv2/(inv2+gamma)) 
    fslmaths $root_dir/sub-{sub_i}/anat/tmp/sub-{sub_i}_T1w_noise.nii.gz -mul $root_dir/sub-{sub_i}/anat/tmp/sub-{sub_i}_T1inv2_div_inv2_gamma.nii.gz $root_dir/sub-{sub_i}/anat/tmp/sub-{sub_i}_T1w.nii.gz 
    #
    #step 4: clean up
    mv $root_dir/sub-{sub_i}/anat/tmp/sub-{sub_i}_T1w.nii.gz $root_dir/sub-{sub_i}/anat 
    mv $root_dir/sub-{sub_i}/anat/tmp/sub-{sub_i}_T1w.json $root_dir/sub-{sub_i}/anat 

    #
    exit
                ''')
        
    os.system(f'''\
                FSL \
                    sh {dir_bids}/sub-{sub_i}/mp2rage_denoise_{sub_i}.sh''')
            


    print('===========================================')
    print(f'   created T1w for: {sub}     ')
    print('===========================================')

    os.system(f'''\
                rm {dir_bids}/sub-{sub_i}/mp2rage_denoise_{sub_i}.sh''')

    # copy subject to dir_backup

    os.system(f'''\
                cp -R {dir_bids}/sub-{sub_i} {dir_backup}''')

    print('===========================================')
    print(f'   copied subject: {sub}     ')
    print('===========================================')

    # remove tmp from anat folder

    os.system(f'''\
                rm -R {dir_bids}/sub-{sub_i}/anat/tmp''')
    
    print('===========================================')
    print(f'   removed tmp folder for: {sub}     ')
    print('===========================================')

    # move subject to dir_final

    os.system(f'''\
                mv {dir_bids}/sub-{sub_i} {dir_final}''')
    
    print('===========================================')
    print(f'   moved subject: {sub}     ')
    print('===========================================')


# empty dir_source

os.system(f'''\
            rm -R {dir_source}*''')

print('===========================================')
print(f'   removed all subjects from {dir_source}     ')
print('===========================================')




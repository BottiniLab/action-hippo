"""
# =============================================================================
# -------------------------- Run fmriprep  ---------------------------
#
# AUTHORS: Theo Schaefer, Marco Badwal, Felix Deilmann, Alex Nitsch
# DATE: April 22 2021
#
# INSTRUCTIONS
#WRITE THAT IN TERMINAL (chemi damwyebi komentari)
# 1) connect to g6 server: getserver -sL -g6 (meore varianti: ssh comps06h04)
# 2) execute script: python3 /data/pt_02459/AS_fmri/data/code/preprocessing/fmriprep/fmriprep_MyNew.py
# *) check status: condor_q   
                   condor_status -submitters
#                  
# *) remove job: condor_rm JOB-ID

# I can also directly run on the current server, to see if it throws errors and which errors it throws: sh /data/pt_02747/action_hippo/code/preprocessing/fmriprep/run_fmriprep_sub-01.sh

# FOR SLURM

# 1) connect to submission server: getserver -sb
# 2) execute script: python3 /data/pt_02747/action_hippo/code/2_fmriprep_acthippo.py



# =============================================================================
"""

# =============================================================================
# Setup
# =============================================================================

import os
import json
from glob import glob





dir_proj    = '/data/pt_02747/action_hippo/'
dir_bids    = dir_proj+ 'data/'



subs = [os.path.basename(f) for f in glob(dir_bids + 'sub*')]
subs_done = [os.path.basename(f) for f in glob(dir_bids + 'derivatives/sub*')]

subs = [sub for sub in subs if sub not in subs_done]

#subs = ['sub-01', 'sub-02', 'sub-03', 'sub-06', 'sub-07', 'sub-08']


print("subs are: ", subs)


# extra code to be sure that we are using the exact fieldmap for each run

for sub in subs:

    
    sub_num = sub.split("-")[-1]

    dir_func    = dir_bids + sub + '/func/'
    dir_fmap    = dir_bids + sub + '/fmap/'

    func_jsons = glob(dir_func + '*_bold.json')
    fmap_jsons = glob(dir_fmap + '*_epi.json')

    print("func_jsons are: ", func_jsons)
    print("fmap_jsons are: ", fmap_jsons)

    for func_json in func_jsons:
        run_num = func_json.split("_")[-2][-2:]

        with open(func_json, 'r') as f:
            data = json.load(f)
            data["B0FieldSource"] = f"pepolarfmap{run_num}{sub_num}"
            print('B0 field source is: ', data["B0FieldSource"])
        with open(func_json, 'w') as f:
            json.dump(data, f, indent=4)

    for fmap_json in fmap_jsons:
        run_num = fmap_json.split("_")[-2][-2:]
        with open(fmap_json, 'r') as f:
            data = json.load(f)
            data["B0FieldIdentifier"] = f"pepolarfmap{run_num}{sub_num}"
            print('B0 field identifier is: ', data["B0FieldIdentifier"])
        with open(fmap_json, 'w') as f:
            json.dump(data, f, indent=4)






#fMRI Raw Data should be copied into the following dir_root and labeld with your subject IDs
dir_proj    = '/data/pt_02747/action_hippo/'
dir_bids    = dir_proj+ 'data/'
dir_code    = dir_proj + 'code/preprocessing/fmriprep/'
dir_deriv   = dir_bids + 'derivatives/'
dir_prep    = dir_deriv + 'fmriprep/'
dir_work    = dir_proj + 'fmriprep_work/'






# Define subjects for which to run fmriprep
# either choose only those subjects which are not preprocessed yet
# or define subjects manually
# Which subs already in BIDS format
#subs_bids = [os.path.basename(f) for f in glob(dir_bids + 'sub*')]
# Which subs are already preprocessed
#subs_prep = [os.path.basename(f) for f in glob(dir_prep + 'sub*')]
# Check which subs to process with fmriprep
#subs = [sub for sub in subs_bids if sub not in subs_prep]

# Define subjects manually, e.g.
#subs=['sub-18', 'sub-19', 'sub-20', 'sub-21', 'sub-22', 'sub-23', 'sub-24', 'sub-25', 'sub-26', 'sub-27', 'sub-28']
#subs=['sub-46']
#subs=['sub-03']


# Get all subjects
# subs = subs_bids

# Create working directory
if not os.path.exists(dir_work): os.makedirs(dir_work + 'condor_log/')

# Create code directory
if not os.path.exists(dir_code): os.makedirs(dir_code)

# Create derivatives directory
if not os.path.exists(dir_deriv): os.makedirs(dir_deriv)

print('-------------------------------------------')
print(f'Processing {len(subs)} subjects: {subs}')
print('-------------------------------------------')

# =============================================================================
# Loop through subjects
# =============================================================================

for sub in subs:
    
    sleep_amount = sub[-1:]

    # Subject specific fmriprep working directory
    dir_sub_work = f'{dir_work}fmriprep_wf/single_subject_{sub.split("-")[1]}_wf/'  # alternative to sub[4:] is sub.split('-')[1]

    # Write fmriprep command file
    with open (dir_code+f'run_fmriprep_{sub}.sh', 'w') as f_fmriprep:
        f_fmriprep.write(f'''\
#! /bin/bash


echo 'Clearing fmriprep working directory...'
# also remove the corresponding work directory if it has not been removed yet
rm -rf {dir_sub_work}

# note: no fsnative or fsaverage surfaces are generated due to slurm errors (memory?)

sleep {sleep_amount}m
singularity run --cleanenv -B \\
{dir_bids},\\
{dir_deriv},\\
{dir_work},\\
/afs/cbs/software/freesurfer/ \\
/data/p_SoftwareServiceLinux_sc/fmriprep/22.0.1/1 \\
{dir_bids} \\
{dir_deriv} \\
participant --participant-label {sub.split("-")[1]} \\
--output-spaces T1w MNI152NLin6Asym \\
--dummy-scans 0 \\
--fs-license-file /afs/cbs/software/freesurfer/licensekeys  --fs-no-reconall \\
-w {dir_sub_work} --clean-workdir \\
--write-graph --stop-on-first-crash --notrack --verbose

# Remove fmriprep working directory
echo 'Fmriprep done. Removing fmriprep working directory...'
rm -rf {dir_sub_work}

''')
        
        
# the best space is MNI152NLin2009cAsym, but fmriprep outputs the affine for this anyway

#when I run fmriprep directly on the current server with sh (is written above), then it was finished successfully, but I has many many warnings starting with: nipype.workflow.warning: could not retrieve profiling information for node ...
#from internet: this is because I executed the script wutg tge --resource-monitor flag enabled. This is telling me that resource monitoring (i.e., profiling the node) did not work. One advises not to use this flag unless I am really getting into the weeds of resource management. 
#this profiler should not have any influence on the results, and this warning would not be a good reason not to trust the outputs
#previous input: --write-graph --stop-on-first-crash --notrack --verbose --resource-monitor


#SBATCH --executable = /afs/cbs.mpg.de/software/scripts/envwrap
#rewrite for slurm this way?
#SBATCH --chdir /afs/cbs.mpg.de/software/scripts/envwrap

    # Write condor submit bash file
    with open (dir_code+f'slurm_submit_{sub}.sh', 'w') as f_condor:
        f_condor.write(f'''#!/bin/bash

#SBATCH --ntasks 1

#SBATCH -c 28                                 

#SBATCH --mem-per-cpu 12G

#SBATCH --time 840                     

#SBATCH -o /data/pt_02747/action_hippo/code/preprocessing/slurm_output/%j.out        # redirect the output of our job to the given file

#SBATCH -e /data/pt_02747/action_hippo/code/preprocessing/slurm_output/%j.err        # redirect stderr to the given file

echo "Number of nodes:$SLURM_JOB_NUM_NODES"

srun {dir_code}run_fmriprep_{sub}.sh 
''')
#srun {dir_code}run_fmriprep_{sub}.sh is the code I need to execute

    # submit job to condor
    os.system(f'''
        chmod a+rwx {dir_code}run_fmriprep_{sub}.sh
        chmod a+rwx {dir_code}slurm_submit_{sub}.sh
        sbatch {dir_code}slurm_submit_{sub}.sh
        ''')


    print('-------------------------------------------')
    print(f'Submitted fmriprep job for subject: {sub}')
    print('-------------------------------------------')





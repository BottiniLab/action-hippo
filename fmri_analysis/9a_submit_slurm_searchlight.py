# FOR SLURM

# 1) connect to submission server: getserver -sb
# 2) execute script: python3 /data/pt_02747/action_hippo/code/10a_submit_slurm_searchlight.py _stickfunction5vis
# add _partials to the end of file name to run with partial correlation, but only after already did neural RDMs

# take contrast information from the command line argument
# this allows us to be lightning-fast when we want to run the same script with different contrasts
# "your script becomes fast as lightning", cit. Kung Fu Panda 

import sys

append_str = sys.argv[1] #_stickfunction5vis

import os
import json
from glob import glob

data_dir = '/data/pt_02747/action_hippo/data/derivatives/first_level/nilearn_glm_runwise/'
dir_code = '/data/pt_02747/action_hippo/code/nilearn/'


subs = [os.path.basename(f) for f in glob(data_dir + 'sub*')]
#subs = ["sub-07", "sub-32", 'sub-21', 'sub-57', 'sub-10', 'sub-62', 'sub-07']  #32, 21, 57, 10, 62, 07

subs = ['sub-10', 'sub-57']

sub_ids = [sub[-2:] for sub in subs if '.' not in sub]


for sub_id in sub_ids:

    # Write glm command file
    with open (dir_code+f'sub-{sub_id}_searchlight{append_str}.sh', 'w') as f_sl:
        f_sl.write(f'''\
#! /bin/bash
/data/u_eperon_software/anaconda3/envs/act_hippo_rsa/bin/python /data/pt_02747/action_hippo/code/10_single_subject_searchlight.py {sub_id} {append_str}

''')
        

    # Write slurm file
    with open (dir_code+f'sub-{sub_id}_searchlight{append_str}.slurm', 'w') as f_slurm:
        f_slurm.write(f'''\
#!/bin/bash
#SBATCH --job-name=sl_{sub_id}
                      #SBATCH --ntasks 1

#SBATCH -c 4                                 

#SBATCH --mem-per-cpu 10G

#SBATCH --time 360

#SBATCH -o /data/pt_02747/action_hippo/code/nilearn/slurm_output/%j.out        # redirect the output of our job to the given file

#SBATCH -e /data/pt_02747/action_hippo/code/nilearn/slurm_output/%j.err        # redirect stderr to the given file

echo "Number of nodes:$SLURM_JOB_NUM_NODES"

srun {dir_code}sub-{sub_id}_searchlight{append_str}.sh
''')
        

    # submit job to condor
    os.system(f'''
        chmod a+rwx {dir_code}sub-{sub_id}_searchlight{append_str}.sh
        chmod a+rwx {dir_code}sub-{sub_id}_searchlight{append_str}.slurm
        sbatch {dir_code}sub-{sub_id}_searchlight{append_str}.slurm
        ''')


    print('-------------------------------------------')
    print(f'Submitted searchlight job for subject sub-{sub_id}')
    print('-------------------------------------------')


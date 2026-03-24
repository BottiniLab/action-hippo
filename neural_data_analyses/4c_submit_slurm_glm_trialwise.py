# FOR SLURM

# 1) connect to submission server: getserver -sb
# 2) execute script: python3 /data/pt_02747/action_hippo/code/4c_submit_slurm_glm_trialwise.py 1_2_3_4


# take contrast information from the command line argument
# this allows us to be lightning-fast when we want to run the same script with different contrasts
# "your script becomes fast as lightning", cit. Kung Fu Panda 

import sys
contrast_ = sys.argv[1]

import os
import json
from glob import glob

data_dir = '/data/pt_02747/action_hippo/data/derivatives/'
dir_code = '/data/pt_02747/action_hippo/code/nilearn/'


subs = [os.path.basename(f) for f in glob(data_dir + 'sub*')]
#subs = ["sub-39"]

sub_ids = [sub[-2:] for sub in subs if '.' not in sub]


for sub_id in sub_ids:

    # Write glm command file
    with open (dir_code+f'sub-{sub_id}_run-glm_trialwise_contrast-{contrast_}.sh', 'w') as f_glm:
        f_glm.write(f'''\
#! /bin/bash
/data/u_eperon_software/anaconda3/envs/act_hippo/bin/python /data/pt_02747/action_hippo/code/4_run_trialwise_glm_single_subject.py {sub_id} {contrast_}

''')
        

    # Write slurm file
    with open (dir_code+f'sub-{sub_id}_run-glm_trialwise_contrast-{contrast_}.slurm', 'w') as f_slurm:
        f_slurm.write(f'''\
#!/bin/bash
#SBATCH --job-name=glm_contrast-{contrast_}
                      #SBATCH --ntasks 1

#SBATCH -c 4                                 

#SBATCH --mem-per-cpu 4G

#SBATCH --time 60                     

#SBATCH -o /data/pt_02747/action_hippo/code/nilearn/slurm_output/%j.out        # redirect the output of our job to the given file

#SBATCH -e /data/pt_02747/action_hippo/code/nilearn/slurm_output/%j.err        # redirect stderr to the given file

echo "Number of nodes:$SLURM_JOB_NUM_NODES"

srun {dir_code}sub-{sub_id}_run-glm_trialwise_contrast-{contrast_}.sh
''')
        

    # submit job to condor
    os.system(f'''
        chmod a+rwx {dir_code}sub-{sub_id}_run-glm_trialwise_contrast-{contrast_}.sh
        chmod a+rwx {dir_code}sub-{sub_id}_run-glm_trialwise_contrast-{contrast_}.slurm
        sbatch {dir_code}sub-{sub_id}_run-glm_trialwise_contrast-{contrast_}.slurm
        ''')


    print('-------------------------------------------')
    print(f'Submitted glm job for subject: sub-{sub_id}')
    print('-------------------------------------------')


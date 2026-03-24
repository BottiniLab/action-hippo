# FOR SLURM

# 1) connect to submission server: getserver -sb
# 2) execute script: 
'''
python3 /data/pt_02747/action_hippo/code/7a_submit_slurm_neural_rdms.py region hemisphere

python3 /data/pt_02747/action_hippo/code/7a_submit_slurm_neural_rdms.py 4a_4p bilateral
python3 /data/pt_02747/action_hippo/code/7a_submit_slurm_neural_rdms.py 4a_4p left
python3 /data/pt_02747/action_hippo/code/7a_submit_slurm_neural_rdms.py 4a_4p right

python3 /data/pt_02747/action_hippo/code/7a_submit_slurm_neural_rdms.py PhG bilateral
python3 /data/pt_02747/action_hippo/code/7a_submit_slurm_neural_rdms.py PhG left
python3 /data/pt_02747/action_hippo/code/7a_submit_slurm_neural_rdms.py PhG right

'''

# take contrast information from the command line argument
# this allows us to be lightning-fast when we want to run the same script with different contrasts
# "your script becomes fast as lightning", cit. Kung Fu Panda 

import sys
region_ = sys.argv[1] # note: unlike in create_rois.py, this is a string with a single region name, not a list
hemisphere_ = sys.argv[2] #note: unlike in create_rois.py, this is a string with a single hemisphere name, not a list

import os
import json
from glob import glob

data_dir = '/data/pt_02747/action_hippo/data/derivatives/first_level/nilearn_glm_runwise/'
dir_code = '/data/pt_02747/action_hippo/code/nilearn/'


subs = [os.path.basename(f) for f in glob(data_dir + 'sub*')]
#subs = ["sub-07", "sub-32", 'sub-21', 'sub-57', 'sub-10', 'sub-62', 'sub-07']  #32, 21, 57, 10, 62, 07

#subs = ['sub-62']

sub_ids = [sub[-2:] for sub in subs if '.' not in sub]




for sub_id in sub_ids:

    # Write glm command file
    with open (dir_code+f'sub-{sub_id}_rdm_region-{region_}-{hemisphere_}.sh', 'w') as f_glm:
        f_glm.write(f'''\
#! /bin/bash
/data/u_eperon_software/anaconda3/envs/act_hippo_rsa/bin/python /data/pt_02747/action_hippo/code/7_create_neural_rdms.py {sub_id} {region_} {hemisphere_}

''')
        

    # Write slurm file
    with open (dir_code+f'sub-{sub_id}_rdm_region-{region_}-{hemisphere_}.slurm', 'w') as f_slurm:
        f_slurm.write(f'''\
#!/bin/bash
#SBATCH --job-name=rdm_sub{sub_id}_region-{region_}-{hemisphere_}
                      #SBATCH --ntasks 1

#SBATCH -c 4                                 

#SBATCH --mem-per-cpu 2G

#SBATCH --time 30                     

#SBATCH -o /data/pt_02747/action_hippo/code/nilearn/slurm_output/%j.out        # redirect the output of our job to the given file

#SBATCH -e /data/pt_02747/action_hippo/code/nilearn/slurm_output/%j.err        # redirect stderr to the given file

echo "Number of nodes:$SLURM_JOB_NUM_NODES"

srun {dir_code}sub-{sub_id}_rdm_region-{region_}-{hemisphere_}.sh
''')
        

    # submit job to condor
    os.system(f'''
        chmod a+rwx {dir_code}sub-{sub_id}_rdm_region-{region_}-{hemisphere_}.sh
        chmod a+rwx {dir_code}sub-{sub_id}_rdm_region-{region_}-{hemisphere_}.slurm
        sbatch {dir_code}sub-{sub_id}_rdm_region-{region_}-{hemisphere_}.slurm
        ''')


    print('-------------------------------------------')
    print(f'Submitted rdm job for subject sub-{sub_id} in region {region_}')
    print('-------------------------------------------')


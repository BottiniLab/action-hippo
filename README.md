# action-hippo
Code repository for the action hippo project


Code is split into separate directories to analyse behavioural, fMRI and eyetracking data. Files are named sequentially, allowing a straightforward replication by following the ordered files. 

All code assumes adherence to the BIDS format for fMRI data; behavioural and eyetracking data is saved separately in a pseudo-BIDS format. For details see code comments.

Code was mostly run on a compute server (SLURM). Code necessary for this is labelled with a letter and 'submit_slurm', but can easily be adapted for use on any personal compute server.

For any further information, feel free to contact Alex (alex.eperon@gmail.com). 

## fMRI analysis

1. Convert raw files to BIDS format and move to working directories
2. Preprocess using fmriprep
3. Create events files and ROIs for future analyses; define event files based on planned analysis (RSA)
4. Run first-level GLM for 4 main conditions using nilearn
5. Create a subject-specific version of the Juelich atlas mamimum probability maps to use for ROI analysis
6. Segment the Juelich atlas into ROIs
7. Create neural RDMs
8. Compare model RDMs in entorhinal ROIs, excluding the effects of other models using partial correlation
9. Run a searchlight analysis in subject space (T1W)
10. Convert searchlight maps to MNI space
11. Run cluster correction to check for significant clusters in searchlight maps
12. Use a permutation-based method to create whole-brain voxel reliability maps for a single subject
13. Create intersected ROIs using the voxel reliability maps created in the previous step


## eye analysis

0. Data preprocessing; blink removal
1. Create an events file to categorise data by condition
2. Visualise data and create a big dataframe organised by subject and condition
3. Test if eye movements are skewed right or left

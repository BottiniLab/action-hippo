# action-hippo
Code repository for the action hippo project


Code is split into separate directories to analyse behavioural, fMRI and eyetracking data. Files are named sequentially, allowing a straightforward replication by following the ordered files.

All code assumes adherence to the BIDS format for fMRI data; behavioural and eyetracking data is saved separately in a pseudo-BIDS format. For details see code comments.

For any further information, feel free to contact Alex (alex.eperon@gmail.com). 

## fMRI analysis

1. Convert raw files to BIDS format and move to working directories
2. Preprocess using fmriprep
3. Create events files and ROIs for future analyses; define event files based on planned analysis (adaptation or RSA)
4. Run first-level GLM for 4 main conditions using nilearn




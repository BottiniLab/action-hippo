
**Paper title:**  
*Action information is integrated into entorhinal representations of conceptual space and is reflected in eye movements*  

---

Code is split into separate directories to analyse behavioural, fMRI and eyetracking data. Files are named sequentially, allowing a straightforward replication by following the ordered files. File paths are relative to the local user, and will need changing to relevant local directories.  

All code assumes adherence to the BIDS format for fMRI data; behavioural and eyetracking data is saved separately in a pseudo-BIDS format. For details see code comments.  

Code was mostly run on a compute server (SLURM). Code necessary for this is labelled with a letter and 'submit_slurm', but can easily be adapted for use on any personal compute server or run as individual files using the 'sys.argv' commands from terminal.  


## fMRI analysis  

1. Convert raw files to BIDS format and move to working directories  
2. Preprocess using fmriprep  
3. Create events files and ROIs for future analyses; define event files based on planned analysis (RSA)  
4. Run first-level GLM for 4 main conditions using nilearn  
5. Create a subject-specific version of the Juelich atlas mamimum probability maps to use for ROI analysis  
6. Segment the Juelich atlas into ROIs  
7. Create neural RDMs  
8. Compare model RDMs in entorhinal ROIs, excluding the effects of other models using partial correlation  
10. Run searchlight analyses 
11. Convert searchlight maps to MNI space  
12. Run cluster correction to check for significant clusters in searchlight maps    
13. Code to predict eye position using deepMReye toolbox (Frey, Nau and Doeller, 2021)  
14. Code to extract events from deepMReye-predicted gaze positions for use in further analyses  

---

## eye analysis  

0. Data preprocessing; blink removal  
1. Create an events file to categorise data by condition  
2. Visualise data and create a big dataframe organised by subject and condition  
3. Test if eye movements are skewed right or left in x and y  
4. Test if deepMReye-predicted eye movements are skewed left or right in x  
5. Test if deepMReye-predicted eye movements are skewed left or right in y  
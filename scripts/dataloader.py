#!/usr/bin/env python
# coding: utf-8
# In[2]:


import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
import os

# ### Data Paths

# In[14]:

data_dir = Path('/mnt/ssd3/ronan/satterthwaite')

bblid_path = data_dir / 'matched_data' / 'matched_bblids.csv'

fmri_dir = data_dir / 'matched_data' / 'REST_data'
fmri_metadata_path = data_dir / 'raw' / 'fMRI' / 'SubjectsIDs_Schaefer_rest_Alone.csv'

sc_dir = data_dir / 'matched_data' / 'Deterministic_FA'
sc_metadata_path = data_dir / 'raw' / 'structural' / 'SubjectsIDs_Schaefer_Diffusion.csv'

demographics_path = data_dir / 'raw' / 'Demographics_MedicalFilter.csv'
clinical_path = data_dir / 'raw' / 'Cognitive_MedicalFilter.csv'
goa_path = data_dir / 'raw' / 'GOA_imputed_MedicalFilter.csv'
schaefer_coords_path = data_dir / 'supplementary' / 'Schaefer2018_400Parcels_17Networks_order.txt'

# ### Load Metadata

# In[15]:

bblids = np.genfromtxt(bblid_path, delimiter=',').astype(int)
fmri_metadata = pd.read_csv(fmri_metadata_path)
sc_metadata = pd.read_csv(sc_metadata_path)
demographics = pd.read_csv(demographics_path)
clinical_scores = pd.read_csv(clinical_path)
goa_scores = pd.read_csv(goa_path)

# ### Load fMRI Scans
# In[38]:
fmri_dict = {int(f.split('_')[0]): f for f in os.listdir(fmri_dir)}

# In[51]:
sc_dict = {int(f.split('_')[0]): f for f in os.listdir(sc_dir)}

def get_files():
    fmri_files = [np.genfromtxt(fmri_dir / fmri_dict[bblid], delimiter=',') for bblid in bblids]
    sc_files = [np.genfromtxt(sc_dir / sc_dict[bblid], delimiter=',') for bblid in bblids]
    goa_matched = [goa_scores.query(f'bblid == {bblid}').to_numpy()[0,1:-6] for bblid in bblids]
    
    return(bblids,fmri_files,sc_files,goa_matched) 
        
def get_covariates():
    ## Sex
    sexes = [demographics.query(f'bblid == {bblid}')['sex'].iloc[0] for bblid in bblids]
    ## Age
    ages = [demographics.query(f'bblid == {bblid}')['ageAtCnb1'].iloc[0] for bblid in bblids]
    ## RMS Motion
    fmri_rms_motion = [fmri_metadata.query(f'bblid == {bblid}')['restRelMeanRMSMotion'].iloc[0] for bblid in bblids]
    sc_rms_motion = [sc_metadata.query(f'bblid == {bblid}')['dti64MeanRelRMS'].iloc[0] for bblid in bblids]

    return((sexes, ages), (fmri_rms_motion, sc_rms_motion))
#%%
if __name__ == "__main__":
    pass






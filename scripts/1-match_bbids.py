#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
import os
from tqdm import tqdm


# ### Data Paths

# In[14]:


data_dir = Path('/mnt/ssd3/ronan/satterthwaite')

fmri_dir = data_dir / 'raw' / 'fMRI' / 'REST_Data'
fmri_metadata_path = data_dir / 'raw' / 'fMRI' / 'SubjectsIDs_Schaefer_rest_Alone.csv'

sc_dir = data_dir / 'raw' / 'structural' / 'Deterministic_FA'
sc_metadata_path = data_dir / 'raw' / 'structural' / 'SubjectsIDs_Schaefer_Diffusion.csv'

demographics_path = data_dir / 'raw' / 'Demographics_MedicalFilter.csv'
clinical_path = data_dir / 'raw' / 'Cognitive_MedicalFilter.csv'
goa_path = data_dir / 'raw' / 'GOA_imputed_MedicalFilter'
schaefer_coords_path = data_dir / 'supplementary' / 'Schaefer2018_400Parcels_17Networks_order.txt'


# ### Load Metadata

# In[15]:


fmri_metadata = pd.read_csv(fmri_metadata_path)
sc_metadata = pd.read_csv(sc_metadata_path)
demographics = pd.read_csv(demographics_path)
clinical_scores = pd.read_csv(clinical_path)


# ### Load fMRI Scans
# In[38]:


fmri_files = [np.genfromtxt(fmri_dir / f, delimiter=' ') for f in os.listdir(fmri_dir)]
fmri_ids = [f.split('_')[0] for f in os.listdir(fmri_dir)]
fmri_bblids = [fmri_metadata.query(f'scanid == {idx}')['bblid'].iloc[0] for idx in fmri_ids]


# In[51]:


names = None
sc_bblids = [int(f.split('_')[0]) for f in os.listdir(sc_dir)]
sc_files = []

for f in os.listdir(sc_dir):
    # one-liner to read a single variable
    names = loadmat(sc_dir / f)['name'][0]
    sc_files.append(loadmat(sc_dir / f)['connectivity'])


# ## Get Matched Indices

# In[52]:


def get_matched_indices(x,y):
    """
    Returns indices of all shared elements of two lists
    """
    matches = set(x).intersection(set(y))
    return(([x.index(i) for i in matches], [y.index(i) for i in matches]))


# In[53]:


idx_fmri, idx_sc = get_matched_indices(fmri_bblids, sc_bblids)


# In[83]:


fmri_mats_matched = [fmri_files[i] for i in idx_fmri]
fmri_bblids_matched = [fmri_bblids[i] for i in idx_fmri]

sc_mats_matched = [sc_files[i] for i in idx_sc]
sc_bblids_matched = [sc_bblids[i] for i in idx_sc]

if not len(np.where(not sc_bblids_matched == fmri_bblids_matched)[0]) == 0:
    print(f'There are {len(np.where(not sc_bblids_matched == fmri_bblids_matched)[0])} mismatches')

bblids_matched = sc_bblids_matched
# %%
### Save matched files

save_dir = data_dir / 'matched_data'
fmri_save_dir = save_dir / 'REST_data'
sc_save_dir = save_dir / 'Deterministic_FA'

for directory in [fmri_save_dir, sc_save_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

for mat,bblid in zip(fmri_mats_matched, bblids_matched):
    np.savetxt(fmri_save_dir / f'{bblid}_Schaefer400_network.csv', mat, delimiter=",")

for mat,bblid in zip(sc_mats_matched, bblids_matched):
    np.savetxt(sc_save_dir / f'{bblid}_SchaeferPNC_400.csv', mat, delimiter=",")

np.savetxt(save_dir / 'matched_bblids.csv', bblids_matched.astype(int), delimiter=',')

# %%

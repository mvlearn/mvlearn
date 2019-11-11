
# ## Load Covariates
# In[84]:
import numpy as np
import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
import dataloader
from sklearn.linear_model import LinearRegression
import pickle

# In[172]:

bblids,fmri_files,sc_files,goa_matched = dataloader.get_files()
(sexes, ages), (fmri_rms_motion, sc_rms_motion) = dataloader.get_covariates()

fmri_covariates = np.vstack([sexes, ages, fmri_rms_motion]).T
sc_covariates = np.vstack([sexes, ages, sc_rms_motion]).T
# ## Regress out Covariates: voxel-wise

# In[185]:


def regress_from_voxel(X, y, return_coef=False):
    """
    y : List of voxels across subjects
    X : list of covariates
    
    returns : list of voxels w/ covariates regressed out
    """
    reg = LinearRegression().fit(X, y)
    
    if return_coef:
        return(reg.coef_)
    else:
        return(y - reg.predict(X))

# ## Load and Regress fMRI scans

# In[190]:

## fmri coefficients
data_dir = Path('/mnt/ssd3/ronan/satterthwaite/regressed_out')

def get_coeffs(covariates, files, n_covariates=3):
    """
    Return regression coefficients
    """
    coeffs = np.zeros((400,400,n_covariates))

    for i in tqdm(range(400)):
        for j in range(400):
            coeff = regress_from_voxel(covariates, [file[i,j] for file in files], return_coef=True)
            coeffs[i,j,:] = coeff

    return(coeffs)

## fMRI
coeffs = get_coeffs(fmri_covariates, fmri_files)

with open(data_dir / 'regression_coeffs_mat.pkl', "wb") as output_file:
    pickle.dump(coeffs, output_file)

for bblid, file, covariate in tqdm(zip(bblids, fmri_files, fmri_covariates)):
    regressed_out = file - coeffs @ covariate
    np.savetxt(data_dir / 'REST_data' / f'{bblid}_Schaeffer400_network_regressed-covariates.csv', regressed_out, delimiter=',')


## Structural
coeffs = get_coeffs(sc_covariates, sc_files)

with open(data_dir / 'regression_coeffs_mat.pkl', "wb") as output_file:
    pickle.dump(coeffs, output_file)

for bblid, file, covariate in tqdm(zip(bblids, sc_files, sc_covariates)):
    regressed_out = file - coeffs @ covariate
    np.savetxt(data_dir / 'Deterministic_FA' / f'{bblid}_SchaeferPNC_400_regressed-covariates.csv', regressed_out, delimiter=',')


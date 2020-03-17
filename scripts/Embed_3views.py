#%%
from multiview.embed.gcca import GCCA
from sklearn.utils import check_array
import numpy as np
from pathlib import Path
import pandas as pd
import dataloader
from datetime import datetime
import pickle

#%%
n = -1
bblids,fmri_files,sc_files,clinical = dataloader.get_files(n=n)

n = fmri_files.shape[0]
uidx = np.triu_indices(fmri_files.shape[1])

Xs = [np.reshape([file[uidx] for file in fmri_files], (n,-1)), 
      np.reshape([file[uidx] for file in sc_files], (n,-1)),
      clinical.reshape(n,-1)]

gcca = GCCA(n_elbows=2)
Xs_projs = gcca.fit_transform(Xs)

print(f'Ranks are {gcca.ranks_}')

save_dir = Path('/mnt/ssd3/ronan/satterthwaite/embedding')
time = datetime.now().strftime("%j-%H:%M")

names = ['fmri', 'sc', 'clinical']
for X,name in zip(Xs_projs,names):
    np.savetxt(save_dir / f'{name}_gcca-n={n}_T={time}.csv', X, delimiter=',')

with open(save_dir / f'gcca_object_T={time}.pkl', 'wb') as handle:
    pickle.dump(gcca, handle)

for mat,name in zip(gcca.projection_mats_,names):
    np.savetxt(save_dir / f'{name}_gcca_projection_mat_T={time}.csv', mat, delimiter=',')